using Label2Train.Train.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Training.Detr;

/// <summary>
/// DETR 集合损失：匈牙利匹配 → 分类 sigmoid BCE + L1（归一化 cxcywh）+ GIoU。
/// <para>
/// 与 DETR 原版的差异：分类用逐类 sigmoid BCE（没有 no-object 类），
/// 与模型的 <c>[B,Q,4+nc]</c> 输出和后续 ONNX 导出（sigmoid）保持一致。
/// </para>
/// </summary>
public sealed class DetrLoss : ILoss
{
    private readonly int _classCount;
    private readonly int _imageWidth;
    private readonly int _imageHeight;
    private readonly DetrLossOptions _options;
    private readonly int _groupDetr;

    // 分项累积（加权），由 AverageParts / ResetParts 读写
    private float _classPart;
    private float _l1Part;
    private float _giouPart;
    private int _partSteps;

    /// <summary>上一次 <see cref="Compute"/> 实际匹配到的（query, gt）对数：诊断用。</summary>
    public int LastMatchedPairs { get; private set; }

    /// <summary>按类别数与（用于把像素框归一化的）输入尺寸创建损失。</summary>
    /// <param name="classCount">类别数。</param>
    /// <param name="imageWidth">输入宽（像素）。</param>
    /// <param name="imageHeight">输入高（像素）。</param>
    /// <param name="options">损失权重。</param>
    /// <param name="groupDetr">Group-DETR 组数：把 query 均分成 G 组，每组独立做一对一匹配（官方 13，默认 1）。</param>
    public DetrLoss(
        int classCount,
        int imageWidth,
        int imageHeight,
        DetrLossOptions? options = null,
        int groupDetr = 1)
    {
        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "类别数必须大于 0。");
        if (imageWidth <= 0 || imageHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageWidth), imageWidth, "输入尺寸必须大于 0。");
        if (groupDetr <= 0)
            throw new ArgumentOutOfRangeException(nameof(groupDetr), groupDetr, "Group-DETR 组数必须大于 0。");

        _classCount = classCount;
        _imageWidth = imageWidth;
        _imageHeight = imageHeight;
        _options = options ?? new DetrLossOptions();
        _groupDetr = groupDetr;
    }

    /// <summary>
    /// 匹配损失 + CDN 去噪损失：前 <paramref name="matchingQueries"/> 个 query 走匈牙利匹配，
    /// 其余按「第 i 个去噪 query ↔ 第 i 个加噪目标」固定配对（<c>Mask=false</c> 的填充项跳过，
    /// 标签为 -1 的翻类样本只回传框）。
    /// </summary>
    public Tensor ComputeWithDenoising(
        Tensor prediction,
        Tensor target,
        int matchingQueries,
        CdnBatch cdn)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(cdn);

        if (matchingQueries <= 0 || matchingQueries >= prediction.shape[1])
        {
            throw new ArgumentException(
                $"matchingQueries 必须在 (0,{prediction.shape[1]}) 内，实际 {matchingQueries}。",
                nameof(matchingQueries));
        }

        Tensor baseLoss = Compute(prediction.narrow(1, 0, matchingQueries), target);

        long batch = prediction.shape[0];
        Tensor denoisePart = prediction.narrow(1, matchingQueries, cdn.Count);
        Tensor predictedBoxes = denoisePart.narrow(2, 0, 4);
        Tensor logits = denoisePart.narrow(2, 4, _classCount);
        Tensor cdnBoxes = cdn.ToBoxes(batch, prediction.device);

        var valid = new List<long>();
        var classTargets = new float[batch * cdn.Count * _classCount];

        long index = 0;
        for (int image = 0; image < batch; image++)
        {
            for (int item = 0; item < cdn.Count; item++, index++)
            {
                if (!cdn.Mask[index])
                    continue;

                valid.Add(index);

                int label = cdn.Labels[index];
                if (label >= 0)
                    classTargets[index * _classCount + Math.Clamp(label, 0, _classCount - 1)] = 1f;
            }
        }

        Tensor scoreTargets = TorchInterop.FromArray(
            classTargets,
            new long[] { batch, cdn.Count, _classCount },
            prediction.device);

        Tensor classLoss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            scoreTargets,
            reduction: torch.nn.Reduction.None).mean();

        Tensor boxLoss;
        Tensor giouLoss;

        if (valid.Count == 0)
        {
            boxLoss = torch.zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: prediction.device);
            giouLoss = torch.zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: prediction.device);
        }
        else
        {
            Tensor selection = torch.tensor(
                valid.ToArray(), new long[] { valid.Count }, device: prediction.device);

            Tensor predicted = predictedBoxes.reshape(-1, 4).index_select(0, selection);
            Tensor truth = cdnBoxes.reshape(-1, 4).index_select(0, selection);

            boxLoss = (predicted - truth).abs().sum(new long[] { 1 }).mean();
            giouLoss = (1f - Giou(predicted, truth)).mean();
        }

        return baseLoss
            + classLoss * _options.ClassWeight
            + boxLoss * _options.L1Weight
            + giouLoss * _options.GiouWeight;
    }

    /// <summary>最近一次的分项值（未乘权重，调试用）。</summary>
    public (float Class, float L1, float Giou) LastParts { get; private set; }

    /// <summary>
    /// 自上次 <see cref="ResetParts"/> 起各分项的**加权**平均（分类 / L1 / GIoU）。
    /// 三项相加即 epoch 平均 loss 的构成（去噪 / 两阶段的额外项不计入），
    /// 用来分辨「分类项压不下来」还是「框回归没收敛」。
    /// </summary>
    public (float Class, float L1, float Giou) AverageParts =>
        _partSteps == 0
            ? (0f, 0f, 0f)
            : (_classPart / _partSteps, _l1Part / _partSteps, _giouPart / _partSteps);

    /// <summary>清空分项累积（每个 epoch 评估结束后由训练循环调用）。</summary>
    public void ResetParts()
    {
        _classPart = 0f;
        _l1Part = 0f;
        _giouPart = 0f;
        _partSteps = 0;
    }

    /// <inheritdoc />
    public string Name => "rf-detr-set-loss";

    /// <inheritdoc />
    public Tensor Compute(Tensor prediction, Tensor target)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(target);

        if (prediction.dim() != 3 || prediction.shape[2] != 4 + _classCount)
        {
            throw new ArgumentException(
                $"prediction 必须是 [B,Q,{4 + _classCount}]，实际是 [{string.Join(",", prediction.shape)}]。",
                nameof(prediction));
        }

        if (target.dim() != 3 || target.shape[2] != 6 || target.shape[0] != prediction.shape[0])
        {
            throw new ArgumentException(
                $"target 必须是 [B,M,6] 且 B 与 prediction 一致，实际是 [{string.Join(",", target.shape)}]。",
                nameof(target));
        }

        long batch = prediction.shape[0];
        long queries = prediction.shape[1];
        long maxTargets = target.shape[1];

        Tensor boxes = prediction.narrow(2, 0, 4);
        Tensor logits = prediction.narrow(2, 4, _classCount);

        Tensor mask = target.select(2, 0) > 0.5f;
        Tensor classIds = target.select(2, 1);
        Tensor gtBoxes = PixelToNormalizedCxcywh(target.narrow(2, 2, 4));

        IReadOnlyList<SimpleDetrImageMatch> matches = DetrMatcher.Match(
            boxes.detach(),
            logits.detach(),
            gtBoxes,
            classIds,
            mask,
            _options,
            groups: _groupDetr);

        LastMatchedPairs = matches.Sum(match => match.Pairs.Count);

        var targetScores = new float[batch * queries * _classCount];

        // 正样本在 [B,Q,nc] 里的扁平位置 + 与 GT 的 IoU：IA-BCE 的正样本权重同时用这两者
        var positiveIndices = new List<long>();
        var positiveIous = new List<float>();
        var predictionIndex = new List<long>();
        var targetIndex = new List<long>();
        float[] classes = TorchInterop.ToArray(classIds.to(ScalarType.Float32));

        float[] predictedBoxesFlat = TorchInterop.ToArray(boxes);
        float[] targetBoxesFlat = TorchInterop.ToArray(gtBoxes);

        for (int image = 0; image < batch; image++)
        {
            foreach ((int query, int targetId) in matches[image].Pairs)
            {
                int classId = Math.Clamp((int)classes[image * maxTargets + targetId], 0, _classCount - 1);

                float iou = IntersectionOverUnion(
                    predictedBoxesFlat, (int)((image * queries + query) * 4),
                    targetBoxesFlat, (int)((image * maxTargets + targetId) * 4));

                targetScores[((image * queries + query) * _classCount) + classId] = 1f;

                positiveIndices.Add(((image * queries) + query) * _classCount + classId);
                positiveIous.Add(iou);

                predictionIndex.Add(image * queries + query);
                targetIndex.Add(image * maxTargets + targetId);
            }
        }

        Tensor scoreTargets = TorchInterop.FromArray(
            targetScores,
            new long[] { batch, queries, _classCount },
            prediction.device);

        // 分类项：正负样本同权、除以正样本数（DETR 的 num_boxes 口径），详见 ClassificationLoss
        Tensor classLoss = ClassificationLoss(
            logits, scoreTargets, positiveIndices, positiveIous, predictionIndex.Count);

        Tensor boxLoss;
        Tensor giouLoss;

        if (predictionIndex.Count == 0)
        {
            boxLoss = torch.zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: prediction.device);
            giouLoss = torch.zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: prediction.device);
        }
        else
        {
            Tensor predictionSelection = torch.tensor(
                predictionIndex.ToArray(), new long[] { predictionIndex.Count }, device: prediction.device);
            Tensor targetSelection = torch.tensor(
                targetIndex.ToArray(), new long[] { targetIndex.Count }, device: prediction.device);

            Tensor predictedBoxes = boxes.view(-1, 4).index_select(0, predictionSelection);
            Tensor matchedTargets = gtBoxes.view(-1, 4).index_select(0, targetSelection);

            boxLoss = (predictedBoxes - matchedTargets).abs().sum(new long[] { 1 }).mean();
            giouLoss = (1f - Giou(predictedBoxes, matchedTargets)).mean();
        }

        float classValue = TorchInterop.ToScalar(classLoss);
        float boxValue = TorchInterop.ToScalar(boxLoss);
        float giouValue = TorchInterop.ToScalar(giouLoss);

        LastParts = (classValue, boxValue, giouValue);

        // 标量已在上面读回，这里只做乘加，不额外同步设备
        _classPart += classValue * _options.ClassWeight;
        _l1Part += boxValue * _options.L1Weight;
        _giouPart += giouValue * _options.GiouWeight;
        _partSteps++;

        return classLoss * _options.ClassWeight
            + boxLoss * _options.L1Weight
            + giouLoss * _options.GiouWeight;
    }

    /// <summary>
    /// 两阶段辅助损失（官方 `enc_out_class_embed` / `enc_out_bbox_embed` 的深监督）：
    /// 对编码器给出的**提议**再做一次"匹配 + 分类 BCE + L1 + GIoU"，按 <paramref name="weight"/> 缩放。
    /// <para>
    /// 实现取巧但等价：提议本身就是 <c>[B,L,4]</c> 框 + <c>[B,L,nc]</c> logits，拼成 <c>[B,L,4+nc]</c>
    /// 后可以直接复用 <see cref="Compute"/> 的整条逻辑（匹配、IA-BCE、L1、GIoU）。
    /// 内部用 <c>groupDetr = 1</c> 的临时实例，避免与主损失的组语义混淆。
    /// </para>
    /// <para>
    /// 与训练器的接线仍待定（需要 <c>ILoss</c> 增加 extras 重载或让模型返回复合结构），
    /// 因此本方法目前是"已实现、已测试、尚未在 <see cref="Compute"/> 里自动调用"。
    /// </para>
    /// </summary>
    /// <param name="proposalBoxes">提议框 <c>[B,L,4]</c>（归一化 cxcywh，通常是 sigmoid 之后）。</param>
    /// <param name="proposalLogits">提议分类 logits <c>[B,L,nc]</c>。</param>
    /// <param name="target">与主损失相同的 GT <c>[B,M,6]</c>（布局 [mask, class, x1, y1, x2, y2]）。</param>
    /// <param name="weight">辅助损失权重（官方常用 0.5~1.0）。</param>
    public Tensor ComputeAuxiliary(
        Tensor proposalBoxes,
        Tensor proposalLogits,
        Tensor target,
        float weight = 0.5f)
    {
        ArgumentNullException.ThrowIfNull(proposalBoxes);
        ArgumentNullException.ThrowIfNull(proposalLogits);
        ArgumentNullException.ThrowIfNull(target);

        if (proposalBoxes.dim() != 3 || proposalBoxes.shape[2] != 4)
        {
            throw new ArgumentException(
                $"proposalBoxes 必须是 [B,L,4]，实际是 [{string.Join(",", proposalBoxes.shape)}]。",
                nameof(proposalBoxes));
        }

        if (proposalLogits.dim() != 3 || proposalLogits.shape[2] != _classCount)
        {
            throw new ArgumentException(
                $"proposalLogits 必须是 [B,L,{_classCount}]，实际是 [{string.Join(",", proposalLogits.shape)}]。",
                nameof(proposalLogits));
        }

        if (weight <= 0f)
            throw new ArgumentOutOfRangeException(nameof(weight), weight, "辅助损失权重必须大于 0。");

        Tensor combined = torch.cat(new[] { proposalBoxes, proposalLogits }, 2);
        var auxiliary = new DetrLoss(_classCount, _imageWidth, _imageHeight, _options, groupDetr: 1);

        return auxiliary.Compute(combined, target) * weight;
    }

    /// <summary>extras 里提议框的键名（<c>[B,L,4]</c>，归一化 cxcywh）。</summary>
    public const string ProposalBoxesKey = "proposal_boxes";

    /// <summary>extras 里提议分类 logits 的键名（<c>[B,L,nc]</c>）。</summary>
    public const string ProposalLogitsKey = "proposal_logits";

    /// <summary>
    /// 覆写 <see cref="ILoss"/> 的 extras 重载：当 extras 同时提供
    /// <see cref="ProposalBoxesKey"/> 与 <see cref="ProposalLogitsKey"/> 时，
    /// 在基准损失上叠加两阶段辅助损失（权重 0.5）；否则等价于普通 <see cref="Compute(Tensor,Tensor)"/>。
    /// </summary>
    public Tensor Compute(
        Tensor prediction,
        Tensor target,
        IReadOnlyDictionary<string, Tensor> extras)
    {
        Tensor baseLoss = Compute(prediction, target);

        if (extras.TryGetValue(ProposalBoxesKey, out Tensor? boxes)
            && extras.TryGetValue(ProposalLogitsKey, out Tensor? logits))
        {
            return baseLoss + ComputeAuxiliary(boxes, logits, target);
        }

        return baseLoss;
    }

    /// <summary>
    /// 分类项，**除以正样本数**归一化（DETR 的 <c>num_boxes</c> 口径）。
    /// <para>
    /// 归一化三种都试过，只有这一种能同时压住负样本、抬得起正样本：
    /// <c>mean(全体元素)</c> 会被 300×nc 个负样本稀释（top1 0.51 → 0.06、mAP 恒 0）；
    /// <c>正负各自求均值</c> 让负样本权重只有正样本的 1/200（未匹配 query 分数 0.97、mAP50 卡 0.10）。
    /// </para>
    /// <para>
    /// <see cref="DetrLossOptions.IouAwareBce"/> 打开时用官方 RF-DETR 的 IA-BCE 公式
    /// （<c>criterion.py::_batched_detection_loss_tensors</c>）：负样本权重 <c>p²</c>，
    /// 正样本权重 <c>clamp(p^α · IoU^(1-α), 0.01)</c>，写成
    /// <c>Σ(neg·x − logσ(x)·(pos+neg)) / num_boxes</c>。
    /// 注意官方**不是**把标签换成 <c>IoU^γ</c> 的软标签 BCE——那样早期 IoU 很小时目标分被压到 0.0x，
    /// 所有 query 的分数都会掉到评估阈值以下。
    /// </para>
    /// </summary>
    /// <param name="logits">分类 logits <c>[B,Q,nc]</c>。</param>
    /// <param name="targets">硬标签（匹配位置为 1，其余 0），仅非 IA-BCE 路径使用。</param>
    /// <param name="positiveIndices">正样本在 <c>[B,Q,nc]</c> 里的扁平位置。</param>
    /// <param name="positiveIous">与 <paramref name="positiveIndices"/> 一一对应的匹配 IoU。</param>
    /// <param name="positiveCount">匹配对数量（归一化分母）。</param>
    internal Tensor ClassificationLoss(
        Tensor logits,
        Tensor targets,
        IReadOnlyList<long> positiveIndices,
        IReadOnlyList<float> positiveIous,
        int positiveCount)
    {
        if (_options.IouAwareBce)
        {
            float alpha = _options.FocalAlpha;

            Tensor flatLogits = logits.reshape(-1);
            Tensor probability = flatLogits.sigmoid();
            Tensor negative = probability.pow(2f);
            Tensor positive = torch.zeros_like(flatLogits);

            if (positiveIndices.Count > 0)
            {
                Tensor selection = torch.tensor(
                    positiveIndices.ToArray(),
                    new long[] { positiveIndices.Count },
                    device: logits.device);

                Tensor iou = torch.tensor(
                    positiveIous.ToArray(),
                    new long[] { positiveIous.Count },
                    device: logits.device);

                // 官方的正样本权重：当前概率与 IoU 的几何插值，clamp 到 0.01 防止梯度全灭
                Tensor score = probability.index_select(0, selection)
                    .pow(alpha)
                    .mul(iou.pow(1f - alpha))
                    .clamp(min: 0.01f)
                    .detach();

                positive.scatter_(0, selection, score);

                using Tensor complement = 1f - score;
                negative.scatter_(0, selection, complement);
            }

            // logσ(x) = -softplus(-x)，比 logsigmoid 在 TorchSharp 里更稳
            Tensor logSigma = torch.nn.functional.softplus(-flatLogits).neg();
            Tensor loss = (negative * flatLogits) - (logSigma * (positive + negative));

            return loss.sum() / Math.Max(1, positiveCount);
        }

        Tensor elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction: torch.nn.Reduction.None);

        if (_options.FocalGamma > 0f)
        {
            // 普通 focal 的调制因子按元素算；归一化仍走 num_boxes 口径
            Tensor probability = logits.sigmoid();
            Tensor probabilityTrue = probability * targets + (1f - probability) * (1f - targets);
            Tensor alphaTrue = _options.FocalAlpha * targets + (1f - _options.FocalAlpha) * (1f - targets);

            elementwise = elementwise * alphaTrue * (1f - probabilityTrue).pow(_options.FocalGamma);
        }

        return elementwise.sum() / Math.Max(1, positiveCount);
    }

    /// <summary>两个归一化 cxcywh 框的 IoU（IA-BCE 的分类目标用）。</summary>
    private static float IntersectionOverUnion(float[] boxes, int boxOffset, float[] targets, int targetOffset)
    {
        float ax1 = boxes[boxOffset] - boxes[boxOffset + 2] / 2f;
        float ay1 = boxes[boxOffset + 1] - boxes[boxOffset + 3] / 2f;
        float ax2 = boxes[boxOffset] + boxes[boxOffset + 2] / 2f;
        float ay2 = boxes[boxOffset + 1] + boxes[boxOffset + 3] / 2f;

        float bx1 = targets[targetOffset] - targets[targetOffset + 2] / 2f;
        float by1 = targets[targetOffset + 1] - targets[targetOffset + 3] / 2f;
        float bx2 = targets[targetOffset] + targets[targetOffset + 2] / 2f;
        float by2 = targets[targetOffset + 1] + targets[targetOffset + 3] / 2f;

        float width = MathF.Max(0f, MathF.Min(ax2, bx2) - MathF.Max(ax1, bx1));
        float height = MathF.Max(0f, MathF.Min(ay2, by2) - MathF.Max(ay1, by1));
        float intersection = width * height;

        float areaA = MathF.Max(0f, ax2 - ax1) * MathF.Max(0f, ay2 - ay1);
        float areaB = MathF.Max(0f, bx2 - bx1) * MathF.Max(0f, by2 - by1);
        float union = areaA + areaB - intersection;

        return union <= 0f ? 0f : intersection / union;
    }

    /// <summary>像素 xyxy → 归一化 cxcywh。</summary>
    private Tensor PixelToNormalizedCxcywh(Tensor pixelBoxes)
    {
        Tensor x1 = pixelBoxes.narrow(2, 0, 1) / _imageWidth;
        Tensor y1 = pixelBoxes.narrow(2, 1, 1) / _imageHeight;
        Tensor x2 = pixelBoxes.narrow(2, 2, 1) / _imageWidth;
        Tensor y2 = pixelBoxes.narrow(2, 3, 1) / _imageHeight;

        return torch.cat(
            new[]
            {
                (x1 + x2) * 0.5f,
                (y1 + y2) * 0.5f,
                torch.clamp(x2 - x1, min: 0f),
                torch.clamp(y2 - y1, min: 0f),
            },
            2);
    }

    /// <summary>归一化 cxcywh 的 GIoU（张量实现，梯度可以回到框）。</summary>
    private static Tensor Giou(Tensor a, Tensor b)
    {
        Tensor aXyxy = CxcywhToXyxy(a);
        Tensor bXyxy = CxcywhToXyxy(b);

        Tensor interX1 = torch.maximum(aXyxy.narrow(1, 0, 1), bXyxy.narrow(1, 0, 1));
        Tensor interY1 = torch.maximum(aXyxy.narrow(1, 1, 1), bXyxy.narrow(1, 1, 1));
        Tensor interX2 = torch.minimum(aXyxy.narrow(1, 2, 1), bXyxy.narrow(1, 2, 1));
        Tensor interY2 = torch.minimum(aXyxy.narrow(1, 3, 1), bXyxy.narrow(1, 3, 1));

        Tensor intersection = torch.clamp(interX2 - interX1, min: 0f)
            * torch.clamp(interY2 - interY1, min: 0f);

        Tensor areaA = torch.clamp(aXyxy.narrow(1, 2, 1) - aXyxy.narrow(1, 0, 1), min: 0f)
            * torch.clamp(aXyxy.narrow(1, 3, 1) - aXyxy.narrow(1, 1, 1), min: 0f);

        Tensor areaB = torch.clamp(bXyxy.narrow(1, 2, 1) - bXyxy.narrow(1, 0, 1), min: 0f)
            * torch.clamp(bXyxy.narrow(1, 3, 1) - bXyxy.narrow(1, 1, 1), min: 0f);

        Tensor union = areaA + areaB - intersection + 1e-9f;

        Tensor encloseWidth = torch.maximum(aXyxy.narrow(1, 2, 1), bXyxy.narrow(1, 2, 1))
            - torch.minimum(aXyxy.narrow(1, 0, 1), bXyxy.narrow(1, 0, 1));

        Tensor encloseHeight = torch.maximum(aXyxy.narrow(1, 3, 1), bXyxy.narrow(1, 3, 1))
            - torch.minimum(aXyxy.narrow(1, 1, 1), bXyxy.narrow(1, 1, 1));

        Tensor enclose = encloseWidth * encloseHeight + 1e-9f;

        return intersection / union - (enclose - union) / enclose;
    }

    private static Tensor CxcywhToXyxy(Tensor boxes) => torch.cat(
        new[]
        {
            boxes.narrow(1, 0, 1) - boxes.narrow(1, 2, 1) * 0.5f,
            boxes.narrow(1, 1, 1) - boxes.narrow(1, 3, 1) * 0.5f,
            boxes.narrow(1, 0, 1) + boxes.narrow(1, 2, 1) * 0.5f,
            boxes.narrow(1, 1, 1) + boxes.narrow(1, 3, 1) * 0.5f,
        },
        1);
}
