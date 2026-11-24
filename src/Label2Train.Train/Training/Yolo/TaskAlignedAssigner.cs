using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Training.Yolo;

/// <summary>TAL 分配结果：锚点级软标签 + 背景/前景标记。</summary>
/// <param name="TargetScores">每个锚点每个类别的目标分数 <c>[B,N,nc]</c>（背景为 0）。</param>
/// <param name="TargetBoxes">每个锚点被分配的 GT 框 <c>[B,N,4]</c>（xyxy，像素；背景为 0）。</param>
/// <param name="Foreground">前景锚点掩码 <c>[B,N]</c>。</param>
public sealed record TalResult(Tensor TargetScores, Tensor TargetBoxes, Tensor Foreground);

/// <summary>
/// Task-Aligned Assigner（TAL）：YOLOv8 / v11 / v26 的正样本分配。
/// <para>
/// 对每个 GT 取「分类分数<sup>α</sup> × <b>CIoU</b><sup>β</sup>（负值截到 0）」最高的前 <c>topK</c> 个锚点，
/// 且锚点中心必须落在 GT 内部。与 ultralytics 的关键细节：
/// 一个锚点被多个 GT 的 top-k 同时选中时，官方直接丢弃它（<c>count_tensor &gt; 1 → 0</c>），
/// 不是「归给分数更高的那个 GT」——后者会让密集目标的正样本偏多。
/// </para>
/// <para>
/// <c>topk2</c> 是 8.4 新增的二次筛选：先按 top-k 选候选，再在每个 GT 的候选里取对齐度最高的
/// <c>topk2</c> 个。YOLO26 的 one-to-one 分支用 <c>topk=7, topk2=1</c>；为 <c>null</c> 时等于
/// <c>topK</c>，即不做二次筛选（v8 / v11 的默认行为）。
/// </para>
/// <para>分配本身不产生梯度：内部包在 <c>no_grad</c> 里，只把结果当作软标签。</para>
/// </summary>
public sealed class TaskAlignedAssigner
{
    private readonly int _classCount;
    private readonly int _topK;
    private readonly int _topK2;
    private readonly float _alpha;
    private readonly float _beta;

    /// <summary>
    /// 创建分配器；<paramref name="topK"/> 默认 10（ultralytics 8.4 的 <c>tal_topk</c> 默认值），
    /// <paramref name="topk2"/> 为 <c>null</c> 表示与 <paramref name="topK"/> 相同、不做二次筛选。
    /// </summary>
    public TaskAlignedAssigner(int classCount, int topK = 10, float alpha = 0.5f, float beta = 6f, int? topk2 = null)
    {
        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "类别数必须大于 0。");
        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), topK, "topK 必须大于 0。");
        if (topk2 is <= 0)
            throw new ArgumentOutOfRangeException(nameof(topk2), topk2, "topk2 必须大于 0（或为 null）。");
        if (alpha <= 0f || beta <= 0f)
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "alpha / beta 必须大于 0。");

        _classCount = classCount;
        _topK = topK;
        _topK2 = topk2 ?? topK;
        _alpha = alpha;
        _beta = beta;
    }

    /// <summary>执行分配（调用方应传 <c>detach()</c> 后的预测，避免建图）。</summary>
    public TalResult Assign(
        Tensor predScores,
        Tensor predBoxes,
        Tensor anchorPoints,
        Tensor gtClasses,
        Tensor gtBoxes,
        Tensor gtMask)
    {
        ArgumentNullException.ThrowIfNull(predScores);
        ArgumentNullException.ThrowIfNull(predBoxes);
        ArgumentNullException.ThrowIfNull(anchorPoints);
        ArgumentNullException.ThrowIfNull(gtClasses);
        ArgumentNullException.ThrowIfNull(gtBoxes);
        ArgumentNullException.ThrowIfNull(gtMask);

        if (predScores.dim() != 3 || predScores.shape[2] != _classCount)
        {
            throw new ArgumentException(
                $"predScores 必须是 [B,N,{_classCount}]，实际是 [{string.Join(",", predScores.shape)}]。",
                nameof(predScores));
        }

        long batch = predScores.shape[0];
        long anchors = predScores.shape[1];
        long maxBoxes = gtBoxes.shape[1];
        Device device = predScores.device;

        var scoreList = new List<Tensor>((int)batch);
        var boxList = new List<Tensor>((int)batch);
        var foregroundList = new List<Tensor>((int)batch);

        using (torch.no_grad())
        {
            long topK = Math.Min(_topK, anchors);

            for (long imageIndex = 0; imageIndex < batch; imageIndex++)
            {
                Tensor scores = predScores[imageIndex];
                Tensor boxes = predBoxes[imageIndex];
                Tensor imageGtBoxes = gtBoxes[imageIndex];
                Tensor imageGtClasses = gtClasses[imageIndex];
                Tensor imageMask = gtMask[imageIndex];

                float[] maskValues = imageMask.to(ScalarType.Float32).cpu().data<float>().ToArray();

                var metricRows = new List<Tensor>((int)maxBoxes);
                var iouRows = new List<Tensor>((int)maxBoxes);
                var topkRows = new List<Tensor>((int)maxBoxes);

                // 每个锚点被几个 GT 的 top-k 命中；> 1 的锚点按官方语义直接丢弃。
                Tensor hitCount = torch.zeros(new long[] { anchors }, dtype: ScalarType.Float32, device: device);

                for (long boxIndex = 0; boxIndex < maxBoxes; boxIndex++)
                {
                    if (maskValues[boxIndex] <= 0.5f)
                    {
                        // padding 的 GT 不参与分配。官方会把它的 top-k 索引抹成 0 并同样计入冲突统计，
                        // 逐图循环跳过更干净，也避免 anchor 0 被无意义地丢掉。
                        metricRows.Add(torch.zeros(new long[] { anchors }, dtype: ScalarType.Float32, device: device));
                        iouRows.Add(torch.zeros(new long[] { anchors }, dtype: ScalarType.Float32, device: device));
                        topkRows.Add(torch.zeros(new long[] { anchors }, dtype: ScalarType.Float32, device: device));
                        continue;
                    }

                    Tensor gt = imageGtBoxes[boxIndex];

                    // 锚点中心必须落在 GT 内部（ultralytics 的 is_in_gts）。
                    Tensor inside = torch.logical_and(
                        torch.logical_and(
                            anchorPoints.select(-1, 0) > gt[0],
                            anchorPoints.select(-1, 0) < gt[2]),
                        torch.logical_and(
                            anchorPoints.select(-1, 1) > gt[1],
                            anchorPoints.select(-1, 1) < gt[3]));

                    Tensor classIndex = imageGtClasses[boxIndex].reshape(1, 1);
                    Tensor classScores = scores.gather(1, classIndex.expand(anchors, 1)).squeeze(1);

                    // ultralytics 的 iou_calculation 用 CIoU 并 clamp 到非负（负值参与 beta 次幂会失真）
                    Tensor iou = BboxOps.Ciou(boxes, gt.expand(anchors, 4)).clamp(min: 0);
                    Tensor metric = classScores.pow(_alpha) * iou.pow(_beta);

                    metric = metric * inside.to(ScalarType.Float32);

                    (Tensor _, Tensor topIndices) = metric.topk((int)topK, dim: 0, largest: true, sorted: false);

                    Tensor row = torch.zeros(new long[] { anchors }, dtype: ScalarType.Float32, device: device);
                    row.scatter_(0, topIndices, torch.ones_like(topIndices, dtype: ScalarType.Float32));

                    hitCount.add_(row);
                    metricRows.Add(metric);
                    iouRows.Add(iou);
                    topkRows.Add(row);
                }

                Tensor rawMetric = torch.stack(metricRows.ToArray(), 0);
                Tensor iouMatrix = torch.stack(iouRows.ToArray(), 0);
                Tensor topkMatrix = torch.stack(topkRows.ToArray(), 0);

                // 官方 count_tensor.masked_fill_(count > 1, 0)：冲突锚点直接变负样本。
                Tensor uniqueMask = torch.le(hitCount, 1f).to(ScalarType.Float32);
                Tensor maskPos = topkMatrix * uniqueMask.unsqueeze(0);

                if (_topK2 < topK)
                {
                    // 官方二次筛选：在 mask_pos 覆盖的对齐度里每 GT 再取 topk2 个。
                    Tensor maskedMetric = rawMetric * maskPos;
                    (Tensor _, Tensor refined) = maskedMetric.topk(
                        (int)Math.Min(_topK2, anchors), dim: 1, largest: true, sorted: false);

                    Tensor refineMask = torch.zeros_like(maskPos);
                    refineMask.scatter_(1, refined, torch.ones_like(refined, dtype: ScalarType.Float32));
                    maskPos = maskPos * refineMask;
                }

                var best = maskPos.max(dim: 0);
                Tensor foreground = torch.gt(best.values, 0f);
                Tensor gtIndex = best.indexes;

                // 归一化到「该 GT 的最佳 IoU」，与 ultralytics 的 get_targets 一致。
                // 官方 pos_overlaps 是 (overlaps * mask_pos).amax(-1)，只取被选中的正样本；
                // 若在整张网格上取 max，one-to-one 分支（topk2=1、只剩 1 个候选）会偏大约 4%。
                Tensor posOverlap = (iouMatrix * maskPos).max(dim: 1).values.unsqueeze(1);
                Tensor posAlign = rawMetric.max(dim: 1).values.unsqueeze(1);
                Tensor metricMatrix = rawMetric * posOverlap / (posAlign + 1e-9f);

                // 目标分用「该锚点列上所有 GT 的归一化对齐度最大值」，与官方 norm_align_metric 一致。
                Tensor normalized = metricMatrix.max(dim: 0).values;
                Tensor scorePerAnchor = torch.where(foreground, normalized, torch.zeros_like(normalized));

                Tensor classTargets = torch.zeros(
                    new long[] { anchors, _classCount }, dtype: ScalarType.Float32, device: device);
                classTargets.scatter_(
                    1,
                    imageGtClasses.index_select(0, gtIndex).unsqueeze(1),
                    scorePerAnchor.unsqueeze(1));

                Tensor boxFloat = foreground.to(ScalarType.Float32).unsqueeze(1);
                Tensor imageBoxes = imageGtBoxes.index_select(0, gtIndex) * boxFloat;

                scoreList.Add(classTargets);
                boxList.Add(imageBoxes);
                foregroundList.Add(foreground);
            }

            return new TalResult(
                torch.stack(scoreList.ToArray(), 0),
                torch.stack(boxList.ToArray(), 0),
                torch.stack(foregroundList.ToArray(), 0));
        }
    }
}
