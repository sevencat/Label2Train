using Label2Train.Train.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Training.Yolo;

/// <summary>解码后的一个检测框（xyxy，网络输入像素）。</summary>
public readonly record struct YoloDetection(int ClassId, float Score, float X1, float Y1, float X2, float Y2)
{
    public float Width => X2 - X1;

    public float Height => Y2 - Y1;
}

/// <summary>评估用的真值框（xyxy，网络输入像素）。</summary>
public readonly record struct YoloGroundTruth(int ClassId, float X1, float Y1, float X2, float Y2);

/// <summary>
/// 把模型原始输出 <c>[1, 4*reg_max + nc, N]</c> 解码成检测框：
/// DFL 积分 → dist2bbox → sigmoid 分类 → 置信度过滤 → NMS。
/// <para>
/// 与训练侧共用 <see cref="YoloAnchorGrid"/> 的锚点顺序；导出 ONNX 时同一套公式用图节点表达。
/// </para>
/// </summary>
public static class YoloPredictionDecoder
{
    /// <summary>无 NMS 解码的默认保留数（官方 <c>max_det=300</c>）。</summary>
    public const int DefaultEndToEndDetections = 300;

    /// <summary>解码单张图的预测。</summary>
    public static IReadOnlyList<YoloDetection> Decode(
        Tensor prediction,
        int imageWidth,
        int imageHeight,
        int classCount,
        int regressionMax = 16,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300)
    {
        ArgumentNullException.ThrowIfNull(prediction);

        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "类别数必须大于 0。");

        if (prediction.dim() == 3)
        {
            if (prediction.shape[0] != 1)
            {
                throw new ArgumentException(
                    $"解码一次只处理一张图，prediction 的 batch 必须是 1，实际是 {prediction.shape[0]}。",
                    nameof(prediction));
            }

            prediction = prediction[0];
        }

        int outputChannels = 4 * regressionMax + classCount;
        if (prediction.dim() != 2 || prediction.shape[0] != outputChannels)
        {
            throw new ArgumentException(
                $"prediction 必须是 [{outputChannels},N]，实际是 [{string.Join(",", prediction.shape)}]。",
                nameof(prediction));
        }

        YoloAnchor[] anchors = YoloAnchorGrid.Create(imageWidth, imageHeight);
        int anchorCount = anchors.Length;

        if (prediction.shape[1] != anchorCount)
        {
            throw new ArgumentException(
                $"prediction 的锚点数 {prediction.shape[1]} 与 {imageWidth}x{imageHeight} 的网格 {anchorCount} 不一致。",
                nameof(prediction));
        }

        // 解码走向量化路径（与 ONNX 导出同一套公式），逐锚点只做一次 argmax，
        // 比原来的「每锚点 4×reg_max 次 exp + 归一化」快一个量级。
        Tensor decoded = DecodeOutput0(prediction.unsqueeze(0), imageWidth, imageHeight, classCount, regressionMax);
        float[] values = TorchInterop.ToArray(decoded[0]);
        var detections = new List<YoloDetection>();

        for (int anchorIndex = 0; anchorIndex < anchorCount; anchorIndex++)
        {
            int bestClass = -1;
            float bestScore = 0f;

            for (int classId = 0; classId < classCount; classId++)
            {
                float score = values[(4 + classId) * anchorCount + anchorIndex];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = classId;
                }
            }

            if (bestClass < 0 || bestScore < confidenceThreshold)
                continue;

            detections.Add(new YoloDetection(
                bestClass,
                bestScore,
                values[anchorIndex],
                values[anchorCount + anchorIndex],
                values[2 * anchorCount + anchorIndex],
                values[3 * anchorCount + anchorIndex]));
        }

        return YoloNms.Apply(detections, iouThreshold, maxDetections);
    }

    /// <summary>
    /// 把原始预测 <c>[B, 4*reg_max+nc, N]</c> 向量化解码成 <c>output0</c>
    /// （<c>[B, 4+nc, N]</c>，<b>xyxy</b> 像素坐标 + sigmoid 分类分数），不做置信度过滤与 NMS。
    /// <para>
    /// 与 ONNX 导出使用同一套公式：DFL 期望值 → 乘 stride → 相对锚点中心偏移。
    /// <c>reg_max = 1</c>（YOLO26）时没有 DFL，框分支输出直接就是距离，不再做 softmax 加权和。
    /// </para>
    /// <para>
    /// 注意：ultralytics 自己的 <c>head.forward</c> 在 <c>eval</c> 下输出的是 <b>xywh</b>
    /// （<c>dist2bbox(xywh=True)</c>），本项目输出 xyxy。两者逐元素对拍前要转换一次，
    /// 见 <c>YoloPretrainedParityTests</c>；如需把导出的 ONNX 交给 ultralytics 的
    /// <c>non_max_suppression</c>，也要把 <c>xywh=False</c> 传对。
    /// </para>
    /// </summary>
    public static Tensor DecodeOutput0(
        Tensor prediction,
        int imageWidth,
        int imageHeight,
        int classCount,
        int regressionMax = 16)
    {
        ArgumentNullException.ThrowIfNull(prediction);

        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "类别数必须大于 0。");

        if (prediction.dim() != 3 || prediction.shape[1] != 4L * regressionMax + classCount)
        {
            throw new ArgumentException(
                $"prediction 必须是 [B,{4 * regressionMax + classCount},N]，"
                + $"实际是 [{string.Join(",", prediction.shape)}]。",
                nameof(prediction));
        }

        YoloAnchor[] anchors = YoloAnchorGrid.Create(imageWidth, imageHeight);
        if (prediction.shape[2] != anchors.Length)
        {
            throw new ArgumentException(
                $"prediction 的锚点数 {prediction.shape[2]} 与 {imageWidth}x{imageHeight} 的网格 {anchors.Length} 不一致。",
                nameof(prediction));
        }

        long batch = prediction.shape[0];
        var levels = new List<Tensor>(YoloAnchorGrid.Strides.Length);

        using (torch.no_grad())
        {
            // transpose(1,2) 之后是 [batch, reg_max, 4, count]，所以 reg_max 落在第 1 维
            Tensor weights = torch.arange(
                regressionMax,
                dtype: ScalarType.Float32,
                device: prediction.device).view(1, regressionMax, 1, 1);

            int offset = 0;

            foreach (int stride in YoloAnchorGrid.Strides)
            {
                int count = imageHeight / stride * (imageWidth / stride);

                Tensor boxLogits = prediction
                    .narrow(1, 0, 4 * regressionMax)
                    .narrow(2, offset, count);

                Tensor classLogits = prediction
                    .narrow(1, 4 * regressionMax, classCount)
                    .narrow(2, offset, count);

                // 与 ultralytics 的 DFL 一致：通道布局是 [4, reg_max]（方向在外、bin 在内）。
                // reg_max = 1（YOLO26）时官方不建 DFL（dfl 是 Identity），框分支输出直接当距离用。
                Tensor distances = regressionMax == 1
                    ? boxLogits.view(batch, 4, count)
                    : boxLogits
                        .view(batch, 4, regressionMax, count)
                        .transpose(1, 2)
                        .softmax(1)
                        .mul(weights)
                        .sum(1)
                        .view(batch, 4, count);

                distances = distances.mul(stride); // 距离的单位是 stride，换成像素

                var centerX = new float[count];
                var centerY = new float[count];
                for (int index = 0; index < count; index++)
                {
                    centerX[index] = anchors[offset + index].PixelX;
                    centerY[index] = anchors[offset + index].PixelY;
                }

                Tensor anchorX = TorchInterop.FromArray(
                    centerX, new long[] { 1, 1, count }, prediction.device);
                Tensor anchorY = TorchInterop.FromArray(
                    centerY, new long[] { 1, 1, count }, prediction.device);

                Tensor boxes = torch.cat(
                    new[]
                    {
                        anchorX - distances.narrow(1, 0, 1),
                        anchorY - distances.narrow(1, 1, 1),
                        anchorX + distances.narrow(1, 2, 1),
                        anchorY + distances.narrow(1, 3, 1),
                    },
                    1);

                levels.Add(torch.cat(new[] { boxes, classLogits.sigmoid() }, 1));
                offset += count;
            }
        }

        return torch.cat(levels.ToArray(), 2);
    }

    /// <summary>
    /// 无 NMS 解码（YOLO26 的 one-to-one 分支）：从原始 <c>[B, 4*reg_max+nc, N]</c>
    /// 直接得到 <c>[B, K, 6]</c>——每行 <c>x1, y1, x2, y2, score, class</c>。
    /// <para>
    /// 与官方 <c>head.postprocess</c> 的等价写法：每个锚点取最高分类分数，再对全部锚点做 top-k。
    /// 与导出成 ONNX 的计算顺序一致（ReduceMax → TopK → Gather），便于逐元素对拍。
    /// </para>
    /// </summary>
    /// <param name="prediction">模型原始输出。</param>
    /// <param name="imageWidth">输入图宽（像素）。</param>
    /// <param name="imageHeight">输入图高（像素）。</param>
    /// <param name="classCount">类别数。</param>
    /// <param name="regressionMax">DFL 区间数；YOLO26 为 1。</param>
    /// <param name="maxDetections">保留的框数 K（官方 300）。</param>
    public static Tensor DecodeEndToEnd(
        Tensor prediction,
        int imageWidth,
        int imageHeight,
        int classCount,
        int regressionMax = 1,
        int maxDetections = DefaultEndToEndDetections)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxDetections);

        Tensor output0 = DecodeOutput0(prediction, imageWidth, imageHeight, classCount, regressionMax);

        using (torch.no_grad())
        {
            long batch = output0.shape[0];
            long anchors = output0.shape[2];
            int keep = (int)Math.Min(maxDetections, anchors);

            Tensor boxes = output0.narrow(1, 0, 4);              // [B,4,N]
            Tensor scores = output0.narrow(1, 4, classCount);    // [B,nc,N]

            (Tensor bestScore, Tensor classIndex) = scores.max(1); // [B,N]
            (Tensor topScore, Tensor topIndex) = bestScore.topk(keep, dim: 1, largest: true, sorted: true);

            Tensor topBoxes = boxes
                .gather(2, topIndex.unsqueeze(1).expand(batch, 4, keep))
                .transpose(1, 2);                                 // [B,K,4]

            Tensor topClass = classIndex
                .gather(1, topIndex)
                .to_type(ScalarType.Float32)
                .unsqueeze(2);                                    // [B,K,1]

            return torch.cat(new[] { topBoxes, topScore.unsqueeze(2), topClass }, 2).contiguous();
        }
    }
}
