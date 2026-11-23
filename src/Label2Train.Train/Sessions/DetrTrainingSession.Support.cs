using Label2Train.Train.Core;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Data.Yolo;
using Label2Train.Train.Models.RfDetr;
using Label2Train.Train.Models.SimpleDetr;
using Label2Train.Train.Training;
using Label2Train.Train.Training.Detr;
using Label2Train.Train.Training.Yolo;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Sessions;

/// <summary>训练辅助：模型装配与验证集评估。</summary>
public sealed partial class DetrTrainingSession
{
    /// <summary>每图参与评估的最大框数（COCO 惯例）。</summary>
    private const int EvaluationMaxDetections = 100;

    /// <summary>
    /// 随机初始化的头（分类头、query 汇总、参考点）想要的学习率：1e-3。
    /// <para>
    /// 官方权重只有主干 / 解码器 / 框头能用（类别数与组数都对不上），这几块必须学得快一点；
    /// 但**不能直接乘固定倍数**：基础 lr 取 1e-3（从零训的档位）时再乘 10 就是 1e-2，
    /// Adam 一步就能把 <c>query_embed</c>（N(0,1) 初始化）和参考点推到 sigmoid 的饱和端，
    /// 之后梯度精确为 0、loss 冻在原地（实测两组不同初始化冻在同一个 10.238）。
    /// </para>
    /// </summary>
    private const float HeadLearningRateCap = 1e-3f;

    /// <summary>随机头最多放大的倍数（基础 lr 已经 ≤ 1e-4 时才用得上）。</summary>
    internal const float HeadGradientScaleMax = 10f;

    /// <summary>
    /// 随机头的梯度放大倍数 = 把它推到 <see cref="HeadLearningRateCap"/> 需要多少倍，
    /// 夹在 [1, <see cref="HeadGradientScaleMax"/>] 内：基础 lr 1e-4 → ×10，1e-3 → ×1（不再放大）。
    /// </summary>
    internal static float HeadGradientScale(float learningRate) =>
        learningRate <= 0f
            ? HeadGradientScaleMax
            : Math.Clamp(HeadLearningRateCap / learningRate, 1f, HeadGradientScaleMax);

    /// <summary>
    /// 算 AP 时先按这个阈值过滤，与 YOLO 路径（0.001）保持一致。
    /// 不能用 <see cref="DetrTrainingOptions.Confidence"/>（默认 0.05）：AP 依赖「按分数排序的全部预测」，
    /// 先砍掉低分框会把还没训好的正确框一起砍掉，mAP 就永远是 0。
    /// </summary>
    private const float EvaluationConfidence = 0.001f;

    /// <summary>一轮验证的 mAP 加诊断量；<see cref="Empty"/> 表示没有验证集。</summary>
    /// <param name="BestIou">阈值过滤后的检测框里，每个 GT 同类别的最佳 IoU。</param>
    /// <param name="AnyIou">不看分数、不看类别，300 个 query 里每个 GT 的最佳 IoU。</param>
    private readonly record struct DetrEvaluation(
        float Map50,
        float Map50_95,
        float DetectionsPerImage,
        float TruthsPerImage,
        float TopScore,
        float BestIou,
        float Iou50Ratio,
        float AnyIou,
        float AnyIou50Ratio)
    {
        public static DetrEvaluation Empty => default;
    }

    /// <summary>
    /// 是不是「权重里拿不到、只能随机初始化」的参数：分类头和 query / 参考点汇总。
    /// 这几块的键名与官方权重对不上（91 类、13 组），会被 <c>load_state_dict</c> 跳过。
    /// </summary>
    private static bool IsRandomHead(string name) =>
        name.Contains("class_embed", StringComparison.Ordinal)
        || name.Contains("query_embed", StringComparison.Ordinal)
        || name.Contains("refpoint_embed", StringComparison.Ordinal);

    /// <summary>按参数装配 RF-DETR 或简化 DETR，并把实际结构一并返回（写进检查点元数据）。</summary>
    private static (Module<Tensor, Tensor> Model, DetrArchitecture Architecture) BuildModel(
        DetrTrainingOptions options, int classCount, Device device, Action<string>? log)
    {
        string? weightsPath = string.IsNullOrWhiteSpace(options.DinoWeights)
            ? null
            : Path.GetFullPath(options.DinoWeights);

        if (weightsPath is not null && !File.Exists(weightsPath))
            throw new FileNotFoundException($"预训练权重不存在：{weightsPath}", weightsPath);

        // 官方整网权重（backbone.0.*）结构与本项目的默认值差很多（patch 16 / 4 层解码器 /
        // C2f 单档 / FFN 2048），必须按检查点来建模型，否则键名对不上、只剩主干能用
        RfDetrWeightsInfo? pretrained = options.UseRfDetr && weightsPath is not null
            ? RfDetrWeightsProbe.Inspect(weightsPath)
            : null;

        if (!options.UseRfDetr)
        {
            return (
                new SimpleDetrModel(
                    classCount, options.Queries, options.EncoderLayers, options.DecoderLayers, device: device),
                new DetrArchitecture(
                    options.Queries, options.EncoderLayers, options.DecoderLayers, options.ModelHeads,
                    options.DinoEmbedding, options.DinoDepth, options.DinoHeads, 14, 518, 1, false));
        }

        var dinoConfig = new DinoV2Config(
            ImageSize: pretrained?.ImageSize ?? 518,
            PatchSize: pretrained?.PatchSize ?? 14,
            EmbeddingSize: pretrained?.EmbeddingSize ?? options.DinoEmbedding,
            Depth: pretrained?.Depth ?? options.DinoDepth,
            Heads: pretrained?.Heads ?? options.DinoHeads);

        // 层数与元数据保持一致，导出时才能按同样的结构重建。
        // query 数按用户设置：官方 query_feat 里的 300 行只是 COCO 的量级，
        // 对几百张、每图 4~5 个目标的数据集负样本会多出十几倍，分类头收敛慢一个量级。
        // 装载时按行切前 N 行（RfDetrPretrained.Prepare），不必被 300 绑住。
        var architecture = new DetrArchitecture(
            options.Queries,
            pretrained is null ? options.EncoderLayers : 0,
            pretrained?.DecoderLayers ?? options.DecoderLayers,
            options.ModelHeads,
            dinoConfig.EmbeddingSize,
            dinoConfig.Depth,
            dinoConfig.Heads,
            dinoConfig.PatchSize,
            dinoConfig.ImageSize,
            1,
            true,
            FfnSize: pretrained is null ? 1024 : 2048,
            DeformHeads: pretrained?.DecoderHeads ?? 0,
            DeformLevels: pretrained?.DeformLevels ?? RfDetrDetector.MaxLevels,
            DeformPoints: pretrained?.DeformPoints ?? 4,
            UseRefPointHead: pretrained is not null,
            UseC2fProjector: pretrained?.HasC2fProjector ?? false);

        var detector = new RfDetrDetector(
            classCount,
            architecture.Queries,
            dinoConfig,
            architecture.Heads,
            device,
            encoderLayers: architecture.EncoderLayers,
            decoderLayers: architecture.DecoderLayers,
            ffnSize: architecture.FfnSize,
            points: architecture.DeformPoints,
            levels: architecture.DeformLevels,
            decoderHeads: architecture.DeformHeads,
            useRefPointHead: architecture.UseRefPointHead,
            useC2fProjector: architecture.UseC2fProjector);

        if (weightsPath is not null)
        {
            if (pretrained is null)
            {
                // 纯 DINOv2 主干权重：键名就是 backbone.* 去掉前缀，直接套
                using Checkpoint dino = Checkpoint.Open(weightsPath);
                var prefixed = new Dictionary<string, Tensor>(StringComparer.Ordinal);

                foreach ((string key, Tensor tensor) in dino.LoadState(device))
                    prefixed["backbone." + key] = tensor;

                detector.load_state_dict(prefixed, strict: false);
                log?.Invoke($"dino  : {weightsPath}（{prefixed.Count} 个张量载入 backbone）");
            }
            else
            {
                RfDetrPretrainedReport report = RfDetrPretrained.Apply(detector, weightsPath, device, pretrained);

                log?.Invoke($"rfdetr: {weightsPath}");
                log?.Invoke($"rfdetr: {report.Summary}");
            }
        }

        return (detector, architecture);
    }

    /// <summary>
    /// 验证集上算 mAP：解码 → 与真值比对（与 CLI 的 DETR 评估口径一致）。
    /// 同时统计 det / gt / top1 score / IoU：mAP 一直是 0 时，光看那个 0 分不清
    /// 「模型还没学到」和「框的位置根本对不上」。
    /// </summary>
    private static DetrEvaluation EvaluateMap(
        Module<Tensor, Tensor> model,
        YoloDataset? validationSet,
        YoloSampleLoader validationLoader,
        DetrTrainingOptions options,
        int classCount,
        Device device)
    {
        if (validationSet is null || options.EvalEvery <= 0)
            return DetrEvaluation.Empty;

        IEnumerable<YoloSample> samples = validationSet.Samples;
        if (options.EvaluationLimit > 0)
            samples = samples.Take(options.EvaluationLimit);

        var predictions = new List<IReadOnlyList<YoloDetection>>();
        var groundTruths = new List<IReadOnlyList<YoloGroundTruth>>();

        int detections = 0;
        double topScoreSum = 0;
        int truthCount = 0;
        double bestIouSum = 0;
        int iouAbove50 = 0;
        double anyIouSum = 0;
        int anyIouAbove50 = 0;

        model.eval();

        using (torch.no_grad())
        {
            foreach (YoloSample sample in samples)
            {
                // 单张前向的激活也是几百 MB：验证集一大，不逐张释放就会把显存撑爆。
                using (var scope = torch.NewDisposeScope())
                {
                    YoloImageSample image = validationLoader.Load(sample);
                    Tensor input = TorchInterop.FromArray(
                        image.ToChw(),
                        new long[] { 1, 3, options.ImageSize, options.ImageSize },
                        device);

                    Tensor output = model.forward(input);

                    // 每图最多留 100 个框：COCO 的评估惯例，也是这一个批次里最多能匹配上的数量级；
                    // 300 个 query 全留下只会让评估慢三倍，对 AP 没有帮助
                    IReadOnlyList<YoloDetection> decoded = DetrPostProcess.Decode(
                        output,
                        options.ImageSize,
                        options.ImageSize,
                        classCount,
                        EvaluationConfidence,
                        maxDetections: EvaluationMaxDetections)[0];

                    predictions.Add(decoded);

                    // det/img 仍按「用户看到的阈值」统计：阈值过滤后一个框都不剩，
                    // 说明模型还没产出可用结果，这个数比 AP 的 0 更直观
                    detections += decoded.Count(detection => detection.Score >= options.Confidence);

                    if (decoded.Count > 0)
                        topScoreSum += decoded[0].Score;

                    YoloGroundTruth[] imageTruths = image.Boxes
                        .Select(box => new YoloGroundTruth(box.ClassId, box.X1, box.Y1, box.X2, box.Y2))
                        .ToArray();

                    groundTruths.Add(imageTruths);
                    truthCount += imageTruths.Length;

                    float[] raw = TorchInterop.ToArray(output);

                    foreach (YoloGroundTruth truth in imageTruths)
                    {
                        float best = 0f;

                        foreach (YoloDetection detection in decoded)
                        {
                            if (detection.ClassId == truth.ClassId)
                                best = MathF.Max(best, Iou(detection, truth));
                        }

                        bestIouSum += best;

                        if (best >= 0.5f)
                            iouAbove50++;

                        // 全体 query（不看分数、不看类别）的最佳 IoU：分数塌缩与框回归失败
                        // 在 iou 这一列上是同一个 0，分开看才知道该修损失还是修框
                        float any = 0f;

                        for (int query = 0; query < raw.Length / (4 + classCount); query++)
                        {
                            int offset = query * (4 + classCount);
                            float halfWidth = raw[offset + 2] / 2f;
                            float halfHeight = raw[offset + 3] / 2f;

                            any = MathF.Max(
                                any,
                                Iou(
                                    Math.Clamp(raw[offset] - halfWidth, 0f, 1f) * options.ImageSize,
                                    Math.Clamp(raw[offset + 1] - halfHeight, 0f, 1f) * options.ImageSize,
                                    Math.Clamp(raw[offset] + halfWidth, 0f, 1f) * options.ImageSize,
                                    Math.Clamp(raw[offset + 1] + halfHeight, 0f, 1f) * options.ImageSize,
                                    truth));
                        }

                        anyIouSum += any;

                        if (any >= 0.5f)
                            anyIouAbove50++;
                    }

                    scope.Dispose();
                }
            }
        }

        model.train();

        if (predictions.Count == 0)
            return DetrEvaluation.Empty;

        DetectionMetricsResult result = DetectionMetrics.Evaluate(predictions, groundTruths, classCount);
        float images = predictions.Count;

        return new DetrEvaluation(
            result.Map50,
            result.Map50_95,
            detections / images,
            truthCount / images,
            (float)(topScoreSum / images),
            truthCount == 0 ? 0f : (float)(bestIouSum / truthCount),
            truthCount == 0 ? 0f : (float)iouAbove50 / truthCount,
            truthCount == 0 ? 0f : (float)(anyIouSum / truthCount),
            truthCount == 0 ? 0f : (float)anyIouAbove50 / truthCount);
    }

    /// <summary>GT 与检测框的 IoU（同类别才比，调用方已过滤）。</summary>
    private static float Iou(YoloDetection detection, YoloGroundTruth truth) =>
        Iou(detection.X1, detection.Y1, detection.X2, detection.Y2, truth);

    /// <summary>像素 xyxy 与 GT 的 IoU。</summary>
    private static float Iou(float x1, float y1, float x2, float y2, YoloGroundTruth truth)
    {
        float width = MathF.Min(x2, truth.X2) - MathF.Max(x1, truth.X1);
        float height = MathF.Min(y2, truth.Y2) - MathF.Max(y1, truth.Y1);

        if (width <= 0f || height <= 0f)
            return 0f;

        float intersection = width * height;
        float union = (x2 - x1) * (y2 - y1)
            + (truth.X2 - truth.X1) * (truth.Y2 - truth.Y1) - intersection;

        return union <= 0f ? 0f : intersection / union;
    }

    /// <summary>检查点元数据：与 CLI 的 detr 检查点同名字段，导出器可直接复用。</summary>
    private static Dictionary<string, string> BuildMetadata(
        DetrTrainingOptions options, DetrArchitecture arch, int classCount, EpochMetrics metrics)
    {
        static string Text(long value) => value.ToString(System.Globalization.CultureInfo.InvariantCulture);

        return new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["format"] = "omnits-checkpoint/1",
            ["architecture"] = (arch.UseRfDetr ? "rfdetr" : "simple-detr") + $"-detect-{classCount}",
            ["epoch"] = Text(metrics.Epoch),
            ["step"] = Text(metrics.Step),
            ["loss"] = metrics.Loss.ToString("G9", System.Globalization.CultureInfo.InvariantCulture),
            ["queries"] = Text(arch.Queries),
            ["encoder_layers"] = Text(arch.EncoderLayers),
            ["decoder_layers"] = Text(arch.DecoderLayers),
            ["heads"] = Text(arch.Heads),
            ["dino_embed"] = Text(arch.DinoEmbedding),
            ["dino_depth"] = Text(arch.DinoDepth),
            ["dino_heads"] = Text(arch.DinoHeads),
            // patch / pos_embed 边长必须一起存：官方权重是 patch 16 + 576，用默认的 14 + 518
            // 重建会把主干参数整片丢掉，导出的 ONNX 就成了随机网
            ["dino_patch"] = Text(arch.DinoPatch),
            ["dino_image"] = Text(arch.DinoImage),
            ["group_detr"] = Text(arch.GroupDetr),
            // 载入官方权重后这些不再等于构造函数默认值，导出必须照抄，否则重建的结构对不上权重
            ["ffn_size"] = Text(arch.FfnSize),
            ["deform_heads"] = Text(arch.DeformHeads),
            ["deform_levels"] = Text(arch.DeformLevels),
            ["deform_points"] = Text(arch.DeformPoints),
            ["ref_point"] = arch.UseRefPointHead ? "1" : "0",
            ["c2f"] = arch.UseC2fProjector ? "1" : "0",
            ["image_size"] = Text(options.ImageSize),
            ["blocks"] = "2,2,2,2",
            ["classes"] = Text(classCount),
            ["backend"] = DeviceFactory.Describe(options.Device),
        };
    }

    /// <summary>写 train.json（字段与 CLI 报告同名，界面与脚本都能直接读）。</summary>
    private static void WriteReport(
        string output,
        DetrTrainingOptions options,
        int classCount,
        long parameters,
        List<TrainProgress> reports,
        float bestMap50,
        float lastMap95)
    {
        var report = new
        {
            architecture = (options.UseRfDetr ? "rfdetr" : "simple-detr") + $"-detect-{classCount}",
            backend = DeviceFactory.Describe(options.Device),
            data = options.Descriptor,
            classes = classCount,
            queries = options.Queries,
            epochs = options.Epochs,
            batchSize = options.BatchSize,
            imageSize = options.ImageSize,
            learningRate = options.LearningRate,
            parameters,
            bestMap50,
            bestMap50_95 = lastMap95,
            history = reports.Select(item => new
            {
                item.Epoch,
                item.Step,
                item.Loss,
                Map50 = item.Map50,
                Map50_95 = item.Map50_95,
            }).ToArray(),
        };

        File.WriteAllText(
            Path.Combine(output, "train.json"),
            System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    }
}
