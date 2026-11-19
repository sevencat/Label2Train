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

        // 层数与元数据保持一致，导出时才能按同样的结构重建
        var architecture = new DetrArchitecture(
            pretrained?.Queries ?? options.Queries,
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

    /// <summary>验证集上算 mAP：解码 → 与真值比对（与 CLI 的 DETR 评估口径一致）。</summary>
    private static (float Map50, float Map50_95) EvaluateMap(
        Module<Tensor, Tensor> model,
        YoloDataset? validationSet,
        YoloSampleLoader validationLoader,
        DetrTrainingOptions options,
        int classCount,
        Device device)
    {
        if (validationSet is null || options.EvalEvery <= 0)
            return (0f, 0f);

        IEnumerable<YoloSample> samples = validationSet.Samples;
        if (options.EvaluationLimit > 0)
            samples = samples.Take(options.EvaluationLimit);

        var predictions = new List<IReadOnlyList<YoloDetection>>();
        var groundTruths = new List<IReadOnlyList<YoloGroundTruth>>();

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

                    predictions.Add(DetrPostProcess.Decode(
                        model.forward(input), options.ImageSize, options.ImageSize, classCount, options.Confidence)[0]);

                    groundTruths.Add(image.Boxes
                        .Select(box => new YoloGroundTruth(box.ClassId, box.X1, box.Y1, box.X2, box.Y2))
                        .ToArray());

                    scope.Dispose();
                }
            }
        }

        model.train();

        if (predictions.Count == 0)
            return (0f, 0f);

        DetectionMetricsResult result = DetectionMetrics.Evaluate(predictions, groundTruths, classCount);
        return (result.Map50, result.Map50_95);
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
