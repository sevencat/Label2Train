using System.Globalization;
using Label2Train.Train.Core;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Data.Yolo;
using Label2Train.Train.Training;
using Label2Train.Train.Training.Detr;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Sessions;

/// <summary>训练执行部分：模型装配、损失、优化器、逐轮评估与检查点。</summary>
public sealed partial class DetrTrainingSession
{
    private static IReadOnlyList<EpochMetrics> Run(
        DetrTrainingOptions options,
        TrainControl control,
        IProgress<TrainProgress>? progress,
        Action<string>? log)
    {
        if (!TrainingPresets.IsValidDetrImageSize(options.ImageSize))
        {
            throw new ArgumentException(
                $"RF-DETR 的 imgsz 必须是 {TrainingPresets.DetrImageSizeStride} 的倍数"
                + "（DINOv2 的 patch 14 与检测头的下采样 16 都要整除），"
                + $"当前 {options.ImageSize}；可用例如 224 / 336 / 448 / 560。",
                nameof(options));
        }

        Device device = DeviceFactory.Create(options.Device);

        (YoloDataset trainSet, YoloDataset? validationSet) = DetectionDatasetSource.Open(
            options.DataYaml, options.CocoJson);
        int classCount = trainSet.Config.Names.Count;

        string output = options.OutputDirectory
            ?? Path.Combine("runs", (options.UseRfDetr ? "rfdetr-" : "detr-")
                + DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(output);

        (Module<Tensor, Tensor> model, DetrArchitecture arch) = BuildModel(options, classCount, device, log);
        long parameters = model.parameters().Sum(parameter => parameter.numel());

        log?.Invoke($"data  : {options.Descriptor} · {trainSet.Samples.Count} 张 · {classCount} 类");
        log?.Invoke($"val   : {(validationSet is null ? "(无验证集)" : $"{validationSet.Samples.Count} 张")}");
        // 结构可能被预训练权重改过（patch 16 / 4 层解码器），所以报实际值而不是 options
        log?.Invoke($"model : {(options.UseRfDetr ? "rfdetr" : "detr")}-detect-{classCount} "
            + $"({parameters} params, Q={arch.Queries}, enc={arch.EncoderLayers}, dec={arch.DecoderLayers}, "
            + $"patch={arch.DinoPatch})");
        log?.Invoke($"device: {DeviceFactory.DescribeDetailed(options.Device)}");
        log?.Invoke($"output: {output}");

        var loss = new DetrLoss(classCount, options.ImageSize, options.ImageSize, groupDetr: options.GroupDetr);
        var optimizer = torch.optim.AdamW(model.parameters(), lr: options.LearningRate, weight_decay: 1e-4f);

        int batchesPerEpoch = Math.Max(1, (trainSet.Samples.Count + options.BatchSize - 1) / options.BatchSize);
        int totalSteps = (options.Epochs * batchesPerEpoch) + 1;
        int warmupSteps = Math.Min(batchesPerEpoch, totalSteps - 1);
        var schedule = new CosineSchedule(options.LearningRate, totalSteps, warmupSteps, options.LearningRate * 0.01f);

        // 顺序由下面的 BuildEpochBatches 按 epoch 自己打乱：Trainer 的委托式 Fit 重载
        // 不读 TrainerOptions.Shuffle，这里写 true 也不会生效，显式 false 免得误以为开着
        var trainer = new Trainer(
            model,
            optimizer,
            loss,
            new TrainerOptions { Epochs = options.Epochs, GradientClipNorm = 1f, Shuffle = false },
            new Rng(4));

        // 预处理按 DETR 的步长校验（112 = patch 14 与参考网格 16 的公倍数）。
        // 用 YOLO 缺省的 32 会拦掉 336 / 560 这些合法边长，训练在加载器构造时就失败
        using var trainLoader = new YoloSampleLoader(new YoloPreprocessOptions
        {
            Width = options.ImageSize,
            Height = options.ImageSize,
            SizeStride = TrainingPresets.DetrImageSizeStride,
        });
        using var validationLoader = new YoloSampleLoader(new YoloPreprocessOptions
        {
            Width = options.ImageSize,
            Height = options.ImageSize,
            SizeStride = TrainingPresets.DetrImageSizeStride,
        });

        var reports = new List<TrainProgress>(options.Epochs);
        float bestMap50 = 0f;
        float lastMap95 = 0f;

        // RF-DETR 从随机初始化开始，前几轮的 mAP 就是 0：只看 map50 > bestMap50 的话
        // best.safetensors 永远不会写出来，用户一停训就「没有可导出的权重」
        bool bestSaved = false;

        IReadOnlyList<EpochMetrics> history = trainer.Fit(
            BuildEpochBatches,
            schedule,
            metrics =>
            {
                (float map50, float map95) = EvaluateMap(
                    model, validationSet, validationLoader, options, classCount, device);

                var item = new TrainProgress(
                    metrics.Epoch, options.Epochs, metrics.Step, metrics.Loss, metrics.LearningRate, map50, map95);

                reports.Add(item);
                lastMap95 = map95;

                // 分项（加权）跟着每轮打一次：loss 卡在平台时，能立刻分清是分类项
                // 压不下来（样本太少 / 类别不平衡）还是框回归没收敛（匹配或归一化问题）
                (float classPart, float l1Part, float giouPart) = loss.AverageParts;
                loss.ResetParts();

                log?.Invoke($"epoch {metrics.Epoch + 1}/{options.Epochs}  loss {metrics.Loss:F4}"
                    + $"  [cls {classPart:F3} / l1 {l1Part:F3} / giou {giouPart:F3}]"
                    + $"  mAP50 {map50:F3}  mAP50-95 {map95:F3}");

                progress?.Report(item);

                // 第一次评估一定落一份：哪怕 mAP 还是 0，也是「训过一轮」可信的权重，
                // 后面只有真的更好才覆盖
                if (!bestSaved || map50 > bestMap50)
                {
                    bestSaved = true;
                    bestMap50 = map50;
                    Checkpoint.Save(
                        Path.Combine(output, "best.safetensors"),
                        model,
                        BuildMetadata(options, arch, classCount, metrics));
                }
            },
            control);

        Checkpoint.Save(
            Path.Combine(output, "detr.safetensors"),
            model,
            BuildMetadata(options, arch, classCount, history[^1]));

        WriteReport(output, options, classCount, parameters, reports, bestMap50, lastMap95);
        log?.Invoke($"checkpoint: {Path.Combine(output, "detr.safetensors")}");
        log?.Invoke($"report    : {Path.Combine(output, "train.json")}");

        return history;

        IEnumerable<Batch> BuildEpochBatches(int epoch)
        {
            IReadOnlyList<YoloSample> samples = trainSet.Samples;

            // 每轮换一次顺序：24 张图 + batch 4 只有 6 步，顺序固定的话每轮梯度方向几乎一样，
            // 表现为 loss 上下震荡而 mAP 起不来（DETR 没有增强，顺序就是唯一的随机性）
            int[] order = ShuffleSamples(samples.Count, new Rng((ulong)(0x51ED270B + epoch * 7919)));

            Batch? previous = null;
            try
            {
                for (int start = 0; start < order.Length; start += options.BatchSize)
                {
                    int count = Math.Min(options.BatchSize, order.Length - start);
                    var images = new YoloImageSample[count];

                    // 一批图串行解码 + letterbox + 增强要几百毫秒，GPU 一步才 ~0.1 s，
                    // 数据准备会拖住整条流水线；各张图互不依赖，按批并行。
                    Parallel.For(
                        0,
                        count,
                        new ParallelOptions { MaxDegreeOfParallelism = Math.Min(count, 8) },
                        offset => images[offset] = trainLoader.Load(samples[order[start + offset]]));

                    YoloBatchResult packed = YoloBatchBuilder.Build(images);

                    // 上一批已训练完（backward 结束），先把它的显存还回去再准备下一批。
                    previous?.Dispose();
                    previous = new Batch(
                        TorchInterop.FromArray(
                            packed.Inputs,
                            new long[] { packed.BatchSize, packed.Channels, packed.Height, packed.Width },
                            device),
                        TorchInterop.FromArray(
                            packed.Targets,
                            new long[] { packed.BatchSize, packed.MaxBoxes, YoloBatchBuilder.TargetColumns },
                            device));

                    yield return previous;
                }
            }
            finally
            {
                previous?.Dispose();
            }
        }
    }

    /// <summary>Fisher-Yates 打乱 <c>[0, count)</c>；与 <see cref="YoloTrainingSession"/> 同一套做法。</summary>
    private static int[] ShuffleSamples(int count, Rng rng)
    {
        var order = new int[count];
        for (int i = 0; i < count; i++)
            order[i] = i;

        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }

        return order;
    }
}
