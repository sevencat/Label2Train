using System.Globalization;
using Label2Train.Train.Core;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Data.Yolo;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training;
using Label2Train.Train.Training.Yolo;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Sessions;

/// <summary>训练执行部分：数据集装配、损失、优化器与训练循环（与 CLI 的 yolo train 同一条路径）。</summary>
public sealed partial class YoloTrainingSession
{
    private static IReadOnlyList<EpochMetrics> Run(
        YoloTrainingOptions options,
        TrainControl control,
        IProgress<TrainProgress>? progress,
        Action<string>? log)
    {
        Device device = DeviceFactory.Create(options.Device);

        // YOLO（data.yaml）与 COCO（json）在这里统一成同一份样本列表。
        (YoloDataset trainSet, YoloDataset? validationSet) = DetectionDatasetSource.Open(
            options.DataYaml, options.CocoJson);
        int classCount = trainSet.Config.Names.Count;

        if (classCount <= 0)
            throw new InvalidOperationException($"{options.Descriptor} 里没有类别定义（names 为空）。");

        string output = options.OutputDirectory
            ?? Path.Combine("runs", "gui-" + DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(output);

        YoloScale scale = YoloScales.Parse(options.Scale);
        YoloModelFamily family = YoloScales.ParseFamily(options.Family);
        YoloDetectionModel model = YoloArchitectures.Create(device, family, scale, classCount);

        // 分类 bias 先验按真实输入尺寸设一次（创建时按 640 设过）。
        model.Head.ApplyClassBiasPrior(options.ImageSize);

        log?.Invoke($"data  : {options.Descriptor} · {trainSet.Samples.Count} 张 · {classCount} 类");
        log?.Invoke($"val   : {(validationSet is null ? "(无验证集)" : $"{validationSet.Samples.Count} 张")}");
        log?.Invoke($"model : {model.Architecture} ({YoloArchitectures.CountParameters(model)} params)");
        log?.Invoke($"device: {DeviceFactory.DescribeDetailed(options.Device)}");
        log?.Invoke($"output: {output}");

        var optimizer = torch.optim.AdamW(model.parameters(), lr: options.LearningRate, weight_decay: 0.0005f);
        var loss = new YoloDetectionLoss(options.ImageSize, options.ImageSize, classCount) { TrackComponents = true };

        int batchesPerEpoch = Math.Max(1, (trainSet.Samples.Count + options.BatchSize - 1) / options.BatchSize);
        int totalSteps = (options.Epochs * batchesPerEpoch) + 1;
        int warmupSteps = Math.Min(Math.Max(1, 3 * batchesPerEpoch), Math.Max(0, totalSteps - 1));

        // 短训练用 0.999 的 EMA 相当于停在初始权重上，按总步数取衰减。
        var ema = new Ema(model, decay: Math.Clamp(1f - 3f / Math.Max(1, totalSteps), 0.9f, 0.999f));
        var schedule = new CosineSchedule(options.LearningRate, totalSteps, warmupSteps, options.LearningRate * 0.01f);

        var trainer = new Trainer(
            model,
            optimizer,
            loss,
            new TrainerOptions { Epochs = options.Epochs, GradientClipNorm = 10f, Shuffle = false },
            new Rng(1),
            ema);

        using var trainLoader = new YoloSampleLoader(new YoloPreprocessOptions
        {
            Width = options.ImageSize,
            Height = options.ImageSize,
            // 与 YoloDatasetTrainingRunner 的训练增强保持一致（小数据集靠它补多样性）。
            HorizontalFlipProbability = 0.5f,
            HueGain = 0.015f,
            SaturationGain = 0.7f,
            ValueGain = 0.4f,
        });
        using var validationLoader = new YoloSampleLoader(new YoloPreprocessOptions
        {
            Width = options.ImageSize,
            Height = options.ImageSize,
        });

        var reports = new List<TrainProgress>(options.Epochs);
        float bestMap50 = 0f;
        float lastMap95 = 0f;

        IReadOnlyList<EpochMetrics> history = trainer.Fit(
            epoch => BuildEpochBatches(epoch),
            schedule,
            metrics =>
            {
                (float map50, float map95) = EvaluateMap(
                    model, validationSet, validationLoader, options, classCount, device);

                var item = new TrainProgress(
                    metrics.Epoch, options.Epochs, metrics.Step, metrics.Loss, metrics.LearningRate, map50, map95);

                reports.Add(item);
                lastMap95 = map95;

                log?.Invoke(
                    $"epoch {metrics.Epoch + 1}/{options.Epochs}  loss {metrics.Loss:F4}  "
                    + $"mAP50 {map50:F3}  mAP50-95 {map95:F3}");

                // 先报进度再存盘：曲线不必等磁盘 IO。
                progress?.Report(item);

                if (map50 > bestMap50)
                {
                    bestMap50 = map50;
                    Save("best.safetensors", metrics, map50, map95);
                }

                if (options.SaveEvery > 0 && (metrics.Epoch + 1) % options.SaveEvery == 0)
                    Save("model.safetensors", metrics, map50, map95);
            },
            control);

        EpochMetrics last = history[^1];

        // 所有轮次 mAP 都为 0 时也要留一份可导出的 best。
        if (bestMap50 <= 0f)
            Save("best.safetensors", last, bestMap50, lastMap95);

        Save("model.safetensors", last, bestMap50, lastMap95);

        Checkpoint.Save(
            Path.Combine(output, "model-ema.safetensors"),
            ema.Shadow,
            BuildMetadata(last, last.Loss, bestMap50, lastMap95, model, options.ImageSize, options.Device));

        WriteReport(output, options, model, reports, bestMap50);
        log?.Invoke($"checkpoint: {Path.Combine(output, "model.safetensors")}");
        log?.Invoke($"report    : {Path.Combine(output, "train.json")}");

        return history;

        IEnumerable<Batch> BuildEpochBatches(int epoch)
        {
            IReadOnlyList<YoloSample> samples = trainSet.Samples;

            // 每轮换一次顺序 + 重新采样增强，别让同一批图总落在同一条梯度方向上。
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
                        offset => images[offset] = trainLoader.Load(
                            samples[order[start + offset]],
                            new Rng((ulong)(epoch * 7919L + start + offset))));

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

        void Save(string fileName, EpochMetrics metrics, float map50, float map95)
        {
            Checkpoint.Save(
                Path.Combine(output, fileName),
                model,
                BuildMetadata(metrics, metrics.Loss, map50, map95, model, options.ImageSize, options.Device));

            // 优化器状态用 TorchSharp 自己的 bin 格式（模型权重仍是可互转的 safetensors）。
            optimizer.save_state_dict(Path.Combine(output, "optimizer.bin"));
        }
    }

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


