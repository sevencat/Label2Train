using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Label2Train.Core.Dataset;
using Label2Train.Core.Simulation;
using Label2Train.Core.Storage;
using Label2Train.Train.Core;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Data.Yolo;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training;
using Label2Train.Train.Training.Yolo;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Runners;

/// <summary>
/// 真训练引擎：把 <see cref="IDatasetStore"/>（YOLO / COCO / 图片目录+数据库三种形态统一后的样本）
/// 喂给 YOLO 训练循环，产出 safetensors 检查点与 <c>train.json</c>。
/// <para>
/// 与 OmniTensorSharp 的原实现差异只有一处：数据来源从「data.yaml / COCO json 直接读盘」
/// 换成 <see cref="IDatasetStore"/>，图片字节走 <see cref="IBlobStore"/>，
/// 因此本地目录、数据库、将来的远端后端都走同一条训练路径。
/// </para>
/// </summary>
public sealed class YoloDatasetTrainingRunner(
    BlobStoreRegistry registry,
    ILogger<YoloDatasetTrainingRunner>? logger = null) : IDatasetTrainingRunner
{
    /// <summary>处理器个数上限：批内并行解码的线程数，别让 8 张图占满 32 个核。</summary>
    private static readonly int Parallelism = Math.Max(1, Math.Min(Environment.ProcessorCount, 8));

    /// <summary>blob 存储未必线程安全，取流仍串行。</summary>
    private readonly object _storeGate = new();

    /// <inheritdoc />
    public async Task<TrainingRunSummary> TrainAsync(
        IDatasetStore store,
        TrainingRunOptions options,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);

        int classCount = store.Classes.Count;
        if (classCount <= 0)
        {
            throw new InvalidOperationException($"{store.Id} 里没有类别定义（ClassCatalog 为空）。");
        }

        (string trainSplit, string? valSplit) = ResolveSplits(store, options);

        IReadOnlyList<DatasetSample> trainSamples = await store
            .OpenSplit(trainSplit)
            .LoadSamplesAsync(ct)
            .ConfigureAwait(false);

        if (trainSamples.Count == 0)
        {
            throw new InvalidOperationException($"split {trainSplit} 里没有样本。");
        }

        IReadOnlyList<DatasetSample> validationSamples = valSplit is null
            ? []
            : await store.OpenSplit(valSplit).LoadSamplesAsync(ct).ConfigureAwait(false);

        // 没有 val split 就从训练集里留 20%：mAP 只能在验证集上算，不留的话曲线上的
        // mAP50 / mAP50-95 永远是 0（库里的图片通常只有 train，split 为空）
        bool autoValidation = validationSamples.Count == 0;

        if (autoValidation)
        {
            (trainSamples, validationSamples) = ValidationHoldout.SplitInMemory(trainSamples);
        }

        int totalBoxes = trainSamples.Sum(sample => sample.Boxes.Count)
                         + validationSamples.Sum(sample => sample.Boxes.Count);

        if (!options.Output.IsFile)
        {
            throw new NotSupportedException("训练产物目前只支持本地目录（file://）。");
        }

        string output = options.Output.Root;
        Directory.CreateDirectory(output);

        ComputeDevice computeDevice = DeviceFactory.Parse(options.Device);
        Device device = DeviceFactory.Create(computeDevice);
        YoloScale scale = YoloScales.Parse(options.Scale);
        YoloModelFamily family = YoloScales.ParseFamily(options.Family);
        YoloDetectionModel model = YoloArchitectures.Create(device, family, scale, classCount);

        // 创建时按 640 设的分类 bias 先验，这里按真实输入尺寸再覆盖一次。
        model.Head.ApplyClassBiasPrior(options.ImageSize);

        Log(options, $"data  : {store.Id} · {trainSamples.Count} 张 · {classCount} 类 · {totalBoxes} 个框");
        Log(options, $"val   : {(validationSamples.Count == 0
            ? "(无验证集，mAP 恒为 0)"
            : $"{validationSamples.Count} 张{(autoValidation ? "（自动从训练集留出 20%）" : string.Empty)}")}");
        Log(options, $"model : {model.Architecture} ({YoloArchitectures.CountParameters(model)} params)");
        Log(options, $"device: {DeviceFactory.DescribeDetailed(computeDevice)}");
        Log(options, $"output: {output}");

        var optimizer = torch.optim.AdamW(model.parameters(), lr: options.LearningRate, weight_decay: 0.0005f);
        var loss = new YoloDetectionLoss(options.ImageSize, options.ImageSize, classCount) { TrackComponents = true };

        int batchesPerEpoch = Math.Max(1, (trainSamples.Count + options.BatchSize - 1) / options.BatchSize);
        int totalSteps = (options.Epochs * batchesPerEpoch) + 1;
        int warmupSteps = Math.Min(Math.Max(1, 3 * batchesPerEpoch), Math.Max(0, totalSteps - 1));

        // 小数据集只有几百步：固定的 0.999 会让影子权重停在初始值附近，按总步数换算衰减。
        var ema = new Ema(model, decay: EmaDecayFor(totalSteps));
        var schedule = new CosineSchedule(options.LearningRate, totalSteps, warmupSteps, options.LearningRate * 0.01f);

        YoloPreprocessOptions augment = YoloPreprocessOptions.TrainDefault(options.ImageSize);

        Log(options,
            $"hyper : lr {options.LearningRate:G3} · steps {totalSteps} · warmup {warmupSteps} · "
            + $"ema {ema.Decay:F4} · hflip {augment.HorizontalFlipProbability:G2} · "
            + $"mosaic {augment.MosaicProbability:G2} · scale ±{augment.ScaleGain:G2} · "
            + $"translate ±{augment.TranslateGain:G2} · "
            + $"hsv {augment.HueGain:G2},{augment.SaturationGain:G2},{augment.ValueGain:G2}");

        var trainer = new Trainer(
            model,
            optimizer,
            loss,
            new TrainerOptions { Epochs = options.Epochs, GradientClipNorm = 10f, Shuffle = false },
            new Rng(1),
            ema);

        using var trainLoader = new YoloSampleLoader(augment);
        using var validationLoader = new YoloSampleLoader(
            YoloPreprocessOptions.InferenceDefault(options.ImageSize));

        float bestMap50 = 0f;
        float lastMap50 = 0f;
        float lastMap95 = 0f;
        bool bestSaved = false;

        // 训练前先落一份「起点」模型：中途取消也能拿到可用权重。
        Save("model.safetensors", new EpochMetrics(0, 0, float.NaN, options.LearningRate));

        IReadOnlyList<EpochMetrics> history;

        // 界面传进来的控制器由界面负责释放；没传就自己建一个（含取消令牌），用完释放。
        bool ownsControl = options.Control is not TrainControl;
        TrainControl control = ownsControl ? new TrainControl(ct) : (TrainControl)options.Control!;

        try
        {
            history = trainer.Fit(
                BatchesForEpoch,
                schedule,
                metrics =>
                {
                    int completed = metrics.Epoch + 1;
                    bool evaluate = options.EvalEvery <= 0
                        || completed % options.EvalEvery == 0
                        || completed >= options.Epochs;

                    float map50 = lastMap50;
                    float map95 = lastMap95;
                    double evalSeconds = 0;

                    if (evaluate)
                    {
                        var evalWatch = System.Diagnostics.Stopwatch.StartNew();
                        (map50, map95) = EvaluateMap(
                            model, validationSamples, validationLoader, options, classCount, device, ct);
                        evalWatch.Stop();
                        evalSeconds = evalWatch.Elapsed.TotalSeconds;

                        lastMap50 = map50;
                        lastMap95 = map95;

                        // 第一次评估就留一份 best：mAP 全为 0（随机初始化）时也要有可导出的权重。
                        if (!bestSaved || map50 > bestMap50)
                        {
                            bestSaved = true;
                            bestMap50 = map50;
                            Save("best.safetensors", metrics, map50, map95);
                        }
                    }

                    // 把一轮拆成「等数据 / 算梯度 / 评估」三段：GPU 慢还是数据管道慢，看这一行就够了。
                    // 两个计时都是整轮累计，除一下步数才是每步耗时
                    int steps = Math.Max(1, trainer.LastBatchCount);

                    Log(options,
                        $"epoch {completed}/{options.Epochs}  loss {metrics.Loss:F4}  "
                        + $"mAP50 {map50:F3}  mAP50-95 {map95:F3}  "
                        + $"time {trainer.LastComputeSeconds + trainer.LastDataSeconds:F1}s / {steps} step "
                        + $"(data {trainer.LastDataSeconds / steps:F2}s + step {trainer.LastComputeSeconds / steps:F2}s) "
                        + (evaluate ? $"+ eval {evalSeconds:F1}s" : "+ eval skipped"));

                    options.Progress?.Report(new TrainingProgress(
                        metrics.Epoch, options.Epochs, metrics.Step, metrics.Loss, metrics.LearningRate, map50, map95));

                    // 权重 + AdamW 状态与模型同量级，每轮都落盘会在几百轮时写掉几十 GB。
                    if (completed >= options.Epochs
                        || (options.SaveEvery > 0 && completed % options.SaveEvery == 0))
                    {
                        Save("model.safetensors", metrics, map50, map95);
                    }
                },
                control);
        }
        finally
        {
            if (ownsControl)
            {
                control.Dispose();
            }
        }

        ct.ThrowIfCancellationRequested();

        EpochMetrics last = history.Count > 0
            ? history[^1]
            : new EpochMetrics(0, 0, float.NaN, options.LearningRate);

        // 评估被节流且中途停止时可能一次评估都没做：兜底把最终权重当成 best，导出才有文件可用。
        if (!bestSaved)
        {
            bestSaved = true;
            Save("best.safetensors", last, lastMap50, lastMap95);
        }

        Save("model.safetensors", last, lastMap50, lastMap95);

        Checkpoint.Save(
            Path.Combine(output, "model-ema.safetensors"),
            ema.Shadow,
            BuildMetadata(last, last.Loss, bestMap50, lastMap95, model, options.ImageSize, options.Device));

        WriteReport(output, store, options, model, history, bestMap50);

        Log(options, $"checkpoint: {Path.Combine(output, "model.safetensors")}");
        Log(options, $"report    : {Path.Combine(output, "train.json")}");

        logger?.LogInformation(
            "训练完成：{Dataset} {Epochs} 轮，best mAP50 {Map50:F3}，产物目录 {Output}",
            store.Id, options.Epochs, bestMap50, output);

        return new TrainingRunSummary(
            store.Id,
            trainSamples.Count + validationSamples.Count,
            totalBoxes,
            [.. store.Classes.Names],
            validationSamples.Count == 0 ? [trainSplit] : [trainSplit, "val"],
            options.Output,
            DryRun: false,
            [])
        {
            BestMap50 = bestMap50,
            BestMap50_95 = lastMap95,
            Architecture = model.Architecture,
        };

        YoloImageSample LoadOne(int sampleIndex, Rng rng)
        {
            DatasetSample sample = trainSamples[sampleIndex];
            Stream stream;
            lock (_storeGate)
                stream = OpenImage(sample);

            using (stream)
            {
                return trainLoader.Load(stream, sample.Image.RelativePath, ToLabels(sample), rng);
            }
        }

        // 取打乱序列里相邻的 4 张拼贴：小数据集靠它补「多目标 + 不同尺度」的样本。
        YoloImageSample LoadMosaic(int[] shuffleOrder, int position, Rng rng)
        {
            var parts = new YoloImageSample[4];
            for (int index = 0; index < parts.Length; index++)
            {
                parts[index] = LoadOne(shuffleOrder[(position + index) % shuffleOrder.Length], rng);
            }

            return trainLoader.Mosaic(parts, rng);
        }

        IEnumerable<Batch> BatchesForEpoch(int epoch)
        {
            // 每个 epoch 重新构建批次：打乱顺序 + 重新采样增强，避免增强结果被缓存。
            var rng = new Rng((ulong)(0x51ED270B + epoch * 7919));
            int[] order = Shuffle(Enumerable.Range(0, trainSamples.Count).ToArray(), rng);

            Batch? previous = null;
            try
            {
                for (int start = 0; start < order.Length; start += options.BatchSize)
                {
                    int count = Math.Min(options.BatchSize, order.Length - start);
                    var images = new YoloImageSample[count];

                    // 解码 + letterbox + 增强是纯 CPU 的，一批 8 张串行要 0.6-0.9 s，而 GPU 一步才 ~0.1 s：
                    // 不铺开的话数据准备就是整条流水线的瓶颈。各张图互不依赖，按批并行即可；
                    // blob 存储的访问留在锁里（存储未必线程安全）。
                    Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = Parallelism }, offset =>
                    {
                        int position = start + offset;
                        var rng = new Rng((ulong)((epoch * 7919L) + position));

                        images[offset] = augment.MosaicProbability > 0f
                            && rng.NextFloat() < augment.MosaicProbability
                            ? LoadMosaic(order, position, rng)
                            : LoadOne(order[position], rng);
                    });

                    YoloBatchResult packed = YoloBatchBuilder.Build(images);

                    // 上一批已训练完（backward 结束），先把它的显存还回去再准备下一批。
                    previous?.Dispose();
                    previous = new Batch(
                        TorchInterop.FromArray(
                            packed.Inputs,
                            [packed.BatchSize, packed.Channels, packed.Height, packed.Width],
                            device),
                        TorchInterop.FromArray(
                            packed.Targets,
                            [packed.BatchSize, packed.MaxBoxes, YoloBatchBuilder.TargetColumns],
                            device));

                    yield return previous;
                }
            }
            finally
            {
                previous?.Dispose();
            }
        }

        void Save(string fileName, EpochMetrics metrics, float map50 = 0f, float map95 = 0f)
        {
            Checkpoint.Save(
                Path.Combine(output, fileName),
                model,
                BuildMetadata(metrics, metrics.Loss, map50, map95, model, options.ImageSize, options.Device));

            // 优化器状态用 TorchSharp 自己的 bin 格式（模型权重仍是可互转的 safetensors）。
            optimizer.save_state_dict(Path.Combine(output, "optimizer.bin"));
        }
    }

    /// <summary>取训练 / 验证 split：优先显式指定，其次按名字猜（train / val）。</summary>
    private static (string Train, string? Validation) ResolveSplits(IDatasetStore store, TrainingRunOptions options)
    {
        IReadOnlyList<string> splits = options.Splits is { Count: > 0 } ? options.Splits : store.Splits;
        if (splits.Count == 0)
        {
            throw new InvalidOperationException("没有可训练的 split：去标注端给图片分配 train/val。");
        }

        string train = splits.FirstOrDefault(split => IsTrain(split)) ?? splits[0];
        string? validation = splits.FirstOrDefault(split => IsValidation(split) && !string.Equals(split, train, StringComparison.OrdinalIgnoreCase));

        return (train, validation);
    }

    private static bool IsTrain(string split)
        => split.Contains("train", StringComparison.OrdinalIgnoreCase);

    private static bool IsValidation(string split)
        => split.Contains("val", StringComparison.OrdinalIgnoreCase);

    private Stream OpenImage(DatasetSample sample)
    {
        IBlobStore blob = registry.Get(sample.Image.Location);
        var key = new StorageKey(sample.Image.Location, sample.Image.RelativePath);
        return blob.OpenReadAsync(key).GetAwaiter().GetResult();
    }

    /// <summary>像素 xyxy → 归一化 YOLO 标签（cx cy w h），并夹到 [0,1]。</summary>
    private static IReadOnlyList<YoloLabel> ToLabels(DatasetSample sample)
    {
        if (sample.Boxes.Count == 0)
        {
            return [];
        }

        if (sample.Width <= 0 || sample.Height <= 0)
        {
            throw new InvalidDataException($"图片尺寸无效：{sample.Image.RelativePath} {sample.Width}x{sample.Height}");
        }

        var labels = new List<YoloLabel>(sample.Boxes.Count);

        foreach (BoxAnnotation box in sample.Boxes)
        {
            float width = Clamp01((box.X2 - box.X1) / sample.Width);
            float height = Clamp01((box.Y2 - box.Y1) / sample.Height);
            float centerX = Clamp01((box.X1 + box.X2) / 2f / sample.Width);
            float centerY = Clamp01((box.Y1 + box.Y2) / 2f / sample.Height);

            labels.Add(new YoloLabel(box.ClassId, YoloTask.Detect, [centerX, centerY, width, height]));
        }

        return labels;
    }

    private static float Clamp01(float value) => Math.Clamp(value, 0f, 1f);

    /// <summary>在验证集上算 mAP：解码 → NMS → 与真值比对。</summary>
    private (float Map50, float Map50_95) EvaluateMap(
        YoloDetectionModel model,
        IReadOnlyList<DatasetSample> validationSamples,
        YoloSampleLoader validationLoader,
        TrainingRunOptions options,
        int classCount,
        Device device,
        CancellationToken ct)
    {
        if (validationSamples.Count == 0)
        {
            return (0f, 0f);
        }

        var predictions = new List<IReadOnlyList<YoloDetection>>(validationSamples.Count);
        var groundTruths = new List<IReadOnlyList<YoloGroundTruth>>(validationSamples.Count);

        model.eval();

        try
        {
            using (torch.no_grad())
            {
                foreach (DatasetSample sample in validationSamples)
                {
                    ct.ThrowIfCancellationRequested();

                    // 单张前向的激活也是几百 MB：验证集一大，不逐张释放就会把显存撑爆。
                    using (var scope = torch.NewDisposeScope())
                    {
                        YoloImageSample image;
                        using (Stream stream = OpenImage(sample))
                        {
                            image = validationLoader.Load(stream, sample.Image.RelativePath, ToLabels(sample));
                        }

                        Tensor input = TorchInterop.FromArray(
                            image.ToChw(),
                            [1, 3, options.ImageSize, options.ImageSize],
                            device);

                        predictions.Add(YoloPredictionDecoder.Decode(
                            model.forward(input),
                            options.ImageSize,
                            options.ImageSize,
                            classCount,
                            DetectHeadModule.DefaultRegressionMax,
                            confidenceThreshold: 0.001f,
                            iouThreshold: 0.7f));

                        groundTruths.Add(image.Boxes
                            .Select(box => new YoloGroundTruth(box.ClassId, box.X1, box.Y1, box.X2, box.Y2))
                            .ToArray());

                        scope.Dispose();
                    }
                }
            }
        }
        finally
        {
            model.train();
        }

        DetectionMetricsResult result = DetectionMetrics.Evaluate(predictions, groundTruths, classCount);
        return (result.Map50, result.Map50_95);
    }

    private static Dictionary<string, string> BuildMetadata(
        EpochMetrics metrics,
        float lossValue,
        float map50,
        float map95,
        YoloDetectionModel model,
        int imageSize,
        string device)
    {
        return new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["format"] = Checkpoint.FormatName,
            ["architecture"] = model.Architecture,
            ["epoch"] = metrics.Epoch.ToString(CultureInfo.InvariantCulture),
            ["step"] = metrics.Step.ToString(CultureInfo.InvariantCulture),
            ["loss"] = lossValue.ToString("G9", CultureInfo.InvariantCulture),
            ["map50"] = map50.ToString("G6", CultureInfo.InvariantCulture),
            ["map50-95"] = map95.ToString("G6", CultureInfo.InvariantCulture),
            ["backend"] = DeviceFactory.Describe(device),
            ["classes"] = model.ClassCount.ToString(CultureInfo.InvariantCulture),
            ["image_size"] = imageSize.ToString(CultureInfo.InvariantCulture),
        };
    }

    private static void WriteReport(
        string output,
        IDatasetStore store,
        TrainingRunOptions options,
        YoloDetectionModel model,
        IReadOnlyList<EpochMetrics> history,
        float bestMap50)
    {
        var report = new
        {
            dataset = store.Id,
            format = store.Format.ToId(),
            architecture = model.Architecture,
            backend = DeviceFactory.Describe(options.Device),
            classes = store.Classes.Names,
            epochs = options.Epochs,
            batchSize = options.BatchSize,
            imageSize = options.ImageSize,
            learningRate = options.LearningRate,
            bestMap50,
            history = history.Select(item => new
            {
                item.Epoch,
                item.Step,
                item.Loss,
                item.LearningRate,
            }).ToArray(),
        };

        File.WriteAllText(
            Path.Combine(output, "train.json"),
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>
    /// 按总步数选 EMA 衰减：<c>1 - 3/steps</c>，夹在 <c>[0.9, 0.999]</c>。
    /// <para>
    /// 30 张图 × 100 轮只有几百步，固定 0.999 的等效记忆长度远超训练步数，
    /// EMA 权重几乎等于初始权重；按步数换算后短训练也能得到有意义的平均权重。
    /// </para>
    /// </summary>
    private static float EmaDecayFor(int totalSteps)
        => Math.Clamp(1f - 3f / Math.Max(1, totalSteps), 0.9f, 0.999f);

    private static int[] Shuffle(int[] order, Rng rng)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }

        return order;
    }

    /// <summary>训练输出既给界面回调，也进 NLog——「日志」窗口和 data/logs 才看得到每轮进度。</summary>
    private void Log(TrainingRunOptions options, string message)
    {
        options.Log?.Invoke(message);
        logger?.LogInformation("YOLO {Message}", message);
    }
}
