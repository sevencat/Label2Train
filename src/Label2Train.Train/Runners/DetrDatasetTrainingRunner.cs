using Microsoft.Extensions.Logging;
using Label2Train.Core.Dataset;
using Label2Train.Core.Dataset.Export;
using Label2Train.Core.Simulation;
using Label2Train.Core.Storage;
using Label2Train.Train.Core;
using Label2Train.Train.Models.RfDetr;
using Label2Train.Train.Sessions;
using Label2Train.Train.Training;

namespace Label2Train.Train.Runners;

/// <summary>
/// RF-DETR 训练入口：DETR 的数据加载器读盘上的 data.yaml，所以先把
/// <see cref="IDatasetStore"/>（YOLO / COCO / 数据库都行）导出成一份 YOLO 数据集，
/// 再交给 <see cref="DetrTrainingSession"/>。
/// <para>
/// 数据集落在 run 目录下的 <c>dataset/</c>，和检查点、<c>train.json</c> 放在一起：
/// 训练时到底喂了什么图、标签长什么样，用户能直接翻，也方便复现问题。
/// </para>
/// </summary>
public sealed class DetrDatasetTrainingRunner(
    DatasetExportService exporter,
    ILogger<DetrDatasetTrainingRunner>? logger = null) : IDatasetTrainingRunner
{
    /// <summary>run 目录里训练数据副本的子目录名。</summary>
    public const string DatasetFolderName = "dataset";

    /// <summary>DETR 显存占用高，批大小按上限截断，避免直接 OOM。</summary>
    private const int MaxBatchSize = 4;

    /// <inheritdoc />
    public async Task<TrainingRunSummary> TrainAsync(
        IDatasetStore store,
        TrainingRunOptions options,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);

        if (store.Classes.Count == 0)
        {
            throw new InvalidOperationException($"{store.Id} 里没有类别定义（ClassCatalog 为空）。");
        }

        if (!options.Output.IsFile)
        {
            throw new NotSupportedException("训练产物目前只支持本地目录（file://）。");
        }

        string[] splits = options.Splits is { Count: > 0 } ? [.. options.Splits] : [.. store.Splits];

        if (splits.Length == 0)
        {
            throw new InvalidOperationException("数据集里没有可训练的 split（先在标注端划分 train / val）。");
        }

        string output = options.Output.Root;
        Directory.CreateDirectory(output);

        // 训练数据副本放在 run 目录里，而不是 %TEMP%：用户要能看到这次训练用的是哪些图
        string staging = Path.Combine(output, DatasetFolderName);

        if (Directory.Exists(staging))
        {
            // 上一轮的 split 目录会残留旧图，先清空再导出
            Directory.Delete(staging, recursive: true);
        }

        string trainSplit = splits.FirstOrDefault(
                split => string.Equals(split, "train", StringComparison.OrdinalIgnoreCase))
            ?? splits[0];

        bool hasValidation = splits.Any(
            split => split.Contains("val", StringComparison.OrdinalIgnoreCase));

        Log(options, "RF-DETR：先导出 YOLO 数据集（DETR 数据加载器读 data.yaml）…");

        DatasetExportReport report = await exporter
            .ExportAsync(
                store,
                new DatasetExportOptions(
                    DatasetFormat.Yolo,
                    StorageLocation.FromPath(staging),
                    splits,
                    CopyImages: true),
                ct)
            .ConfigureAwait(false);

        string dataYaml = Path.Combine(staging, "data.yaml");

        Log(options, $"export: {report.Summary}");
        Log(options, $"data  : {dataYaml}");

        // 没有 val split 就从训练集里留 20%：不留的话 mAP 只能在空集上算，永远是 0，
        // 曲线看着像「训练没生效」（库里的数据集通常只有 train，split 为空）
        int heldOut = 0;

        if (!hasValidation)
        {
            heldOut = ValidationHoldout.SplitExportedYoloSplit(staging, trainSplit);

            if (heldOut > 0 && ValidationHoldout.AddValidationSplit(dataYaml))
            {
                Log(options, $"val   : 自动从训练集留出 {heldOut} 张（20%，按文件名固定取样）");
            }
            else
            {
                heldOut = 0;
                Log(options, "val   : 没有验证集（图片少于 5 张），mAP 会一直是 0");
            }
        }

        // 边长跟着界面设置走：主干的 patch（14 / 官方权重 16）与检测头下采样 16 都要求 112 的倍数。
        // 非法值就近纠正并写日志——以前这里硬编码 224，界面上填 560 也按 224 训练，很难发现
        int imageSize = TrainingPresets.ResolveDetrImageSize(options.ImageSize);

        Log(options, imageSize == options.ImageSize
            ? $"imgsz : {imageSize}（RF-DETR 的输入边长，见 train.json）"
            : $"imgsz : {options.ImageSize} → {imageSize}（RF-DETR 要求 {TrainingPresets.DetrImageSizeStride} 的倍数）");

        string[] reportedSplits = heldOut > 0 ? [.. splits, "val"] : splits;

        // 有官方权重就热启动：30 张图从零训 DETR，mAP 几十轮都是 0，几乎看不出在学
        string? pretrained = RfDetrWeightSource.Resolve();

        Log(options, pretrained is null ? RfDetrWeightSource.Hint : $"rfdetr: 预训练权重 {pretrained}");

        var detrOptions = new DetrTrainingOptions
        {
            DataYaml = dataYaml,
            Epochs = options.Epochs,
            BatchSize = Math.Clamp(options.BatchSize, 1, MaxBatchSize),
            ImageSize = imageSize,
            // 预训练权重微调用官方 AdamW 的 1e-4，从零训才敢用 1e-3。
            // 曾经用 2e-4：batch 被截到 4（有效批量远小于官方的 16），配上 2e-4 会让 loss
            // 在小数据集上震荡回升（2.2 → 3.3 → 2.4 → 3.2），mAP 一直起不来
            LearningRate = pretrained is null ? 1e-3f : 1e-4f,
            OutputDirectory = output,
            Device = DeviceFactory.Parse(options.Device),
            UseRfDetr = true,
            Augment = options.Augment,
            DinoWeights = pretrained ?? string.Empty,
        };

        using var session = new DetrTrainingSession();
        using CancellationTokenRegistration cancellation = ct.Register(session.Stop);

        float bestMap50 = 0;
        float bestMap95 = 0;

        IProgress<TrainProgress>? progress = options.Progress is null
            ? null
            : new Progress<TrainProgress>(item =>
            {
                bestMap50 = Math.Max(bestMap50, item.Map50);
                bestMap95 = Math.Max(bestMap95, item.Map50_95);

                options.Progress.Report(new TrainingProgress(
                    item.Epoch,
                    item.TotalEpochs,
                    item.Step,
                    item.Loss,
                    item.LearningRate,
                    item.Map50,
                    item.Map50_95));
            });

        // 会话自己的每轮输出（epoch / loss 分项 / mAP / 耗时）也走 Log：
        // 以前直接把 options.Log 交下去，那些行只进界面回调、不进 NLog，
        // 「日志」窗口和 data/logs/label.log 里一行训练输出都没有
        Action<string>? sessionLog = options.Log is null ? null : message => Log(options, message);

        Task<IReadOnlyList<EpochMetrics>> training = session.StartAsync(detrOptions, progress, sessionLog);

        // 外部的暂停 / 继续转发给 DETR 会话：训练线程只在批次边界响应，转发是安全的
        using var bridgeCancellation = new CancellationTokenSource();
        Task bridge = Task.Run(
            async () =>
            {
                while (!bridgeCancellation.IsCancellationRequested)
                {
                    if (options.Control is { } control)
                    {
                        if (control.IsPaused)
                        {
                            session.Pause();
                        }
                        else
                        {
                            session.Resume();
                        }
                    }

                    try
                    {
                        await Task.Delay(150, bridgeCancellation.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        return;
                    }
                }
            },
            CancellationToken.None);

        try
        {
            await training.ConfigureAwait(false);
        }
        finally
        {
            await bridgeCancellation.CancelAsync().ConfigureAwait(false);
            await bridge.ConfigureAwait(false);
        }

        return new TrainingRunSummary(
            store.Id,
            report.ImagesExported,
            report.BoxesExported,
            [.. store.Classes.Names],
            reportedSplits,
            options.Output,
            DryRun: false,
            report.Warnings)
        {
            BestMap50 = bestMap50,
            BestMap50_95 = bestMap95,
            Architecture = $"rfdetr-detect-{store.Classes.Count}",
        };
    }

    private void Log(TrainingRunOptions options, string message)
    {
        options.Log?.Invoke(message);
        logger?.LogInformation("RF-DETR {Message}", message);
    }
}

/// <summary>
/// 按 <see cref="TrainingRunOptions.Family"/> 把训练分发给 YOLO 或 RF-DETR 运行器
/// （<c>rfdetr</c> 走 DETR，其余走 YOLO）。
/// </summary>
public sealed class DatasetTrainingRunnerDispatcher(
    YoloDatasetTrainingRunner yolo,
    DetrDatasetTrainingRunner detr) : IDatasetTrainingRunner
{
    /// <inheritdoc />
    public Task<TrainingRunSummary> TrainAsync(
        IDatasetStore store,
        TrainingRunOptions options,
        CancellationToken ct = default)
        => string.Equals(options.Family, "rfdetr", StringComparison.OrdinalIgnoreCase)
            ? detr.TrainAsync(store, options, ct)
            : yolo.TrainAsync(store, options, ct);
}
