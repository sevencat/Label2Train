using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.ExceptionServices;
using Label2Train.Train.Core;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Data.Yolo;
using Label2Train.Train.Training;
using Label2Train.Train.Training.Detr;
using TorchSharp;
using TorchSharp.Modules;
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

        // 有预训练权重时 lr 跟「从零训」不是一个量级：1e-3 在这份数据上实测有 1/4 的初始化会
        // 直接把框/分数推到饱和然后冻住（mAP 恒 0）。跑之前先提醒，比跑完再看曲线猜便宜得多。
        if (options.UseRfDetr
            && !string.IsNullOrWhiteSpace(options.DinoWeights)
            && options.LearningRate > 3e-4f)
        {
            log?.Invoke($"warn  : lr {options.LearningRate:0.###e+0} 对微调偏高（有官方权重时建议 1e-4）："
                + "实测 1e-3 有约 1/4 的初始化会把模型推到饱和后冻住，mAP 一直是 0");
        }

        // IA-BCE 是官方 RF-DETR 的默认分类损失（ia_bce_loss=True）：
        // 负样本权重 p²、正样本 clamp(p^α · IoU^(1-α), 0.01)，其余权重走 DetrLossOptions 默认值。
        var lossOptions = new DetrLossOptions { IouAwareBce = options.IouAwareBce };

        if (options.IouAwareBce)
            log?.Invoke("loss  : IA-BCE（neg=p², pos=clamp(p^0.25·IoU^0.75, 0.01)）");

        var loss = new DetrLoss(classCount, options.ImageSize, options.ImageSize, lossOptions, options.GroupDetr);

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
            new TrainerOptions
            {
                Epochs = options.Epochs,
                GradientClipNorm = 1f,
                Shuffle = false,
                // 预训练权重里没有分类头（官方 91 类）和 query/参考点（官方 13 组），
                // 这几块只能随机初始化。Adam 每步只挪 lr：1e-4 跑 20 轮（约 1600 步）权重才动 0.16，
                // 分类头会一直停在「所有 query 都输出 0.5」——mAP 恒为 0、det 一堆、分数一条直线。
                // 全局把 lr 提到 1e-3 能救分类，却会把预训练的框头带崩（giou 0.99 → 1.43），
                // 所以只放大随机初始化那几块的梯度；放大后有效 lr 封顶 1e-3，避免 1e-3 基础 lr
                // 配 ×10 变成 1e-2 把 query/参考点推到饱和（训练会直接冻住）
                GradientScale = name => IsRandomHead(name) ? HeadGradientScale(options.LearningRate) : 1f,
            },
            new Rng(4));

        // 训练集开增强（hflip / 缩放 / 平移 / HSV），验证集保持纯 letterbox：
        // 之前两边参数完全一样，几十张图几十轮就被背下来，加轮数只能拟合噪声。
        // Augment=false 只用于对照实验。
        YoloPreprocessOptions augment = options.Augment
            ? TrainingPresets.DetrTrainPreprocess(options.ImageSize)
            : new YoloPreprocessOptions
            {
                Width = options.ImageSize,
                Height = options.ImageSize,
                SizeStride = TrainingPresets.DetrImageSizeStride,
            };

        using var trainLoader = new YoloSampleLoader(augment);

        log?.Invoke(options.Augment
            ? $"augment: hflip {augment.HorizontalFlipProbability:G2} · scale ±{augment.ScaleGain:G2} · "
                + $"translate ±{augment.TranslateGain:G2} · "
                + $"hsv {augment.HueGain:G2},{augment.SaturationGain:G2},{augment.ValueGain:G2}"
            : "augment: 关闭（纯 letterbox，仅用于对照）");

        // 预处理按 DETR 的步长校验（112 = patch 14 与参考网格 16 的公倍数）。
        // 用 YOLO 缺省的 32 会拦掉 336 / 560 这些合法边长，训练在加载器构造时就失败
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

        // 冻住之后不会自己恢复（参数停在 sigmoid 饱和端），继续跑只是浪费时间。
        // 与其让用户对着 mAP 恒 0 的曲线猜，不如直接点出来；只提示一次，免得每轮刷屏。
        float? previousLoss = null;
        int stagnantEpochs = 0;
        bool stallReported = false;

        IReadOnlyList<EpochMetrics> history = trainer.Fit(
            BuildEpochBatches,
            schedule,
            metrics =>
            {
                DetrEvaluation evaluation = EvaluateMap(
                    model, validationSet, validationLoader, options, classCount, device);

                float map50 = evaluation.Map50;
                float map95 = evaluation.Map50_95;

                var item = new TrainProgress(
                    metrics.Epoch, options.Epochs, metrics.Step, metrics.Loss, metrics.LearningRate, map50, map95);

                reports.Add(item);
                lastMap95 = map95;

                // 分项（加权）跟着每轮打一次：loss 卡在平台时，能立刻分清是分类项
                // 压不下来（样本太少 / 类别不平衡）还是框回归没收敛（匹配或归一化问题）
                (float classPart, float l1Part, float giouPart) = loss.AverageParts;
                loss.ResetParts();

                // 两个累计量都是「整个 epoch」，先除步数才是每步耗时——
                // 之前直接打印累计值，看起来像 8 s/step，其实一整个 epoch 才 8 s
                int steps = Math.Max(1, trainer.LastBatchCount);

                // pairs 是这一步实际匹配上的（query, gt）对数：恒为 0 说明匹配没跑起来，
                // mAP 再怎么等也不会动；time 把「等数据」和「GPU 计算」分开，慢在哪一眼可见
                log?.Invoke($"epoch {metrics.Epoch + 1}/{options.Epochs}  loss {metrics.Loss:F4}"
                    + $"  [cls {classPart:F3} / l1 {l1Part:F3} / giou {giouPart:F3}]"
                    + $"  pairs {loss.LastMatchedPairs}"
                    + $"  time {trainer.LastComputeSeconds + trainer.LastDataSeconds:F1}s / {steps} step"
                    + $" (data {trainer.LastDataSeconds / steps:F2}s + step {trainer.LastComputeSeconds / steps:F2}s)"
                    + $"  mAP50 {map50:F3}  mAP50-95 {map95:F3}"
                    // 验证诊断：iou 是阈值过滤后的检测框、all 是 300 个 query 里不看分数的最佳值。
                    // 两者都接近 0 → 框没学出来；iou 低而 all 高 → 框是对的、分数塌了（分类损失问题）
                    + $"  det {evaluation.DetectionsPerImage:F1}/img  gt {evaluation.TruthsPerImage:F1}/img"
                    + $"  top1 {evaluation.TopScore:F3}"
                    + $"  iou {evaluation.BestIou:F2}/{evaluation.AnyIou:F2}"
                    + $" (>=0.5 {evaluation.Iou50Ratio:P0}/{evaluation.AnyIou50Ratio:P0})"
                    // 梯度范数：健康训练在 1e-2 ~ 1e1 量级，掉到 1e-3 以下基本就是「参数没在动」
                    + $"  grad {trainer.LastGradientNorm:0.##e+0}");

                // 冻结的特征是「loss 两轮降不到 1%，mAP 还是 0」：参数被推到饱和端之后，
                // 剩余梯度只够让 loss 抖动百分之几，曲线看着在动、其实什么也没学到
                if (previousLoss is { } before
                    && map50 <= 0f
                    && Math.Abs(metrics.Loss - before) <= before * 0.01f)
                {
                    stagnantEpochs++;
                }
                else
                {
                    stagnantEpochs = 0;
                }

                previousLoss = metrics.Loss;

                if (stagnantEpochs >= 2 && !stallReported)
                {
                    stallReported = true;
                    log?.Invoke($"warn  : loss 连 {stagnantEpochs + 1} 轮降不到 1%（{metrics.Loss:F4}，"
                        + $"梯度范数 {trainer.LastGradientNorm:0.##e+0}）而 mAP 还是 0，"
                        + "基本可以判定冻住了：sigmoid 已被推到饱和端，继续跑不会恢复；"
                        + "有官方权重时 lr 用 1e-4、从零训才用 1e-3，建议换个 lr 重跑。");
                }

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
            // 表现为 loss 上下震荡而 mAP 起不来（增强之外再叠一层顺序随机性）
            int[] order = ShuffleSamples(samples.Count, new Rng((ulong)(0x51ED270B + epoch * 7919)));

            // 生产者只负责「解码 + letterbox + 上传」，消费者（训练线程）拿到批就开始前向。
            // 队列深度 1 就够重叠：GPU 在算第 N 批时，后台线程正好在准备第 N+1 批；
            // 再深只会多占显存，对吞吐没有帮助。
            using var queue = new BlockingCollection<Batch>(1);
            using var abort = new CancellationTokenSource();
            Exception? failure = null;

            Task producer = Task.Factory.StartNew(
                () =>
                {
                    try
                    {
                        for (int start = 0; start < order.Length; start += options.BatchSize)
                        {
                            abort.Token.ThrowIfCancellationRequested();

                            int count = Math.Min(options.BatchSize, order.Length - start);
                            var images = new YoloImageSample[count];

                            // 增强的种子按「epoch + 样本序号」派生：同一张图每轮长得不一样，
                            // 但同一个 seed 重跑能复现；派生在并行之前做完，避免共享一个 Rng。
                            var seeds = new ulong[count];
                            for (int offset = 0; offset < count; offset++)
                                seeds[offset] = AugmentSeed(epoch, start + offset);

                            // 各张图互不依赖，按批并行解码 + letterbox
                            Parallel.For(
                                0,
                                count,
                                new ParallelOptions { MaxDegreeOfParallelism = Math.Min(count, 8) },
                                offset => images[offset] = trainLoader.Load(
                                    samples[order[start + offset]], rng: new Rng(seeds[offset])));

                            YoloBatchResult packed = YoloBatchBuilder.Build(images);

                            queue.Add(
                                new Batch(
                                    TorchInterop.FromArray(
                                        packed.Inputs,
                                        new long[] { packed.BatchSize, packed.Channels, packed.Height, packed.Width },
                                        device),
                                    TorchInterop.FromArray(
                                        packed.Targets,
                                        new long[] { packed.BatchSize, packed.MaxBoxes, YoloBatchBuilder.TargetColumns },
                                        device)),
                                abort.Token);
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        // 训练提前结束，消费者不再取批：生产者被唤醒后正常收尾
                    }
                    catch (Exception error)
                    {
                        // 生产者线程上的异常不能丢，遍历结束后在消费者线程上原样抛出
                        failure = error;
                    }
                    finally
                    {
                        queue.CompleteAdding();
                    }
                },
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            Batch? previous = null;
            try
            {
                foreach (Batch batch in queue.GetConsumingEnumerable())
                {
                    // 走到这里说明上一批已经跑完 forward/backward，可以还它的显存了：
                    // 此时生产者手上是另一批张量，释放不会撞上正在准备的数据
                    previous?.Dispose();
                    previous = batch;

                    yield return batch;
                }
            }
            finally
            {
                previous?.Dispose();

                // 中途停止 / 抛异常：先打断可能阻塞在 Add 上的生产者，再把队里剩下的一并释放
                abort.Cancel();
                queue.CompleteAdding();

                foreach (Batch leftover in queue.GetConsumingEnumerable())
                    leftover.Dispose();

                producer.Wait();
            }

            if (failure is not null)
                ExceptionDispatchInfo.Capture(failure).Throw();
        }
    }

    /// <summary>每张训练图的增强种子：只由 (epoch, 该轮内的样本序号) 决定，重启训练也能复现。</summary>
    private static ulong AugmentSeed(int epoch, int sample) =>
        0x51ED270BUL
        ^ ((ulong)(uint)epoch * 0x9E3779B97F4A7C15UL)
        ^ ((ulong)(uint)sample * 0xC2B2AE3D27D4EB4FUL);

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
