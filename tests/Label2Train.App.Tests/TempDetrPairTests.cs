using System.Diagnostics;
using System.Text;
using ImageFacade.SkiaSharp;
using Label2Train.Core.Imaging;
using Label2Train.Train.Core;
using Label2Train.Train.Sessions;
using Label2Train.Train.Training;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 临时：增强开 / 关的严格配对实验。一组一个进程（同一个种子才会落在同一起点上），
/// 用环境变量选组：L2T_PAIR_SEED / L2T_PAIR_AUGMENT / L2T_PAIR_EPOCHS / L2T_PAIR_BATCH。
/// 不设 L2T_PAIR_SEED 时整个用例直接返回，不会污染常规测试。
/// </summary>
public sealed class TempDetrPairTests
{
    [Fact]
    public async Task Run_One_Arm()
    {
        string? seedText = Environment.GetEnvironmentVariable("L2T_PAIR_SEED");

        if (string.IsNullOrWhiteSpace(seedText))
            return;

        int seed = int.Parse(seedText, System.Globalization.CultureInfo.InvariantCulture);
        bool augment = Environment.GetEnvironmentVariable("L2T_PAIR_AUGMENT") != "0";
        int epochs = int.Parse(
            Environment.GetEnvironmentVariable("L2T_PAIR_EPOCHS") ?? "20",
            System.Globalization.CultureInfo.InvariantCulture);
        int batch = int.Parse(
            Environment.GetEnvironmentVariable("L2T_PAIR_BATCH") ?? "2",
            System.Globalization.CultureInfo.InvariantCulture);

        string tag = $"seed{seed}-aug{(augment ? "on" : "off")}-bs{batch}";
        string report = Path.Combine(Path.GetTempPath(), $"l2t-pair-{tag}.txt");
        string dataYaml = Path.Combine(
            "F:/code/Label2Train/src/Label2Train.App/bin/Debug/net10.0/data/projects/4/models/rfdetr/dataset",
            "data.yaml");

        // 会话不会自己去 data/weights 里找权重，必须显式给：没有官方 medium 权重就是随机初始化，
        // 那跑出来的是另一个实验（实测 20 轮 mAP50 只有 0.26，有权重是 0.7 量级）
        string weights = Path.Combine(
            "F:/code/Label2Train/src/Label2Train.App/bin/Debug/net10.0/data/weights",
            "rf-detr-medium.safetensors");

        var text = new StringBuilder();
        text.AppendLine($"arm    : {tag}");
        text.AppendLine($"seed   : {seed}（进程内第一次建模型，配对成立）");
        text.AppendLine($"augment: {(augment ? "开 hflip/scale/translate/hsv" : "关 纯 letterbox")}");
        text.AppendLine($"config : {epochs} 轮 · batch {batch} · imgsz 224 · Q 50 · lr 1e-4 · EvalEvery 2 · 100 张验证图");
        text.AppendLine($"data   : {dataYaml}");
        text.AppendLine($"weights: {weights}");

        void Write(string line)
        {
            File.AppendAllText(report, line + Environment.NewLine, Encoding.UTF8);
        }

        Write(text.ToString().TrimEnd());

        ImageRuntime.Use(new SkiaBackend());

        // 必须在建模型之前：tokenizer / 权重初始化都吃全局 RNG
        TorchInterop.SetSeed(seed);

        var options = new DetrTrainingOptions
        {
            DataYaml = dataYaml,
            Epochs = epochs,
            BatchSize = batch,
            ImageSize = 224,
            LearningRate = 1e-4f,
            Queries = 50,
            UseRfDetr = true,
            Augment = augment,
            EvalEvery = 2,
            EvaluationLimit = 100,
            DinoWeights = weights,
            OutputDirectory = Path.Combine(Path.GetTempPath(), $"l2t-pair-{tag}"),
        };

        int lastEpoch = -1;
        var best = new Dictionary<int, (float Map50, float Map5095)>();
        var watch = Stopwatch.StartNew();

        var progress = new Progress<TrainProgress>(item =>
        {
            if (item.Epoch != lastEpoch)
            {
                if (lastEpoch >= 0)
                {
                    (float Map50, float Map5095) done = best.TryGetValue(lastEpoch, out (float Map50, float Map5095) prev)
                        ? prev
                        : (0f, 0f);

                    Write(
                        $"e{lastEpoch + 1,-3} {watch.Elapsed.TotalSeconds,7:F1} s   "
                        + $"mAP50 {done.Map50:F3}   mAP50-95 {done.Map5095:F3}");
                    watch.Restart();
                }

                lastEpoch = item.Epoch;
            }

            if (item.Map50 > 0 || item.Map50_95 > 0)
            {
                if (!best.TryGetValue(item.Epoch, out (float Map50, float Map5095) current)
                    || item.Map50 > current.Map50)
                {
                    best[item.Epoch] = (item.Map50, item.Map50_95);
                }
            }
        });

        using var session = new DetrTrainingSession();
        bool weightsApplied = false;

        IReadOnlyList<EpochMetrics> metrics = await session.StartAsync(
            options,
            progress,
            line =>
            {
                if (line.StartsWith("rfdetr", StringComparison.OrdinalIgnoreCase)
                    || line.Contains("augment", StringComparison.OrdinalIgnoreCase))
                {
                    Write($"log    : {line}");
                }

                if (line.StartsWith("rfdetr:", StringComparison.OrdinalIgnoreCase))
                    weightsApplied = true;
            });

        if (!weightsApplied)
            throw new InvalidOperationException($"官方权重没装上：{weights}（跑出来的曲线不可与 README 表格对比）");

        Write(string.Empty);
        Write("epoch   mAP50   mAP50-95");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            (float Map50, float Map5095) value = best.TryGetValue(epoch, out (float Map50, float Map5095) found)
                ? found
                : (0f, 0f);

            Write($"e{epoch + 1,-6} {value.Map50,6:F3}   {value.Map5095,6:F3}");
        }

        Write(string.Empty);
        Write($"loss    : {metrics[^1].Loss:F3}");
        Write($"finished: {DateTime.Now:HH:mm:ss}");
    }
}
