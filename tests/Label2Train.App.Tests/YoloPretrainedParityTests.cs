using Label2Train.Train.Core;
using Label2Train.Train.IO.SafeTensors;
using Label2Train.Train.Models.Weights;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training.Yolo;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Label2Train.App.Tests;

/// <summary>
/// 与官方的数值对拍：载入 ultralytics 权重后，同一输入的前向输出必须逐元素一致。
/// <para>
/// 参考数据由 <c>tools/convert_yolo_weights.py --parity</c> 生成，放在
/// <c>data/weights/{name}-parity.safetensors</c>。没有文件时跳过（不阻塞常规回归）。
/// </para>
/// <para>
/// 这个测试的价值远大于「参数量对得上」：它一次性证明 backbone / neck / head
/// 的每一层语义（含 BN、激活、拼接顺序、锚点顺序、DFL 布局、解码公式）都与官方一致。
/// </para>
/// </summary>
public sealed class YoloPretrainedParityTests
{
    private const int ClassCount = 80;

    private const float Tolerance = 1e-3f;

    private const float RawTolerance = 1e-4f;

    /// <summary>
    /// one-to-one 分支的原始输出容差：分类分支的 logits 量级到 ~37，
    /// float32 累加的相对误差 ~1e-5 在这里就是 4e-4 的绝对误差（实测 1.3e-4），
    /// 所以这一路只能放宽到 1e-3 仍然能抓住真正的结构差异（那会到 1e-2 以上）。
    /// </summary>
    private const float OneToOneRawTolerance = 1e-3f;

    private static string? Find(string name)
    {
        foreach (string directory in PretrainedWeights.Directories)
        {
            string path = Path.Combine(directory, name);

            if (File.Exists(path))
            {
                return path;
            }
        }

        return null;
    }

    private static float MaxDiff(Tensor actual, Tensor expected)
    {
        using Tensor diff = (actual - expected).abs();

        return diff.max().item<float>();
    }

    private void CheckParity(YoloModelFamily family, YoloScale scale, string name)
    {
        string? weights = Find($"{name}.safetensors");
        string? parity = Find($"{name}-parity.safetensors");

        if (weights is null || parity is null)
        {
            Assert.Skip($"缺 {name}.safetensors 或 {name}-parity.safetensors；"
                + "先跑 tools/convert_yolo_weights.py --parity");
        }

        using var file = SafeTensorFile.Open(parity);

        SafeTensorEntry? inputEntry = file.Tensors.FirstOrDefault(entry => entry.Name == "input");
        if (inputEntry is null)
        {
            Assert.Skip("参考文件里缺 input 张量");
        }

        Device device = torch.CPU;
        using YoloDetectionModel model = YoloArchitectures.Create(device, family, scale, ClassCount);

        model.eval();

        YoloPretrainedReport report = YoloPretrained.Apply(model, weights, device);

        // 官方 80 类与本项目 80 类一致，所以应当全量覆盖（计数 buffer 不计）
        Assert.True(
            report.Untouched.Count == 0,
            $"{name} 未覆盖 {report.Untouched.Count} 个：{string.Join("、", report.Untouched)}");

        using Tensor input = TorchInterop.FromArray(
            file.ReadFloat32("input"), inputEntry.Shape, device);

        using var noGrad = torch.no_grad();
        IReadOnlyList<Tensor> trace = model.ForwardTrace(input);
        Tensor raw = trace[^1];

        int regressionMax = model.Head.RegressionMax;

        // 第一步：只比网络前向（one-to-many 分支）。本项目 raw 是 [B,4*reg_max+nc,N]，
        // 官方参考拆成 boxes / scores 两个张量
        CompareBranch(
            file, "boxes", "scores", raw, regressionMax, $"{name} one-to-many", device, RawTolerance);

        // 第二步：YOLO26 的 one-to-one 分支（推理用它去 NMS）
        if (model.Head.UsesEndToEnd)
        {
            IReadOnlyList<int> sources = model.InputSources[^1];
            var features = new Tensor[sources.Count];
            for (int index = 0; index < sources.Count; index++)
            {
                features[index] = trace[sources[index]];
            }

            using Tensor oneToOne = model.Head.ForwardOneToOne(features, detachFeatures: false);
            CompareBranch(
                file,
                "boxes1",
                "scores1",
                oneToOne,
                regressionMax,
                $"{name} one-to-one",
                device,
                OneToOneRawTolerance);
        }

        // 第三步：解码。参考里的 output 已由脚本统一转成 xyxy（官方默认输出 xywh）；
        // YOLO26 的 eval 输出是 top-k 后的 [B,N,6]，形状不同，不参与这一步
        SafeTensorEntry? outputEntry = file.Tensors.FirstOrDefault(entry => entry.Name == "output");
        if (outputEntry is not null)
        {
            using Tensor expected = TorchInterop.FromArray(
                file.ReadFloat32("output"), outputEntry.Shape, device);

            int height = (int)inputEntry.Shape[2];
            int width = (int)inputEntry.Shape[3];
            using Tensor actual = YoloPredictionDecoder.DecodeOutput0(
                raw, width, height, ClassCount, regressionMax);

            Assert.Equal(expected.shape, actual.shape);

            float maximum = MaxDiff(actual, expected);
            float boxDiff = MaxDiff(actual.narrow(1, 0, 4), expected.narrow(1, 0, 4));
            float classDiff = MaxDiff(
                actual.narrow(1, 4, ClassCount), expected.narrow(1, 4, ClassCount));

            Assert.True(
                maximum < Tolerance,
                $"{name} 与官方解码最大偏差 {maximum:G4}（框 {boxDiff:G4} / 分类 {classDiff:G4}）"
                + $"超过 {Tolerance:G4}（权重 {report.Summary}）");
        }
    }

    private void CompareBranch(
        SafeTensorFile file,
        string boxesName,
        string scoresName,
        Tensor raw,
        int regressionMax,
        string label,
        Device device,
        float tolerance)
    {
        SafeTensorEntry? boxesEntry = file.Tensors.FirstOrDefault(entry => entry.Name == boxesName);
        SafeTensorEntry? scoresEntry = file.Tensors.FirstOrDefault(entry => entry.Name == scoresName);

        if (boxesEntry is null || scoresEntry is null)
        {
            return;
        }

        using Tensor expectedBoxes = TorchInterop.FromArray(
            file.ReadFloat32(boxesName), boxesEntry.Shape, device);
        using Tensor expectedScores = TorchInterop.FromArray(
            file.ReadFloat32(scoresName), scoresEntry.Shape, device);

        float boxRawDiff = MaxDiff(raw.narrow(1, 0, 4 * regressionMax), expectedBoxes);
        float scoreRawDiff = MaxDiff(raw.narrow(1, 4 * regressionMax, ClassCount), expectedScores);

        Assert.True(
            boxRawDiff < tolerance && scoreRawDiff < tolerance,
            $"{label} 网络前向就不一致：box 分支 {boxRawDiff:G4} / 分类分支 {scoreRawDiff:G4}"
            + $"（容差 {tolerance:G4}）");
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void yolov8n_前向与官方逐元素一致() =>
        CheckParity(YoloModelFamily.V8, YoloScale.N, "yolov8n");

    [Fact]
    [Trait("Category", "Slow")]
    public void yolo11n_前向与官方逐元素一致() =>
        CheckParity(YoloModelFamily.V11, YoloScale.N, "yolo11n");

    [Fact]
    [Trait("Category", "Slow")]
    public void yolo26n_前向与官方逐元素一致() =>
        CheckParity(YoloModelFamily.V26, YoloScale.N, "yolo26n");
}
