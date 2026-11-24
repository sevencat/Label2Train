using Label2Train.Train.Core;
using Label2Train.Train.IO.SafeTensors;
using Label2Train.Train.Models.Yolo;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Label2Train.App.Tests;

/// <summary>
/// YOLO 权重装载：名字规范化、形状校验、部分覆盖与未覆盖清单。
/// 用 v8n(nc=2) 这个最小结构构造假权重，不依赖真实官方文件。
/// </summary>
public sealed class YoloPretrainedTests : IDisposable
{
    private const int ClassCount = 2;

    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "label2train-yolo-weights-" + Guid.NewGuid().ToString("N"));

    private readonly Device _device = torch.CPU;

    private readonly YoloV8DetectionModel _model;

    public YoloPretrainedTests()
    {
        Directory.CreateDirectory(_directory);
        _model = YoloV8.Create(_device, YoloScale.N, ClassCount);
    }

    public void Dispose()
    {
        _model.Dispose();

        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    private string Path_(string name) => Path.Combine(_directory, name);

    private static (long[] Shape, float[] Values) Ones(long[] shape)
    {
        long count = 1;
        foreach (long dim in shape)
        {
            count *= dim;
        }

        var values = new float[count];
        Array.Fill(values, 1f);

        return (shape, values);
    }

    [Theory]
    [InlineData("model.0.conv.weight", "model.0.conv.weight")]
    [InlineData("module.model.0.conv.weight", "model.0.conv.weight")]
    [InlineData("model.model.0.conv.weight", "model.0.conv.weight")]
    [InlineData("module.model.model.22.cv2.0.0.conv.weight", "model.22.cv2.0.0.conv.weight")]
    public void 名字规范化去掉冗余前缀(string source, string expected) =>
        Assert.Equal(expected, YoloPretrained.Normalize(source));

    [Fact]
    public void 一个张量都对不上时抛异常()
    {
        var writer = new SafeTensorWriter();

        (long[] shape, float[] values) = Ones([2, 2]);
        writer.AddFloat32("something.else.weight", shape, values);

        string path = Path_("bad.safetensors");
        writer.Save(path);

        Assert.Throws<InvalidDataException>(() => YoloPretrained.Apply(_model, path, _device));
    }

    [Fact]
    public void 形状一致才载入_形状不符进未覆盖清单()
    {
        Dictionary<string, Tensor> state = _model.state_dict();

        (long[] convShape, float[] convValues) = Ones(state["model.0.conv.weight"].shape);
        (long[] biasShape, float[] biasValues) = Ones(state["model.0.bn.bias"].shape);

        var writer = new SafeTensorWriter();
        writer.AddFloat32("model.0.conv.weight", convShape, convValues);

        // 故意把 bn.bias 写错形状：应当被跳过，并留在未覆盖清单里
        var wrong = new float[biasValues.Length + 1];
        writer.AddFloat32("model.0.bn.bias", new long[] { wrong.Length }, wrong);

        string path = Path_("partial.safetensors");
        writer.Save(path);

        YoloPretrainedReport report = YoloPretrained.Apply(_model, path, _device);

        Assert.Equal(1, report.Loaded);
        Assert.Contains("model.0.bn.bias", report.Untouched);
        Assert.DoesNotContain("model.0.conv.weight", report.Untouched);
    }

    [Fact]
    public void 载入后参数值真的变了()
    {
        Dictionary<string, Tensor> state = _model.state_dict();
        long[] shape = state["model.0.conv.weight"].shape;

        var values = new float[shape[0] * shape[1] * shape[2] * shape[3]];
        Array.Fill(values, 0.125f);

        var writer = new SafeTensorWriter();
        writer.AddFloat32("model.0.conv.weight", shape, values);

        // 为了通过「一个都对不上就抛」的检查，再带一个合法的 bn.weight
        (long[] bnShape, float[] bnValues) = Ones(state["model.0.bn.weight"].shape);
        writer.AddFloat32("model.0.bn.weight", bnShape, bnValues);

        string path = Path_("values.safetensors");
        writer.Save(path);

        YoloPretrained.Apply(_model, path, _device);

        float[] actual = TorchInterop.ToArray(_model.state_dict()["model.0.conv.weight"]);
        Assert.All(actual, value => Assert.Equal(0.125f, value, 6));
    }

    [Fact]
    public void 权重文件不存在时抛_文件未找到()
    {
        string missing = Path_("nope.safetensors");

        Assert.Throws<FileNotFoundException>(() => YoloPretrained.Apply(_model, missing, _device));
    }

    [Fact]
    public void 权重来源按家族与规模拼片段()
    {
        // 只验证提示文案里的片段，不做真实目录查找
        string hint = YoloWeightSource.Hint(YoloModelFamily.V11, YoloScale.S);

        Assert.Contains("yolo11s", hint, StringComparison.Ordinal);
        Assert.Contains(YoloWeightSource.Variable, hint, StringComparison.Ordinal);
    }
}
