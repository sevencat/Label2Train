using Label2Train.Train.Models.Weights;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 权重定位：候选目录回退、扩展名过滤、文件名片段优先。
/// 只测纯路径逻辑（<see cref="PretrainedWeights.ResolveIn"/>），不依赖真实权重文件。
/// </summary>
public sealed class PretrainedWeightsTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "label2train-weights-" + Guid.NewGuid().ToString("N"));

    public PretrainedWeightsTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root))
        {
            Directory.Delete(_root, recursive: true);
        }
    }

    private string Touch(string directory, string name)
    {
        string target = Path.Combine(_root, directory);

        Directory.CreateDirectory(target);

        string path = Path.Combine(target, name);

        File.WriteAllBytes(path, [0]);

        return path;
    }

    [Fact]
    public void 目录里没有任何权重时返回空()
    {
        string empty = Path.Combine(_root, "empty");

        Directory.CreateDirectory(empty);

        Assert.Null(PretrainedWeights.ResolveIn([empty], "yolov8"));
    }

    [Fact]
    public void 目录不存在时返回空()
    {
        Assert.Null(PretrainedWeights.ResolveIn([Path.Combine(_root, "missing")], "yolov8"));
    }

    [Fact]
    public void 命中文件名片段的优先()
    {
        Touch("w", "rf-detr-medium.safetensors");
        string wanted = Touch("w", "yolov8n.safetensors");
        Touch("w", "notes.txt");

        // 片段优先压过排序：'yolov8' 命中，'rf-detr-medium' 字母序更靠前也不选它
        Assert.Equal(wanted, PretrainedWeights.ResolveIn([Path.Combine(_root, "w")], "yolov8"));
    }

    [Fact]
    public void 只有非safetensors文件时不选()
    {
        Touch("w", "yolov8n.pt");
        Touch("w", "yolov8n.txt");

        Assert.Null(PretrainedWeights.ResolveIn([Path.Combine(_root, "w")], "yolov8"));
    }

    [Fact]
    public void 前一个目录为空时回退到后一个()
    {
        string empty = Path.Combine(_root, "a");
        string full = Path.Combine(_root, "b");

        Directory.CreateDirectory(empty);

        string wanted = Touch("b", "yolo11n.safetensors");

        Assert.Equal(wanted, PretrainedWeights.ResolveIn([empty, full], "yolo11"));
    }

    [Fact]
    public void 多个片段里命中任意一个都优先()
    {
        Touch("w", "aaa.safetensors");
        string wanted = Touch("w", "YOLO11S.safetensors");

        // 大小写不敏感
        Assert.Equal(wanted, PretrainedWeights.ResolveIn([Path.Combine(_root, "w")], "yolov8", "yolo11"));
    }

    [Fact]
    public void 提示里列出候选目录与环境变量名()
    {
        string hint = PretrainedWeights.Hint("LABEL2TRAIN_DEMO_WEIGHTS", "yolov8");

        Assert.Contains("LABEL2TRAIN_DEMO_WEIGHTS", hint, StringComparison.Ordinal);
        Assert.Contains("yolov8", hint, StringComparison.Ordinal);
        Assert.Contains("safetensors", hint, StringComparison.Ordinal);
    }
}
