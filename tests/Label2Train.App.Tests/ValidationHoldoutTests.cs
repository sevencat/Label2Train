using Label2Train.Train.Data.Yaml;
using Label2Train.Train.Runners;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 没有 val split 时自动留出验证集：mAP 只能在验证集上算，一张都不留的话曲线上的
/// mAP50 / mAP50-95 会永远是 0（库里的图片通常只有 train）。
/// </summary>
public sealed class ValidationHoldoutTests : IDisposable
{
    private readonly string _root =
        Path.Combine(Path.GetTempPath(), "sevencat-holdout-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root))
        {
            Directory.Delete(_root, recursive: true);
        }
    }

    /// <summary>每 5 张搬 1 张到 val，标签跟着图片成对搬；剩下的仍留在 train。</summary>
    [Fact]
    public void SplitExportedYoloSplit_MovesEveryFifthImageWithItsLabel()
    {
        string images = Path.Combine(_root, "images", "train");
        string labels = Path.Combine(_root, "labels", "train");
        Directory.CreateDirectory(images);
        Directory.CreateDirectory(labels);

        for (int index = 0; index < 10; index++)
        {
            File.WriteAllBytes(Path.Combine(images, $"img{index}.jpg"), [1]);
            File.WriteAllText(Path.Combine(labels, $"img{index}.txt"), "0 0.5 0.5 0.2 0.2");
        }

        int moved = ValidationHoldout.SplitExportedYoloSplit(_root, "train");

        Assert.Equal(2, moved);
        Assert.False(File.Exists(Path.Combine(images, "img4.jpg")));
        Assert.False(File.Exists(Path.Combine(labels, "img4.txt")));
        Assert.True(File.Exists(Path.Combine(_root, "images", "val", "img4.jpg")));
        Assert.True(File.Exists(Path.Combine(_root, "labels", "val", "img4.txt")));
        Assert.Equal(8, Directory.GetFiles(images).Length);
        Assert.Equal(8, Directory.GetFiles(labels).Length);
    }

    /// <summary>图片少于 5 张时不划：训练集不能为空，这种小样本也不值得验证。</summary>
    [Fact]
    public void SplitExportedYoloSplit_LeavesTinyDatasetsAlone()
    {
        string images = Path.Combine(_root, "images", "train");
        Directory.CreateDirectory(images);

        for (int index = 0; index < 4; index++)
        {
            File.WriteAllBytes(Path.Combine(images, $"img{index}.jpg"), [1]);
        }

        Assert.Equal(0, ValidationHoldout.SplitExportedYoloSplit(_root, "train"));
        Assert.False(Directory.Exists(Path.Combine(_root, "images", "val")));
    }

    /// <summary>data.yaml 只补一次 val：重复调用不能再追加第二行。</summary>
    [Fact]
    public void AddValidationSplit_AppendsOnce()
    {
        Directory.CreateDirectory(_root);

        string yaml = Path.Combine(_root, "data.yaml");
        File.WriteAllText(yaml, "path: .\ntrain: images/train\nnames:\n  0: \"a\"\n");

        Assert.True(ValidationHoldout.AddValidationSplit(yaml));
        Assert.Contains("val: images/val", File.ReadAllText(yaml));
        Assert.False(ValidationHoldout.AddValidationSplit(yaml));

        // 追加的位置（names 列表之后）必须还能被 data.yaml 解析器读出来
        IReadOnlyDictionary<string, object?> parsed = MiniYaml.ParseFile(yaml);

        Assert.Equal("images/val", MiniYaml.AsString(parsed["val"], "val"));
        Assert.Equal("images/train", MiniYaml.AsString(parsed["train"], "train"));
    }
}
