using Label2Train.Core.Dataset;

namespace Label2Train.Train.Runners;

/// <summary>
/// 没有 <c>val</c> split 时，按固定比例从训练集里留出一份验证集。
/// <para>
/// 标注库里导入的图片通常没有 split（都算训练集），而 mAP 只能在验证集上算——
/// 不留一份出来，曲线上的 mAP50 / mAP50-95 会永远是 0，看起来像「训练没生效」。
/// </para>
/// <para>
/// 取样规则是「按文件路径排序后每 5 张取 1 张」（20%），同一份数据每次划出的验证集完全一致，
/// 换一轮训练后 mAP 才可比。图片少于 5 张时不划：训练集不能为空，这种小样本也不值得验证。
/// </para>
/// </summary>
internal static class ValidationHoldout
{
    /// <summary>留出比例的分母：每 5 张取 1 张（20%）。</summary>
    internal const int Every = 5;

    /// <summary>内存里划：依次取第 5、10、15… 张当验证集，其余留给训练。</summary>
    public static (IReadOnlyList<DatasetSample> Train, IReadOnlyList<DatasetSample> Validation)
        SplitInMemory(IReadOnlyList<DatasetSample> samples)
    {
        if (samples.Count < Every)
        {
            return (samples, []);
        }

        var train = new List<DatasetSample>(samples.Count);
        var validation = new List<DatasetSample>(samples.Count / Every);

        for (int index = 0; index < samples.Count; index++)
        {
            if ((index + 1) % Every == 0)
            {
                validation.Add(samples[index]);
            }
            else
            {
                train.Add(samples[index]);
            }
        }

        return (train, validation);
    }

    /// <summary>
    /// 把已经导出的 YOLO 目录按同样的规则搬到 <c>val</c>：图片与标签成对移动，
    /// 搬完在 <c>data.yaml</c> 末尾补一行 <c>val: images/val</c>（顶层键，缩进 0）。
    /// </summary>
    /// <returns>搬走的图片张数；0 表示没搬（图片太少或目录不存在）。</returns>
    public static int SplitExportedYoloSplit(string root, string split)
    {
        string imageDirectory = Path.Combine(root, "images", split);
        string labelDirectory = Path.Combine(root, "labels", split);

        if (!Directory.Exists(imageDirectory))
        {
            return 0;
        }

        List<string> images =
        [
            .. Directory.EnumerateFiles(imageDirectory).OrderBy(path => path, StringComparer.Ordinal),
        ];

        if (images.Count < Every)
        {
            return 0;
        }

        string valImages = Path.Combine(root, "images", "val");
        string valLabels = Path.Combine(root, "labels", "val");
        Directory.CreateDirectory(valImages);
        Directory.CreateDirectory(valLabels);

        int moved = 0;

        // 先按序号挑好再搬：在枚举结果上边搬边删会漏掉后面的文件
        foreach (string image in images.Skip(Every - 1).Where((_, offset) => offset % Every == 0))
        {
            string fileName = Path.GetFileName(image);
            string stem = Path.GetFileNameWithoutExtension(image);
            string label = Path.Combine(labelDirectory, stem + ".txt");

            File.Move(image, Path.Combine(valImages, fileName));

            if (File.Exists(label))
            {
                File.Move(label, Path.Combine(valLabels, stem + ".txt"));
            }

            moved++;
        }

        return moved;
    }

    /// <summary>
    /// 在 <c>data.yaml</c> 末尾补一行 <c>val: images/val</c>（顶层键，缩进 0）。
    /// 导出器只写它认识的 split，所以自动留出的验证集得在这里补。
    /// </summary>
    /// <returns>真的写了才返回 <c>true</c>；已经写过或文件不存在返回 <c>false</c>。</returns>
    public static bool AddValidationSplit(string dataYamlPath)
    {
        if (!File.Exists(dataYamlPath))
        {
            return false;
        }

        string text = File.ReadAllText(dataYamlPath);

        // 顶层 val 键已存在就不再追加，免得出现两个 val（YAML 后者覆盖前者，也能用，但容易误导）
        foreach (string line in text.Split('\n'))
        {
            string trimmed = line.TrimEnd('\r');

            if (trimmed.Length > 0 && trimmed[0] != ' '
                && trimmed.StartsWith("val:", StringComparison.Ordinal))
            {
                return false;
            }
        }

        File.AppendAllText(
            dataYamlPath,
            $"{Environment.NewLine}val: images/val{Environment.NewLine}");

        return true;
    }
}
