using Label2Train.Train.Core;
using Label2Train.Train.IO.SafeTensors;
using Label2Train.Train.Models.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Models.Yolo;

/// <summary>
/// 把官方 ultralytics YOLO 权重（<c>yolov8n.pt</c> / <c>yolo11n.pt</c> 转成的 <c>.safetensors</c>）
/// 装载进本项目模型。
/// <para>
/// 本项目的模块命名刻意与 ultralytics 对齐（<c>model.2.cv1.conv.weight</c>、
/// <c>model.22.cv2.0.0.conv.weight</c>），所以这里<b>按名字直取</b>，不需要逐层映射表。
/// 只做形状校验：形状不一致就跳过并记账——最典型的是分类分支
/// （官方 80 类，用户数据集 3 类），那几层应当保持随机初始化。
/// </para>
/// <para>
/// 未覆盖的键会完整报出来（<see cref="YoloPretrainedReport.Untouched"/>），所以命名一旦有偏差
/// 第一次运行就能看见，而不是悄悄少载一半。
/// </para>
/// </summary>
public static class YoloPretrained
{
    /// <summary>载入权重，返回覆盖情况。</summary>
    public static YoloPretrainedReport Apply(YoloDetectionModel model, string path, Device device)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"找不到权重文件 '{path}'。", path);
        }

        Dictionary<string, Tensor> state = model.state_dict();
        var loaded = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        int skipped = 0;

        using SafeTensorFile file = SafeTensorFile.Open(path);

        foreach (SafeTensorEntry entry in file.Tensors)
        {
            string key = Normalize(entry.Name);

            if (!state.TryGetValue(key, out Tensor? current))
            {
                // 检查点里有、本项目结构没有的（官方 Detect 的 stride buffer 等不算参数）
                skipped++;

                continue;
            }

            long[] want = current.shape;

            if (!want.SequenceEqual(entry.Shape))
            {
                // 类别数不同 → cv3.* 形状不一致，保持随机初始化
                skipped++;

                continue;
            }

            loaded[key] = TorchInterop
                .FromArray(file.ReadFloat32(entry.Name), want, device);
        }

        if (loaded.Count == 0)
        {
            throw new InvalidDataException($"'{path}' 里没有一个张量能对上 {model.Architecture} 的结构。");
        }

        model.load_state_dict(loaded, strict: false);

        // num_batches_tracked 这类计数 buffer 不参与前向，也不算「未覆盖」；
        // 只报仍然随机初始化的权重（最典型是类别数不同时的 cv3.*）
        string[] untouched =
        [
            .. state.Keys
                .Where(key => !loaded.ContainsKey(key)
                    && !key.EndsWith(".num_batches_tracked", StringComparison.Ordinal))
                .OrderBy(key => key, StringComparer.Ordinal),
        ];

        return new YoloPretrainedReport(loaded.Count, skipped, untouched);
    }

    /// <summary>
    /// 去掉导出工具可能加的冗余前缀，得到与本项目 <c>state_dict</c> 一致的名字。
    /// <list type="bullet">
    /// <item><c>module.model.0.conv.weight</c> → <c>model.0.conv.weight</c>（DataParallel 包装）；</item>
    /// <item><c>model.model.0.conv.weight</c> → <c>model.0.conv.weight</c>（ultralytics 外层 nn.Module 也叫 model）。</item>
    /// </list>
    /// </summary>
    internal static string Normalize(string name)
    {
        string key = name;

        while (true)
        {
            if (key.StartsWith("module.", StringComparison.Ordinal))
            {
                key = key["module.".Length..];

                continue;
            }

            if (key.StartsWith("model.model.", StringComparison.Ordinal))
            {
                key = key["model.".Length..];

                continue;
            }

            return key;
        }
    }
}

/// <summary>权重载入结果：载入多少、跳过多少、哪些参数仍然是随机初始化的。</summary>
public sealed record YoloPretrainedReport(int Loaded, int Skipped, IReadOnlyList<string> Untouched)
{
    /// <summary>给训练日志用的一句话摘要。</summary>
    public string Summary => Untouched.Count == 0
        ? $"载入 {Loaded} 个张量（检查点里另有 {Skipped} 个用不上）"
        : $"载入 {Loaded} 个张量（检查点里另有 {Skipped} 个用不上；"
            + $"{Untouched.Count} 个参数保持随机初始化，如 "
            + $"{string.Join("、", Untouched.Take(3))}{(Untouched.Count > 3 ? " 等" : string.Empty)}）";
}

/// <summary>
/// 去哪儿找 YOLO 的预训练权重：环境变量 <c>LABEL2TRAIN_YOLO_WEIGHTS</c> 优先，
/// 否则在各候选目录里按 <c>yolov8n</c> / <c>yolo11n</c> 这类文件名片段挑。
/// </summary>
public static class YoloWeightSource
{
    /// <summary>环境变量名。</summary>
    public const string Variable = "LABEL2TRAIN_YOLO_WEIGHTS";

    /// <summary>找一份与家族 / 规模匹配的权重；没有返回 <c>null</c>。</summary>
    public static string? Resolve(YoloModelFamily family, YoloScale scale) =>
        PretrainedWeights.Resolve(Variable, Fragments(family, scale));

    /// <summary>没找到时给用户看的提示（只在训练日志里打一次）。</summary>
    public static string Hint(YoloModelFamily family, YoloScale scale) =>
        PretrainedWeights.Hint(Variable, Fragments(family, scale));

    /// <summary>文件名片段：<c>yolov8n</c> 与 <c>yolov8-n</c> 两种写法都认。</summary>
    private static string[] Fragments(YoloModelFamily family, YoloScale scale)
    {
        string prefix = YoloScales.Prefix(family);
        char letter = YoloScales.ToLetter(scale);

        return [$"{prefix}{letter}", $"{prefix}-{letter}"];
    }
}
