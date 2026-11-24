using System.Globalization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Models.Yolo;

/// <summary>
/// 按 <c>architecture</c> 字符串创建 / 解析 YOLO 模型的统一入口：
/// <c>yolov8n-detect-3</c> 与 <c>yolo11n-detect-3</c> 共用一套解析规则，调用方不必关心家族。
/// </summary>
public static class YoloArchitectures
{
    /// <summary>任务段（目前只有 detect）。</summary>
    public const string DetectTask = "detect";

    /// <summary>已注册的家族前缀（顺序即匹配顺序，长的放前面避免前缀互相吃掉）。</summary>
    private static readonly (string Prefix, YoloModelFamily Family)[] FamilyPrefixes =
    [
        (YoloScales.V8Prefix, YoloModelFamily.V8),
        (YoloScales.V11Prefix, YoloModelFamily.V11),
        (YoloScales.V26Prefix, YoloModelFamily.V26),
    ];

    /// <summary>当前支持的全部家族（界面下拉 / 文档用）。</summary>
    public static IReadOnlyList<YoloModelFamily> SupportedFamilies { get; } =
        [.. FamilyPrefixes.Select(item => item.Family)];

    /// <summary>生成 architecture 字符串。</summary>
    public static string Format(YoloModelFamily family, YoloScale scale, int classCount)
    {
        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "类别数必须大于 0。");

        return YoloScales.Describe(family, scale, classCount);
    }

    /// <summary>
    /// 解析 <c>&lt;prefix&gt;&lt;scale&gt;-detect-&lt;nc&gt;</c>；
    /// 也接受 <c>yolov8n</c> / <c>yolo11s</c> 这种省略类别数的写法（默认 80 类，与 ultralytics 一致）。
    /// </summary>
    public static (YoloModelFamily Family, YoloScale Scale, int ClassCount) Parse(string architecture)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(architecture);

        string text = architecture.Trim();
        YoloModelFamily family = DetectFamily(text, out int prefixLength);

        string rest = text[prefixLength..];
        int separator = rest.IndexOf('-');

        if (separator < 0)
            return (family, YoloScales.Parse(rest), 80);

        string scaleText = rest[..separator];
        string[] parts = rest[separator..].Split('-', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 2 || !string.Equals(parts[0], DetectTask, StringComparison.OrdinalIgnoreCase))
        {
            throw new FormatException(
                $"architecture '{architecture}' 的格式应为 '<family><scale>-detect-<nc>'。");
        }

        if (!int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int classCount)
            || classCount <= 0)
        {
            throw new FormatException($"architecture '{architecture}' 的类别数 '{parts[1]}' 不是正整数。");
        }

        return (family, YoloScales.Parse(scaleText), classCount);
    }

    /// <summary>判断 architecture 是否属于 YOLO 家族（供预测器分发用）。</summary>
    public static bool TryParseFamily(string architecture, out YoloModelFamily family)
    {
        family = default;

        if (string.IsNullOrWhiteSpace(architecture))
            return false;

        return TryMatchPrefix(architecture.Trim(), out family, out _);
    }

    /// <summary>按家族分派到对应工厂（v8 / v11 / v26）。</summary>
    public static YoloDetectionModel Create(Device device, YoloModelFamily family, YoloScale scale, int classCount) =>
        family switch
        {
            YoloModelFamily.V8 => YoloV8.Create(device, scale, classCount),
            YoloModelFamily.V11 => YoloV11.Create(device, scale, classCount),
            YoloModelFamily.V26 => YoloV26.Create(device, scale, classCount),
            _ => throw new ArgumentOutOfRangeException(nameof(family), family, "未知的 YOLO 家族。"),
        };

    /// <summary>按 architecture 字符串创建模型。</summary>
    public static YoloDetectionModel CreateFromArchitecture(Device device, string architecture)
    {
        (YoloModelFamily family, YoloScale scale, int classCount) = Parse(architecture);
        return Create(device, family, scale, classCount);
    }

    /// <summary>参数总量（含 BN 的 weight / bias；不含 running_mean 等 buffer）。</summary>
    public static long CountParameters(Module module)
    {
        ArgumentNullException.ThrowIfNull(module);
        return module.parameters().Sum(parameter => parameter.numel());
    }

    /// <summary>逐层参数量摘要（CLI 与日志用）。</summary>
    public static IReadOnlyList<(int Layer, string Type, long Parameters)> DescribeLayers(YoloDetectionModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        var rows = new List<(int Layer, string Type, long Parameters)>(model.LayerCount);
        for (int index = 0; index < model.LayerCount; index++)
        {
            Module layer = model.Layer(index);
            rows.Add((index, layer.GetType().Name, layer.parameters().Sum(parameter => parameter.numel())));
        }

        return rows;
    }

    private static bool TryMatchPrefix(
        string text, out YoloModelFamily family, out int prefixLength)
    {
        foreach ((string prefix, YoloModelFamily candidate) in FamilyPrefixes)
        {
            if (text.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                family = candidate;
                prefixLength = prefix.Length;
                return true;
            }
        }

        family = default;
        prefixLength = 0;
        return false;
    }

    private static YoloModelFamily DetectFamily(string architecture, out int prefixLength)
    {
        if (TryMatchPrefix(architecture, out YoloModelFamily family, out prefixLength))
            return family;

        string supported = string.Join(" / ", FamilyPrefixes.Select(item => $"{item.Prefix}*"));
        throw new NotSupportedException(
            $"architecture '{architecture}' 不是已知的 YOLO 家族（支持 {supported}）。");
    }
}
