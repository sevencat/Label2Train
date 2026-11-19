using Label2Train.Core.Localization;
using Label2Train.Core.Model;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Vision;

namespace Label2Train.Core.AutoLabel;

/// <summary>AI 标注推理器的抽象：输入 BGRA 像素，输出原图坐标的候选框。</summary>
public interface IAutoLabelRunner : IDisposable
{
    IReadOnlyList<AutoLabelCandidate> Predict(
        ReadOnlySpan<byte> bgra,
        int width,
        int height,
        AutoLabelOptions options);
}

/// <summary>
/// YOLO（Ultralytics 导出的 <c>.onnx</c>）推理：交给 OnnxSharp 的 <see cref="YoloDetector"/>，
/// letterbox、归一化、输出解码与 NMS 都在它内部完成，我们只做坐标裁剪与类别名映射。
/// 支持 v8/v11 的「4 + 类别数」输出以及端到端（EndToEnd）导出。
/// </summary>
public sealed class YoloOnnxRunner : IAutoLabelRunner
{
    private readonly YoloDetector _detector;

    public YoloOnnxRunner(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(Text.Get("Error.ModelFileNotFound"), modelPath);
        }

        // 阈值每次 Predict 会按调用方参数改写，这里的初值只是兜底
        _detector = new YoloDetector(modelPath, new DetectorOptions
        {
            Confidence = 0.25f,
            IouThreshold = 0.45f,
        });
    }

    public IReadOnlyList<AutoLabelCandidate> Predict(
        ReadOnlySpan<byte> bgra,
        int width,
        int height,
        AutoLabelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (width <= 0 || height <= 0 || bgra.Length < width * height * 4)
        {
            return [];
        }

        // 阈值每次调用都可能变，DetectorOptions 的属性是 init-only，复制一份再改阈值
        DetectorOptions defaults = _detector.Options;
        var settings = new DetectorOptions
        {
            Confidence = (float)options.Confidence,
            IouThreshold = (float)options.IouThreshold,
            MaxDetections = defaults.MaxDetections,
            MaxCandidates = defaults.MaxCandidates,
            ClassAgnosticNms = defaults.ClassAgnosticNms,
            ClassCount = defaults.ClassCount,
            Layout = defaults.Layout,
            TargetWidth = defaults.TargetWidth,
            TargetHeight = defaults.TargetHeight,
        };

        // OnnxSharp 的 Preprocess 只接受「紧凑 3 字节」像素（BGR/RGB），
        // 传 BGRA 会让它按 3 字节步进读取 4 字节数据：每行错位累积，图像被撕成竖条纹，
        // 模型实际看到的是一张噪声图（检测框位置、类别、置信度全错）。
        byte[] bgr = ToPackedBgr(bgra, width, height);

        (Tensor input, Letterbox transform) = _detector.Preprocess(
            bgr,
            width,
            height,
            bgr: true,
            rowStride: width * 3);

        _detector.Net.SetInput(_detector.InputName, input);
        Tensor output = _detector.Net.Forward();

        IReadOnlyList<Detection> detections = YoloDecoder.Decode(
            output,
            in transform,
            settings,
            _detector.ClassNames);

        if (detections.Count == 0)
        {
            return [];
        }

        var candidates = new List<AutoLabelCandidate>(detections.Count);

        foreach (Detection detection in detections)
        {
            double left = Math.Clamp(detection.X, 0, width);
            double top = Math.Clamp(detection.Y, 0, height);
            double right = Math.Clamp(detection.Right, 0, width);
            double bottom = Math.Clamp(detection.Bottom, 0, height);

            if (right - left < 1 || bottom - top < 1)
            {
                continue;
            }

            candidates.Add(new AutoLabelCandidate(
                RectD.FromLtrb(left, top, right, bottom),
                ResolveLabel(detection.ClassId, options),
                detection.ClassId,
                detection.Score));
        }

        return candidates;
    }

    /// <summary>把 BGRA 拆成紧凑的 BGR（去掉 alpha，每像素 3 字节）。</summary>
    internal static byte[] ToPackedBgr(ReadOnlySpan<byte> bgra, int width, int height)
    {
        int pixels = width * height;
        var bgr = new byte[pixels * 3];

        for (int i = 0, src = 0, dst = 0; i < pixels; i++, src += 4, dst += 3)
        {
            bgr[dst] = bgra[src];
            bgr[dst + 1] = bgra[src + 1];
            bgr[dst + 2] = bgra[src + 2];
        }

        return bgr;
    }

    /// <summary>项目类别映射优先，其次模型自带类别名，最后退回 <c>class_{id}</c>。</summary>
    internal static string ResolveLabel(int classId, AutoLabelOptions options)
    {
        if (options.ClassMap is { } map
            && map.TryGetValue(classId, out string? mapped)
            && !string.IsNullOrWhiteSpace(mapped))
        {
            return mapped.Trim();
        }

        if (options.ClassNames is { } names
            && classId >= 0
            && classId < names.Count
            && !string.IsNullOrWhiteSpace(names[classId]))
        {
            return names[classId].Trim();
        }

        return $"class_{classId}";
    }

    /// <summary>OnnxSharp 的会话全托管，没有需要释放的原生句柄。</summary>
    public void Dispose()
    {
    }
}
