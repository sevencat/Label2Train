using Label2Train.Core.Localization;
using Label2Train.Core.Model;
using OnnxSharp.Tensors;
using OnnxSharp.Vision;

namespace Label2Train.Core.AutoLabel;

/// <summary>
/// RF-DETR 推理：模型由本仓库的 <c>RfDetrOnnxExporter</c> 导出（<c>boxes</c> + <c>logits</c> 双输出），
/// letterbox、解码与坐标还原都复用 OnnxSharp 的 <see cref="RfDetrDetector"/> / <see cref="RfDetrDecoder"/>。
/// <para>
/// 与 <see cref="YoloOnnxRunner"/> 的关键差别在<b>预处理</b>：训练侧的 DETR 走的是 YOLO 那套
/// letterbox + <c>/255</c>（<c>DetrTrainingSession</c> 用的就是 <c>YoloSampleLoader</c>），
/// 不是 RF-DETR 官方的「拉伸 + ImageNet 归一化」。所以这里用 <see cref="ImagePreprocessor.Letterbox"/>
/// 造输入，解码按 letterbox 尺寸还原再减填边——直接用 <see cref="RfDetrDetector.Detect(string)"/> 的
/// 默认预处理会让框整体偏移、置信度崩掉（训练 / 推理预处理必须一致）。
/// </para>
/// </summary>
public sealed class RfDetrOnnxRunner : IAutoLabelRunner
{
    /// <summary>DETR 的一个查询可以同时在多个类别上过阈值，与官方 helper 一致取前 300 个。</summary>
    private const int MaxDetections = 300;

    private readonly RfDetrDetector _detector;

    public RfDetrOnnxRunner(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(Text.Get("Error.ModelFileNotFound"), modelPath);
        }

        _detector = new RfDetrDetector(modelPath, CreateOptions(0.25f));
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

        // 与 YOLO 一样：OnnxSharp 的预处理只吃紧凑的 3 字节像素
        byte[] bgr = YoloOnnxRunner.ToPackedBgr(bgra, width, height);

        (Tensor input, Letterbox transform) = ImagePreprocessor.Letterbox(
            bgr,
            width,
            height,
            _detector.TargetWidth,
            _detector.TargetHeight,
            bgr: true,
            rowStride: width * 3);

        _detector.Net.SetInput(_detector.InputName, input);
        List<Tensor> outputs = _detector.Net.ForwardAll();

        Tensor boxes = FindOutput(outputs, _detector.BoxesOutputName);
        Tensor logits = FindOutput(outputs, _detector.LogitsOutputName);

        // 输出是相对「整幅 letterbox 图」的归一化 cxcywh：先按 letterbox 尺寸解成像素，
        // 再减填边、除比例回到原图坐标系
        IReadOnlyList<Detection> detections = RfDetrDecoder.Decode(
            boxes,
            logits,
            transform.TargetWidth,
            transform.TargetHeight,
            CreateOptions((float)options.Confidence));

        if (detections.Count == 0)
        {
            return [];
        }

        var candidates = new List<AutoLabelCandidate>(detections.Count);

        foreach (Detection detection in detections)
        {
            float left = Math.Clamp(transform.ToSourceX(detection.X), 0, width);
            float top = Math.Clamp(transform.ToSourceY(detection.Y), 0, height);
            float right = Math.Clamp(transform.ToSourceX(detection.Right), 0, width);
            float bottom = Math.Clamp(transform.ToSourceY(detection.Bottom), 0, height);

            if (right - left < 1 || bottom - top < 1)
            {
                continue;
            }

            candidates.Add(new AutoLabelCandidate(
                RectD.FromLtrb(left, top, right, bottom),
                YoloOnnxRunner.ResolveLabel(detection.ClassId, options),
                detection.ClassId,
                detection.Score));
        }

        return candidates;
    }

    /// <summary>按输出名在 <see cref="OnnxSharp.Core.Net.ForwardAll"/> 的结果里取张量（顺序同 OutputNames）。</summary>
    private Tensor FindOutput(List<Tensor> outputs, string name)
    {
        IReadOnlyList<string> names = _detector.Net.OutputNames;

        for (int index = 0; index < names.Count && index < outputs.Count; index++)
        {
            if (string.Equals(names[index], name, StringComparison.Ordinal))
            {
                return outputs[index];
            }
        }

        throw new InvalidOperationException($"模型输出里没有「{name}」：{string.Join(", ", names)}");
    }

    private static RfDetrOptions CreateOptions(float confidence) => new()
    {
        Confidence = confidence,
        MaxDetections = MaxDetections,
        // 我们的导出里 class_embed 直接输出 nc 个类，没有背景列（COCO 91 类那种才有）
        BackgroundClassId = null,
        Normalization = ChannelNormalization.Divide255,
    };

    /// <summary>OnnxSharp 的会话全托管，没有需要释放的原生句柄。</summary>
    public void Dispose()
    {
    }
}
