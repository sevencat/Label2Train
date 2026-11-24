using System.Globalization;
using System.Text.Json;
using Label2Train.Train.Models.RfDetr;
using Label2Train.Train.Models.SimpleDetr;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Export;

/// <summary>导出结果：目标目录里落了哪些文件。</summary>
/// <param name="Directory">导出目录。</param>
/// <param name="SafetensorsPath">复制过去的权重（<c>{模型名}.safetensors</c>）。</param>
/// <param name="ConfigPath">配套的配置（<c>{模型名}.json</c>）。</param>
/// <param name="OnnxPath">导出的 ONNX（<c>{模型名}.onnx</c>）；没要求导出 ONNX 时为 <c>null</c>。</param>
public sealed record TrainingExportResult(
    string Directory,
    string SafetensorsPath,
    string ConfigPath,
    string? OnnxPath);

/// <summary>
/// 把一次训练的产物导成能直接用的模型文件：
/// <c>{模型名}.safetensors</c> + <c>{模型名}.json</c>（架构 / 类别 / 输入尺寸），
/// 以及可交付的 <c>{模型名}.onnx</c>。
/// <para>
/// 文件名按「模型名_规模」取：YOLO 是 <c>yolov8_s</c> / <c>yolov11_m</c>，DETR 是 <c>rfdetr</c> /
/// <c>simple-detr</c>。同一个项目的 <c>models</c> 目录里可以同时放下各家族、各规模的模型，
/// 检测时按前缀挑；不带规模的话，换规模重训会把上一份覆盖掉。
/// </para>
/// <para>
/// 支持三种架构，按检查点元数据里的 <c>architecture</c> 分派：
/// <c>yolov8*</c> / <c>yolo11*</c>（YOLO）、<c>rfdetr-detect-N</c>（RF-DETR）、
/// <c>simple-detr-detect-N</c>（简化 DETR）。
/// 架构与类别都从检查点元数据、<c>train.json</c> 里取，不靠界面猜。
/// </para>
/// </summary>
public sealed class TrainingExportService
{
    /// <summary>训练输出目录里的最优权重文件名（导出时会按模型名改名）。</summary>
    public const string WeightsFileName = "best.safetensors";

    /// <summary>旧版导出用的配置文件名；现在只在读取老目录时用。</summary>
    public const string ConfigFileName = "model.json";

    /// <summary>旧版导出用的 ONNX 文件名；检测找不到新名字时回退到它。</summary>
    public const string OnnxFileName = "best.onnx";

    private const string RfDetrPrefix = "rfdetr";
    private const string SimpleDetrPrefix = "simple-detr";
    private const string V8Prefix = "yolov8";
    private const string V11Prefix = "yolo11";
    private const string V26Prefix = "yolo26";

    /// <summary>
    /// 架构 → 导出文件名前缀：<c>yolov8*</c>→<c>yolov8</c>、<c>yolo11*</c>→<c>yolov11</c>、
    /// <c>rfdetr*</c>→<c>rfdetr</c>、<c>simple-detr*</c>→<c>simple-detr</c>。
    /// </summary>
    public static string PrefixFor(string architecture)
    {
        string text = (architecture ?? string.Empty).Trim();

        if (text.StartsWith(V26Prefix, StringComparison.OrdinalIgnoreCase))
        {
            return "yolo26";
        }

        if (text.StartsWith(V11Prefix, StringComparison.OrdinalIgnoreCase))
        {
            return "yolov11";
        }

        if (text.StartsWith(V8Prefix, StringComparison.OrdinalIgnoreCase))
        {
            return V8Prefix;
        }

        if (text.StartsWith(RfDetrPrefix, StringComparison.OrdinalIgnoreCase))
        {
            return RfDetrPrefix;
        }

        if (text.StartsWith(SimpleDetrPrefix, StringComparison.OrdinalIgnoreCase))
        {
            return SimpleDetrPrefix;
        }

        return "model";
    }

    /// <summary>
    /// 架构 → 默认产物文件名（不含扩展名）：YOLO 带规模（<c>yolov8_s</c> / <c>yolov11_m</c>），
    /// DETR 的体量由 queries / 层数决定，没有规模字母（<c>rfdetr</c> / <c>simple-detr</c>）。
    /// </summary>
    public static string DefaultPrefixFor(string architecture)
    {
        if (!YoloArchitectures.TryParseFamily(architecture, out _))
        {
            return PrefixFor(architecture);
        }

        (_, YoloScale scale, _) = YoloArchitectures.Parse(architecture);
        return $"{PrefixFor(architecture)}_{YoloScales.ToLetter(scale)}";
    }

    /// <summary>
    /// 导出。<paramref name="runDirectory"/> 是训练输出目录（里面有 best.safetensors / train.json），
    /// <paramref name="targetDirectory"/> 是导出目录（例如 <c>{输出目录}/export</c>）。
    /// <paramref name="filePrefix"/> 为空时按架构自动取名（yolov8 / yolov11 / rfdetr）。
    /// </summary>
    public TrainingExportResult Export(
        string runDirectory,
        string targetDirectory,
        bool exportOnnx,
        string? filePrefix = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(runDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(targetDirectory);

        string source = Path.Combine(runDirectory, WeightsFileName);

        if (!File.Exists(source))
        {
            throw new FileNotFoundException(
                $"{runDirectory} 里没有 {WeightsFileName}：先训练一轮，或换输出目录。", source);
        }

        Directory.CreateDirectory(targetDirectory);

        using Checkpoint checkpoint = Checkpoint.Open(source);

        string architecture = checkpoint.Metadata.GetValueOrDefault("architecture") ?? string.Empty;

        if (string.IsNullOrWhiteSpace(architecture))
        {
            throw new InvalidDataException($"{WeightsFileName} 缺少 architecture 元数据，无法重建模型。");
        }

        Device device = CPU;
        int imageSize = ReadInt(checkpoint.Metadata, "image_size", 640);
        IReadOnlyList<string> classNames = ReadClassNames(runDirectory);

        // 产物按「模型名_规模」落盘：同一个 models 目录要能同时放下 yolov8_s / yolov11_m / rfdetr，
        // 换规模重训不覆盖上一份
        string prefix = string.IsNullOrWhiteSpace(filePrefix)
            ? DefaultPrefixFor(architecture)
            : filePrefix.Trim();
        string onnxName = prefix + ".onnx";

        string weights = Path.Combine(targetDirectory, prefix + ".safetensors");
        File.Copy(source, weights, overwrite: true);

        string? onnx = null;
        string config = Path.Combine(targetDirectory, prefix + ".json");

        if (YoloArchitectures.TryParseFamily(architecture, out _))
        {
            (YoloModelFamily family, YoloScale scale, int classCount) = YoloArchitectures.Parse(architecture);

            bool nmsFree = false;

            if (exportOnnx)
            {
                onnx = Path.Combine(targetDirectory, onnxName);

                using YoloDetectionModel model = YoloArchitectures.CreateFromArchitecture(device, architecture);
                checkpoint.LoadInto(model, device, strict: false);

                // YOLO26 带 one-to-one 分支，按官方默认导出成无 NMS 的 [1, 300, 6]；
                // 其余家族仍是 [1, 4+nc, anchors] + 外部 NMS。
                nmsFree = model.Head.UsesEndToEnd;
                YoloOnnxExporter.Save(
                    model, onnx, imageHeight: imageSize, imageWidth: imageSize, endToEnd: nmsFree);
            }

            WriteYoloConfig(
                config,
                architecture,
                family,
                scale,
                classCount,
                imageSize,
                classNames,
                checkpoint.Metadata,
                onnx is null ? null : onnxName,
                nmsFree);
        }
        else if (TryParseDetr(architecture, out bool useRfDetr, out int classCount))
        {
            int queries = ReadInt(checkpoint.Metadata, "queries", useRfDetr ? 300 : 50);
            int encoderLayers = ReadInt(checkpoint.Metadata, "encoder_layers", useRfDetr ? 1 : 2);
            int decoderLayers = ReadInt(checkpoint.Metadata, "decoder_layers", 2);
            int heads = ReadInt(checkpoint.Metadata, "heads", 8);
            int dinoEmbedding = ReadInt(checkpoint.Metadata, "dino_embed", 384);
            int dinoDepth = ReadInt(checkpoint.Metadata, "dino_depth", 12);
            int dinoHeads = ReadInt(checkpoint.Metadata, "dino_heads", 6);

            if (exportOnnx)
            {
                onnx = Path.Combine(targetDirectory, onnxName);

                if (useRfDetr)
                {
                    // 与 DetrTrainingSession.BuildModel 的装配方式保持一致，键才对得上。
                    // 用官方权重热启动时结构是照检查点建的，所以这些值只能从元数据里读回来
                    var dinoConfig = new DinoV2Config(
                        ImageSize: ReadInt(checkpoint.Metadata, "dino_image", 518),
                        PatchSize: ReadInt(checkpoint.Metadata, "dino_patch", 14),
                        EmbeddingSize: dinoEmbedding,
                        Depth: dinoDepth,
                        Heads: dinoHeads);

                    using var detector = new RfDetrDetector(
                        classCount,
                        queries,
                        dinoConfig,
                        heads,
                        device,
                        encoderLayers: encoderLayers,
                        decoderLayers: decoderLayers,
                        ffnSize: ReadInt(checkpoint.Metadata, "ffn_size", 1024),
                        points: ReadInt(checkpoint.Metadata, "deform_points", 4),
                        levels: ReadInt(checkpoint.Metadata, "deform_levels", RfDetrDetector.MaxLevels),
                        decoderHeads: ReadInt(checkpoint.Metadata, "deform_heads", 0),
                        useRefPointHead: ReadInt(checkpoint.Metadata, "ref_point", 0) != 0,
                        useC2fProjector: ReadInt(checkpoint.Metadata, "c2f", 0) != 0);

                    checkpoint.LoadInto(detector, device, strict: false);
                    RfDetrOnnxExporter.Save(detector, onnx, imageSize, imageSize);
                }
                else
                {
                    using var model = new SimpleDetrModel(
                        classCount,
                        queries,
                        encoderLayers,
                        decoderLayers,
                        device: device);

                    checkpoint.LoadInto(model, device, strict: false);
                    SimpleDetrOnnxExporter.Save(model, onnx, imageSize, imageSize);
                }
            }

            WriteDetrConfig(
                config,
                architecture,
                useRfDetr,
                classCount,
                imageSize,
                queries,
                encoderLayers,
                decoderLayers,
                heads,
                dinoEmbedding,
                dinoDepth,
                dinoHeads,
                classNames,
                checkpoint.Metadata,
                onnx is null ? null : onnxName);
        }
        else
        {
            throw new NotSupportedException(
                $"不认识的模型架构「{architecture}」：支持 YOLO / RF-DETR / Simple-DETR。");
        }

        return new TrainingExportResult(targetDirectory, weights, config, onnx);
    }

    // ── 配置 json ─────────────────────────────────────────────────────────────

    private static void WriteYoloConfig(
        string path,
        string architecture,
        YoloModelFamily family,
        YoloScale scale,
        int classCount,
        int imageSize,
        IReadOnlyList<string> classNames,
        IReadOnlyDictionary<string, string> metadata,
        string? onnxFileName,
        bool nmsFree)
    {
        // Utf8JsonWriter 直接写：不依赖反射 / 源生成，AOT 或禁用反射的环境同样能导出
        using FileStream stream = File.Create(path);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });

        writer.WriteStartObject();
        writer.WriteString("format", "sevencat-model/1");
        writer.WriteString("architecture", architecture);
        writer.WriteString("family", family switch
        {
            YoloModelFamily.V26 => "yolo26",
            YoloModelFamily.V11 => "yolov11",
            _ => "yolov8",
        });
        writer.WriteString("scale", YoloScales.ToLetter(scale).ToString());
        writer.WriteString("task", "detect");
        writer.WriteNumber("classCount", classCount);
        WriteClasses(writer, classNames);
        writer.WriteNumber("imageSize", imageSize);

        writer.WriteStartObject("input");
        writer.WriteString("name", "input");
        WriteImageShape(writer, imageSize);
        writer.WriteEndObject();

        writer.WriteStartObject("output");
        writer.WriteString("name", "output0");
        writer.WriteStartArray("shape");
        writer.WriteNumberValue(1);

        if (nmsFree)
        {
            // 无 NMS 导出：[1, K, 6]，每行 x1、y1、x2、y2、score、class
            writer.WriteNumberValue(YoloOnnxExporter.DefaultEndToEndDetections);
            writer.WriteNumberValue(6);
        }
        else
        {
            writer.WriteNumberValue(4 + classCount);
            writer.WriteStringValue("anchors");
        }

        writer.WriteEndArray();
        writer.WriteEndObject();

        WriteOnnxAndBest(writer, metadata, onnxFileName);
        writer.WriteEndObject();
        writer.Flush();
    }

    private static void WriteDetrConfig(
        string path,
        string architecture,
        bool useRfDetr,
        int classCount,
        int imageSize,
        int queries,
        int encoderLayers,
        int decoderLayers,
        int heads,
        int dinoEmbedding,
        int dinoDepth,
        int dinoHeads,
        IReadOnlyList<string> classNames,
        IReadOnlyDictionary<string, string> metadata,
        string? onnxFileName)
    {
        using FileStream stream = File.Create(path);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });

        writer.WriteStartObject();
        writer.WriteString("format", "sevencat-model/1");
        writer.WriteString("architecture", architecture);
        writer.WriteString("family", useRfDetr ? "rf-detr" : "simple-detr");
        writer.WriteString("task", "detect");
        writer.WriteNumber("classCount", classCount);
        WriteClasses(writer, classNames);
        writer.WriteNumber("imageSize", imageSize);
        writer.WriteNumber("queries", queries);
        writer.WriteNumber("encoderLayers", encoderLayers);
        writer.WriteNumber("decoderLayers", decoderLayers);
        writer.WriteNumber("heads", heads);

        if (useRfDetr)
        {
            writer.WriteStartObject("dino");

            writer.WriteNumber("embedding", dinoEmbedding);
            writer.WriteNumber("depth", dinoDepth);
            writer.WriteNumber("heads", dinoHeads);

            writer.WriteEndObject();
        }

        writer.WriteStartObject("input");
        writer.WriteString("name", useRfDetr ? RfDetrOnnxExporter.InputName : "input");
        WriteImageShape(writer, imageSize);
        writer.WriteEndObject();

        // RF-DETR / Simple-DETR 的 ONNX 都是 boxes（sigmoid 归一化 cxcywh）+ logits
        writer.WriteStartArray("outputs");

        writer.WriteStringValue(RfDetrOnnxExporter.BoxesOutputName);
        writer.WriteStringValue(RfDetrOnnxExporter.LogitsOutputName);

        writer.WriteEndArray();

        WriteOnnxAndBest(writer, metadata, onnxFileName);
        writer.WriteEndObject();
        writer.Flush();
    }

    private static void WriteClasses(Utf8JsonWriter writer, IReadOnlyList<string> classNames)
    {
        writer.WriteStartArray("classes");

        foreach (string name in classNames)
        {
            writer.WriteStringValue(name);
        }

        writer.WriteEndArray();
    }

    private static void WriteImageShape(Utf8JsonWriter writer, int imageSize)
    {
        writer.WriteStartArray("shape");

        writer.WriteNumberValue(1);
        writer.WriteNumberValue(3);
        writer.WriteNumberValue(imageSize);
        writer.WriteNumberValue(imageSize);

        writer.WriteEndArray();
    }

    private static void WriteOnnxAndBest(
        Utf8JsonWriter writer,
        IReadOnlyDictionary<string, string> metadata,
        string? onnxFileName)
    {
        if (onnxFileName is null)
        {
            writer.WriteNull("onnx");
        }
        else
        {
            writer.WriteString("onnx", onnxFileName);
        }

        writer.WriteStartObject("best");
        WriteMaybeString(writer, "epoch", metadata.GetValueOrDefault("epoch"));
        WriteMaybeString(writer, "loss", metadata.GetValueOrDefault("loss"));
        WriteMaybeString(writer, "map50", metadata.GetValueOrDefault("map50"));
        WriteMaybeString(writer, "map50_95", metadata.GetValueOrDefault("map50-95"));
        writer.WriteEndObject();

        writer.WriteString("exportedAt", DateTimeOffset.Now.ToString("O", CultureInfo.InvariantCulture));
    }

    // ── 辅助 ──────────────────────────────────────────────────────────────────

    private static void WriteMaybeString(Utf8JsonWriter writer, string name, string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            writer.WriteNull(name);
        }
        else
        {
            writer.WriteString(name, value);
        }
    }

    private static int ReadInt(IReadOnlyDictionary<string, string> metadata, string key, int fallback)
        => int.TryParse(
            metadata.GetValueOrDefault(key),
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out int value) && value > 0
            ? value
            : fallback;

    /// <summary>识别 <c>rfdetr-detect-N</c> / <c>simple-detr-detect-N</c>。</summary>
    private static bool TryParseDetr(string architecture, out bool useRfDetr, out int classCount)
    {
        useRfDetr = false;
        classCount = 0;

        string text = architecture.Trim();

        if (text.StartsWith(RfDetrPrefix, StringComparison.OrdinalIgnoreCase))
        {
            useRfDetr = true;
        }
        else if (!text.StartsWith(SimpleDetrPrefix, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        int separator = text.LastIndexOf('-');

        if (separator < 0
            || !int.TryParse(
                text[(separator + 1)..], NumberStyles.Integer, CultureInfo.InvariantCulture, out classCount)
            || classCount <= 0)
        {
            classCount = 0;
            return false;
        }

        return true;
    }

    /// <summary>train.json 里的类别名（没有就空数组，配置里仍带 classCount）。</summary>
    private static IReadOnlyList<string> ReadClassNames(string runDirectory)
    {
        string path = Path.Combine(runDirectory, "train.json");

        if (!File.Exists(path))
        {
            return [];
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

            if (document.RootElement.TryGetProperty("classes", out JsonElement classes)
                && classes.ValueKind == JsonValueKind.Array)
            {
                return [.. classes.EnumerateArray().Select(item => item.GetString() ?? string.Empty)];
            }
        }
        catch (JsonException)
        {
            // train.json 坏了不影响导出，类别名留空即可
        }

        return [];
    }
}
