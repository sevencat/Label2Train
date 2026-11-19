using ImageFacade;
using ImageFacade.Backends;
using ImageFacade.Pipeline;
using Label2Train.Train.Core;
using Label2Train.Train.Data.Geometry;
using Label2Train.Train.Data.Yolo;

namespace Label2Train.Train.Data.Images;

/// <summary>
/// 把一张 YOLO 检测样本读成「网络尺寸的 RGB 像素 + 目标框」：
/// 解码（宿主注册的 ImageFacade 后端，本项目是 SkiaSharp）→ letterbox → 翻转 / HSV → 目标框映射；
/// 另有 <see cref="LoadMosaic"/> 做 4 图拼贴。
/// <para>
/// 解码引擎来自 <see cref="ImageCodecs.Engine"/>（宿主在 exe 入口注册的进程级单例），
/// 加载器本身不持有后端；训练管线是单线程的，不要跨线程共享同一个加载器。
/// </para>
/// </summary>
public sealed class YoloSampleLoader : IDisposable
{
    /// <summary>裁剪后小于这个像素边长的框直接丢弃（与 ultralytics 的 <c>min box</c> 过滤一致）。</summary>
    private const float MinimumBoxSize = 2f;

    private bool _disposed;

    private static ImageEngine Engine => ImageCodecs.Engine;

    /// <summary>按预处理参数创建加载器（参数非法时立刻抛错）。</summary>
    public YoloSampleLoader(YoloPreprocessOptions? options = null)
    {
        Options = (options ?? new YoloPreprocessOptions()).Validate();
    }

    /// <summary>当前预处理参数。</summary>
    public YoloPreprocessOptions Options { get; }

    /// <summary>加载一条 <see cref="YoloSample"/>（图片 + 已解析的标签，含翻转 / HSV）。</summary>
    public YoloImageSample Load(YoloSample sample, Rng? rng = null)
    {
        ArgumentNullException.ThrowIfNull(sample);
        return Load(sample.ImagePath, sample.Labels, rng);
    }

    /// <summary>按图片路径与标签加载；<paramref name="labels"/> 为空表示背景图。</summary>
    public YoloImageSample Load(string imagePath, IReadOnlyList<YoloLabel>? labels = null, Rng? rng = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);
        rng ??= new Rng(1);

        string fullPath = Path.GetFullPath(imagePath);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"图像不存在：{fullPath}", fullPath);

        using FacadeImage source = Engine.Load(fullPath, new LoadOptions { AutoOrient = true });
        return BuildSample(source, fullPath, labels, rng);
    }

    /// <summary>
    /// 从已编码的图像流加载：数据集走 <c>IBlobStore</c>（远端 / 压缩包 / 数据库）时没有本地路径，
    /// 直接把字节流交给解码器；<paramref name="displayName"/> 只用于报错与日志。
    /// </summary>
    public YoloImageSample Load(
        Stream encodedImage,
        string displayName,
        IReadOnlyList<YoloLabel>? labels = null,
        Rng? rng = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(encodedImage);
        ArgumentException.ThrowIfNullOrWhiteSpace(displayName);
        rng ??= new Rng(1);

        using FacadeImage source = Engine.Load(encodedImage, new LoadOptions { AutoOrient = true });
        return BuildSample(source, displayName, labels, rng);
    }

    /// <summary>letterbox（可选缩放 / 平移）→ 目标框映射 → 翻转 / HSV。</summary>
    private YoloImageSample BuildSample(
        FacadeImage source,
        string displayPath,
        IReadOnlyList<YoloLabel>? labels,
        Rng rng)
    {
        int modelWidth = Options.Width;
        int modelHeight = Options.Height;

        // 不缩放 / 不平移时走原路径：验证与推理必须和导出端逐像素一致。
        if (Options.ScaleGain <= 0f && Options.TranslateGain <= 0f)
        {
            using FacadeImage plain = Letterbox.Apply(
                source, LetterboxOptionsFor(modelWidth, modelHeight), out LetterboxInfo plainInfo);
            byte[] plainPixels = plain.CopyPixels(PixelFormat.Rgb24);

            LetterboxTransform plainTransform = CreateTransform(plainInfo);
            var plainBoxes = new List<YoloTargetBox>(BuildBoxes(
                labels ?? Array.Empty<YoloLabel>(),
                plainTransform,
                displayPath));

            bool plainFlipped = ApplyAugment(plainPixels, plainBoxes, rng);

            return new YoloImageSample(
                displayPath,
                plainInfo.SourceSize.Width,
                plainInfo.SourceSize.Height,
                modelWidth,
                modelHeight,
                plainPixels,
                plainTransform,
                plainBoxes,
                plainFlipped);
        }

        // 随机缩放：先 letterbox 到 (1±ScaleGain) 倍的尺寸，再贴回网络尺寸画布。
        float zoom = Math.Clamp(
            rng.NextFloat(1f - Options.ScaleGain, 1f + Options.ScaleGain),
            0.25f,
            4f);

        int fitWidth = Math.Max(32, (int)MathF.Round(modelWidth * zoom));
        int fitHeight = Math.Max(32, (int)MathF.Round(modelHeight * zoom));

        using FacadeImage scaled = Letterbox.Apply(
            source, LetterboxOptionsFor(fitWidth, fitHeight), out LetterboxInfo info);
        byte[] scaledPixels = scaled.CopyPixels(PixelFormat.Rgb24);

        int offsetX = RandomOffset(rng, modelWidth, fitWidth, Options.TranslateGain);
        int offsetY = RandomOffset(rng, modelHeight, fitHeight, Options.TranslateGain);

        var canvas = new RgbCanvas(modelWidth, modelHeight, Options.PadValue);
        canvas.Blit(scaledPixels, fitWidth, fitHeight, offsetX, offsetY);

        // 缩放与平移都是可加的平移仿射：scale 沿用 letterbox，pad 加上画布偏移即可。
        var transform = new LetterboxTransform(
            info.Scale,
            info.Scale,
            info.PadLeft + offsetX,
            info.PadTop + offsetY,
            info.SourceSize.Width,
            info.SourceSize.Height,
            modelWidth,
            modelHeight);

        var boxes = ClipBoxes(
            BuildBoxes(labels ?? Array.Empty<YoloLabel>(), transform, displayPath),
            modelWidth,
            modelHeight);

        bool flipped = ApplyAugment(canvas.Pixels, boxes, rng);

        return new YoloImageSample(
            displayPath,
            info.SourceSize.Width,
            info.SourceSize.Height,
            modelWidth,
            modelHeight,
            canvas.Pixels,
            transform,
            boxes,
            flipped);
    }

    private LetterboxOptions LetterboxOptionsFor(int width, int height) => new()
    {
        Width = width,
        Height = height,
        PadColor = new Rgba32(Options.PadValue, Options.PadValue, Options.PadValue, 255),
        Sampler = Sampler.Bilinear,
        Center = true,
    };

    /// <summary>
    /// 内容贴到画布时的起始偏移：比画布小时在余量里随机放，比画布大时随机裁掉多余部分；
    /// <paramref name="translate"/> 为 0 时居中（等同普通 letterbox）。
    /// </summary>
    private static int RandomOffset(Rng rng, int canvasSize, int contentSize, float translate)
    {
        int low = Math.Min(0, canvasSize - contentSize);
        int high = Math.Max(0, canvasSize - contentSize);
        int centered = (canvasSize - contentSize) / 2;

        if (translate <= 0f || low == high)
            return Math.Clamp(centered, low, high);

        int jitter = (int)MathF.Round(rng.NextFloat(-translate, translate) * canvasSize);
        return Math.Clamp(centered + jitter, low, high);
    }

    /// <summary>裁剪到画布内，并丢掉裁完小于 <see cref="MinimumBoxSize"/> 的框。</summary>
    private static List<YoloTargetBox> ClipBoxes(IReadOnlyList<YoloTargetBox> boxes, int width, int height)
    {
        var clipped = new List<YoloTargetBox>(boxes.Count);

        foreach (YoloTargetBox box in boxes)
        {
            float x1 = Math.Clamp(box.X1, 0f, width);
            float y1 = Math.Clamp(box.Y1, 0f, height);
            float x2 = Math.Clamp(box.X2, 0f, width);
            float y2 = Math.Clamp(box.Y2, 0f, height);

            if (x2 - x1 < MinimumBoxSize || y2 - y1 < MinimumBoxSize)
                continue;

            clipped.Add(box with { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2 });
        }

        return clipped;
    }

    /// <summary>
    /// 4 图 mosaic：把 4 张已 letterbox 的图贴到 2 倍画布的四个象限，
    /// 再随机裁出 <see cref="YoloPreprocessOptions.Width"/>×<see cref="YoloPreprocessOptions.Height"/>。
    /// <para>
    /// 少于 4 条样本时循环复用（小数据集也能训练）；裁剪后小于 2px 的框会被丢掉，
    /// 其余框裁剪到窗口内，坐标仍是网络输入像素。
    /// </para>
    /// </summary>
    public YoloImageSample LoadMosaic(IReadOnlyList<YoloSample> samples, Rng? rng = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0)
            throw new ArgumentException("mosaic 至少需要 1 条样本。", nameof(samples));

        var tiles = new List<MosaicTile>(4);

        for (int index = 0; index < 4; index++)
        {
            YoloSample sample = samples[index % samples.Count];
            (string fullPath, Decoded decoded) = Decode(sample.ImagePath);

            LetterboxTransform transform = CreateTransform(decoded.Info);
            tiles.Add(new MosaicTile(decoded.Rgb, BuildBoxes(sample.Labels, transform, fullPath)));
        }

        return ComposeMosaic(tiles, samples[0].ImagePath, rng ?? new Rng(1));
    }

    /// <summary>
    /// 把已经 letterbox 好的若干张样本拼成 2 倍画布再随机裁回网络尺寸。
    /// 数据集走 <c>IBlobStore</c> 时没有本地路径，调用方先加载再交给这里，
    /// 免得为了 mosaic 再解一次码、或者让加载器依赖文件系统。
    /// </summary>
    public YoloImageSample Mosaic(IReadOnlyList<YoloImageSample> images, Rng? rng = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
            throw new ArgumentException("mosaic 至少需要 1 张已解码样本。", nameof(images));

        var tiles = new List<MosaicTile>(4);

        for (int index = 0; index < 4; index++)
        {
            YoloImageSample image = images[index % images.Count];

            if (image.ModelWidth != Options.Width || image.ModelHeight != Options.Height)
            {
                throw new ArgumentException(
                    $"mosaic 的输入必须是 {Options.Width}x{Options.Height}，"
                    + $"实际 {image.ModelWidth}x{image.ModelHeight}（{image.ImagePath}）。",
                    nameof(images));
            }

            tiles.Add(new MosaicTile(image.Rgb24, image.Boxes));
        }

        // 单图加载时已经做过 hflip / scale / translate / HSV，这里再整体翻转一次就过强了。
        return ComposeMosaic(tiles, images[0].ImagePath, rng ?? new Rng(1), applyAugment: false);
    }

    /// <summary>4 象限拼贴 + 随机裁剪；<paramref name="tiles"/> 的框须是网络输入像素坐标。</summary>
    private YoloImageSample ComposeMosaic(
        IReadOnlyList<MosaicTile> tiles,
        string imagePath,
        Rng rng,
        bool applyAugment = true)
    {
        int width = Options.Width;
        int height = Options.Height;
        var canvas = new RgbCanvas(width * 2, height * 2, Options.PadValue);

        // 拼贴中心在画布中部 1 倍尺寸范围内抖动，保证 4 个象限都有机会露出。
        int centerX = (int)MathF.Round(rng.NextFloat(0.5f, 1.5f) * width);
        int centerY = (int)MathF.Round(rng.NextFloat(0.5f, 1.5f) * height);

        (int X, int Y)[] origins =
        {
            (centerX - width, centerY - height),
            (centerX, centerY - height),
            (centerX - width, centerY),
            (centerX, centerY),
        };

        var canvasBoxes = new List<YoloTargetBox>();

        for (int index = 0; index < origins.Length; index++)
        {
            MosaicTile tile = tiles[index % tiles.Count];
            (int offsetX, int offsetY) = origins[index];

            canvas.Blit(tile.Rgb, width, height, offsetX, offsetY);

            foreach (YoloTargetBox box in tile.Boxes)
            {
                canvasBoxes.Add(new YoloTargetBox(
                    box.ClassId,
                    box.X1 + offsetX,
                    box.Y1 + offsetY,
                    box.X2 + offsetX,
                    box.Y2 + offsetY));
            }
        }

        int cropX = Math.Clamp(centerX - width / 2, 0, width);
        int cropY = Math.Clamp(centerY - height / 2, 0, height);

        byte[] cropped = canvas.Crop(cropX, cropY, width, height);

        var boxes = new List<YoloTargetBox>(canvasBoxes.Count);
        foreach (YoloTargetBox box in canvasBoxes)
        {
            float x1 = Math.Clamp(box.X1 - cropX, 0f, width);
            float y1 = Math.Clamp(box.Y1 - cropY, 0f, height);
            float x2 = Math.Clamp(box.X2 - cropX, 0f, width);
            float y2 = Math.Clamp(box.Y2 - cropY, 0f, height);

            if (x2 - x1 < MinimumBoxSize || y2 - y1 < MinimumBoxSize)
                continue;

            boxes.Add(new YoloTargetBox(box.ClassId, x1, y1, x2, y2));
        }

        bool flipped = applyAugment && ApplyAugment(cropped, boxes, rng);

        // 画布 → 裁剪窗口：scale = 1，pad = -crop，因此 Forward/Inverse 仍是自洽的仿射变换。
        var cropTransform = new LetterboxTransform(
            1f,
            1f,
            -cropX,
            -cropY,
            width * 2,
            height * 2,
            width,
            height);

        return new YoloImageSample(
            imagePath,
            width * 2,
            height * 2,
            width,
            height,
            cropped,
            cropTransform,
            boxes,
            flipped,
            isMosaic: true);
    }

    /// <summary>释放底层图像引擎。</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }

    private (string FullPath, Decoded Decoded) Decode(string imagePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);

        string fullPath = Path.GetFullPath(imagePath);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"图像不存在：{fullPath}", fullPath);

        using FacadeImage source = Engine.Load(fullPath, new LoadOptions { AutoOrient = true });

        var letterboxOptions = new LetterboxOptions
        {
            Width = Options.Width,
            Height = Options.Height,
            PadColor = new Rgba32(Options.PadValue, Options.PadValue, Options.PadValue, 255),
            Sampler = Sampler.Bilinear,
            Center = true,
        };

        using FacadeImage letterboxed = Letterbox.Apply(source, letterboxOptions, out LetterboxInfo info);
        return (fullPath, new Decoded(letterboxed.CopyPixels(PixelFormat.Rgb24), info));
    }

    private static LetterboxTransform CreateTransform(LetterboxInfo info) => new(
        info.Scale,
        info.Scale,
        info.PadLeft,
        info.PadTop,
        info.SourceSize.Width,
        info.SourceSize.Height,
        info.ModelSize.Width,
        info.ModelSize.Height);

    /// <summary>按概率做翻转 / HSV；返回是否翻转（框会同步镜像）。</summary>
    private bool ApplyAugment(byte[] rgb, List<YoloTargetBox> boxes, Rng rng)
    {
        bool flipped = Options.HorizontalFlipProbability > 0f
            && rng.NextFloat() < Options.HorizontalFlipProbability;

        if (flipped)
        {
            RgbAugment.FlipHorizontal(rgb, Options.Width, Options.Height);

            for (int index = 0; index < boxes.Count; index++)
            {
                YoloTargetBox box = boxes[index];
                boxes[index] = box with
                {
                    X1 = Options.Width - box.X2,
                    X2 = Options.Width - box.X1,
                };
            }
        }

        if (Options.HueGain > 0f || Options.SaturationGain > 0f || Options.ValueGain > 0f)
        {
            // 与 ultralytics 一致：每次增强采样一次 [-1,1] 的随机系数。
            float hueDegrees = rng.NextFloat(-1f, 1f) * Options.HueGain * 180f;
            float saturationScale = 1f + rng.NextFloat(-1f, 1f) * Options.SaturationGain;
            float valueScale = 1f + rng.NextFloat(-1f, 1f) * Options.ValueGain;

            RgbAugment.AdjustHsv(rgb, hueDegrees, saturationScale, valueScale);
        }

        return flipped;
    }

    private static IReadOnlyList<YoloTargetBox> BuildBoxes(
        IReadOnlyList<YoloLabel> labels,
        LetterboxTransform transform,
        string imagePath)
    {
        if (labels.Count == 0)
            return Array.Empty<YoloTargetBox>();

        var boxes = new List<YoloTargetBox>(labels.Count);

        foreach (YoloLabel label in labels)
        {
            if (label.Task != YoloTask.Detect)
            {
                throw new NotSupportedException(
                    $"{imagePath}：P1 只支持 detect 标签（class cx cy w h），实际是 {label.Task}。");
            }

            float pixelWidth = label.Values[2] * transform.SourceWidth;
            float pixelHeight = label.Values[3] * transform.SourceHeight;
            float pixelX = label.Values[0] * transform.SourceWidth - pixelWidth / 2f;
            float pixelY = label.Values[1] * transform.SourceHeight - pixelHeight / 2f;

            (float modelX, float modelY, float modelBoxWidth, float modelBoxHeight) =
                transform.Forward(pixelX, pixelY, pixelWidth, pixelHeight);

            boxes.Add(new YoloTargetBox(
                label.ClassId,
                modelX,
                modelY,
                modelX + modelBoxWidth,
                modelY + modelBoxHeight));
        }

        return boxes;
    }

    private sealed record Decoded(byte[] Rgb, LetterboxInfo Info);

    /// <summary>mosaic 的一块：已 letterbox 的像素 + 网络坐标下的框。</summary>
    private readonly record struct MosaicTile(byte[] Rgb, IReadOnlyList<YoloTargetBox> Boxes);
}
