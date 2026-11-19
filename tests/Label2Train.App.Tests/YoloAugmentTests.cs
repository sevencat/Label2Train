using ImageFacade.SkiaSharp;
using Label2Train.Core.Imaging;
using Label2Train.Train.Core;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Data.Yolo;
using SkiaSharp;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 训练增强：mosaic 拼贴、随机缩放 / 平移的正确性。
/// 这几项错了不会报错，只会让框慢慢偏，所以要用坐标反变换把它钉住。
/// </summary>
public sealed class YoloAugmentTests : IDisposable
{
    private const int Size = 640;

    private readonly string _root;

    public YoloAugmentTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "l2t-augment", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        ImageRuntime.Use(new SkiaBackend());
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // 临时目录清不掉不影响断言结果
        }
    }

    [Fact]
    public void Mosaic_Composes_Network_Sized_Sample()
    {
        var parts = new List<YoloImageSample>(4);

        for (int index = 0; index < 4; index++)
        {
            string path = CreateImage($"tile{index}.png", 640, 480, new SKRect(200, 140, 440, 340));
            using var loader = new YoloSampleLoader(YoloPreprocessOptions.InferenceDefault(Size));
            parts.Add(loader.Load(path, Labels(), new Rng((ulong)(index + 1))));
        }

        using var mosaicLoader = new YoloSampleLoader(YoloPreprocessOptions.InferenceDefault(Size));
        YoloImageSample mosaic = mosaicLoader.Mosaic(parts, new Rng(11));

        Assert.True(mosaic.IsMosaic);
        Assert.Equal(Size, mosaic.ModelWidth);
        Assert.Equal(Size, mosaic.ModelHeight);
        Assert.Equal(Size * Size * 3, mosaic.Rgb24.Length);

        Assert.NotEmpty(mosaic.Boxes);
        Assert.All(mosaic.Boxes, box =>
        {
            Assert.InRange(box.X1, 0f, Size);
            Assert.InRange(box.Y1, 0f, Size);
            Assert.InRange(box.X2, 0f, Size);
            Assert.InRange(box.Y2, 0f, Size);
            Assert.True(box.X2 - box.X1 >= 2f);
            Assert.True(box.Y2 - box.Y1 >= 2f);
        });
    }

    [Fact]
    public void Scale_And_Translate_Map_Boxes_Back_To_Source()
    {
        string path = CreateImage("one.png", 640, 480, new SKRect(240, 160, 400, 320));

        using var loader = new YoloSampleLoader(YoloPreprocessOptions.TrainDefault(Size));

        // 同一个种子序列跑多轮：缩放 / 平移 / 翻转随机，反变换回去必须还是原图的那个框。
        int checkedSamples = 12;

        for (int seed = 1; seed <= 12; seed++)
        {
            YoloImageSample sample = loader.Load(path, Labels(), new Rng((ulong)seed));

            YoloTargetBox box = Assert.Single(sample.Boxes);

            Assert.InRange(box.X1, 0f, Size);
            Assert.InRange(box.Y1, 0f, Size);
            Assert.InRange(box.X2, 0f, Size);
            Assert.InRange(box.Y2, 0f, Size);
            Assert.True(box.X2 - box.X1 >= 2f);
            Assert.True(box.Y2 - box.Y1 >= 2f);

            // 翻转是在画布空间做的，Transform 不记录它（只由 Flipped 标记），
            // 所以反变换只在未翻转的样本上自洽。
            if (sample.Flipped)
            {
                checkedSamples--;
                continue;
            }

            (float x, float y, float width, float height) = sample.Transform.Inverse(
                box.X1, box.Y1, box.X2 - box.X1, box.Y2 - box.Y1);

            // 标注框 0.5/0.5/0.25/0.25 在 640x480 上就是 (240,180)-(400,300)
            Assert.Equal(240f, x, 1.5f);
            Assert.Equal(180f, y, 1.5f);
            Assert.Equal(160f, width, 1.5f);
            Assert.Equal(120f, height, 1.5f);
        }

        Assert.True(checkedSamples >= 3, $"只有 {checkedSamples} 个未翻转样本，覆盖不足。");
    }

    [Fact]
    public void Inference_Default_Disables_All_Random_Augmentations()
    {
        YoloPreprocessOptions options = YoloPreprocessOptions.InferenceDefault(640);

        Assert.Equal(0f, options.HorizontalFlipProbability);
        Assert.Equal(0f, options.MosaicProbability);
        Assert.Equal(0f, options.ScaleGain);
        Assert.Equal(0f, options.TranslateGain);
        Assert.Equal(0f, options.HueGain);
    }

    private static YoloLabel[] Labels() => [new YoloLabel(0, YoloTask.Detect, [0.5f, 0.5f, 0.25f, 0.25f])];

    private string CreateImage(string name, int width, int height, SKRect box)
    {
        using var bitmap = new SKBitmap(width, height);

        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(new SKColor(114, 114, 114));
            using var paint = new SKPaint { Color = new SKColor(220, 60, 60) };
            canvas.DrawRect(box, paint);
        }

        string path = Path.Combine(_root, name);

        using (SKImage image = SKImage.FromBitmap(bitmap))
        using (SKData data = image.Encode(SKEncodedImageFormat.Png, 100))
        using (FileStream stream = File.Create(path))
        {
            data.SaveTo(stream);
        }

        return path;
    }
}
