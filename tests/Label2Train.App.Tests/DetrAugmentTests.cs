using ImageFacade.SkiaSharp;
using Label2Train.Core.Imaging;
using Label2Train.Train.Core;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Data.Yolo;
using Label2Train.Train.Training;
using SkiaSharp;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// DETR 训练集的增强配方：之前训练集与验证集共用「只 letterbox」的参数，
/// 几十张图几十轮就被背下来，加轮数只是拟合噪声。这里把「确实开了增强」钉住。
/// </summary>
public sealed class DetrAugmentTests : IDisposable
{
    private const int Size = 224;

    private readonly string _root;

    public DetrAugmentTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "l2t-detr-augment", Guid.NewGuid().ToString("N"));
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
    public void Train_Preprocess_Enables_Augmentation_Without_Mosaic()
    {
        YoloPreprocessOptions options = TrainingPresets.DetrTrainPreprocess(Size);

        // 步长必须还是 DETR 那一档，否则 336 / 560 这类合法边长会在加载器构造时被拦掉
        Assert.Equal(TrainingPresets.DetrImageSizeStride, options.SizeStride);

        Assert.Equal(0.5f, options.HorizontalFlipProbability);
        Assert.True(options.ScaleGain > 0f);
        Assert.True(options.TranslateGain > 0f);
        Assert.True(options.HueGain > 0f);
        Assert.True(options.SaturationGain > 0f);
        Assert.True(options.ValueGain > 0f);

        // 官方 RF-DETR 配方没有 mosaic：小数据集拼贴后目标常只剩半截
        Assert.Equal(0f, options.MosaicProbability);
    }

    [Fact]
    public void Train_Preprocess_Differs_By_Seed_And_Keeps_Boxes_In_Canvas()
    {
        string path = CreateImage("one.png", 640, 480, new SKRect(240, 160, 400, 320));
        YoloLabel[] labels = [new YoloLabel(0, YoloTask.Detect, [0.5f, 0.5f, 0.25f, 0.25f])];

        using var loader = new YoloSampleLoader(TrainingPresets.DetrTrainPreprocess(Size));

        YoloImageSample first = loader.Load(path, labels, new Rng(11));
        YoloImageSample second = loader.Load(path, labels, new Rng(22));

        Assert.Equal(Size, first.ModelWidth);
        Assert.Equal(Size, first.ModelHeight);
        Assert.Equal(Size * Size * 3, first.Rgb24.Length);
        Assert.False(first.Rgb24.AsSpan().SequenceEqual(second.Rgb24));

        Assert.NotEmpty(first.Boxes);
        Assert.All(first.Boxes, box =>
        {
            Assert.InRange(box.X1, 0f, Size);
            Assert.InRange(box.Y1, 0f, Size);
            Assert.InRange(box.X2, 0f, Size);
            Assert.InRange(box.Y2, 0f, Size);
            Assert.True(box.X2 - box.X1 >= 2f);
            Assert.True(box.Y2 - box.Y1 >= 2f);
        });
    }

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
