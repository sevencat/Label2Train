using ImageFacade;
using ImageFacade.Pipeline;
using Label2Train.Core.Imaging;
using Label2Train.Train.Data.Geometry;
using Label2Train.Train.Data.Images;
using OnnxSharp.Vision;
using Xunit;
using FacadeLetterbox = ImageFacade.Pipeline.Letterbox;

namespace Label2Train.App.Tests;

/// <summary>
/// 训练端（ImageFacade）与推理端（OnnxSharp）必须把同一张图放在同一位置、同一通道顺序，
/// 否则框会整体偏移。这两个包的 letterbox 各写各的，升级后最容易悄悄分叉。
/// </summary>
public class LetterboxParityTests
{
    private const int Width = 640;
    private const int Height = 480;
    private const int Target = 640;

    /// <summary>640x480 图 letterbox 到 640x640：scale=1，上下各 pad 80，内容从 y=80 开始。</summary>
    [Fact]
    public void Training_Letterbox_Matches_Expected_Geometry()
    {
        ImageRuntime.Use(new ImageFacade.SkiaSharp.SkiaBackend());

        byte[] rgb = new byte[Width * Height * 3];
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                int i = ((y * Width) + x) * 3;
                rgb[i] = 255;
                rgb[i + 1] = 255;
                rgb[i + 2] = 255;
            }
        }

        using FacadeImage source = ImageCodecs.Engine.FromPixels(
            rgb, Width, Height, Width * 3, PixelFormat.Rgb24);

        using FacadeImage letterboxed = FacadeLetterbox.Apply(
            source,
            new LetterboxOptions
            {
                Width = Target,
                Height = Target,
                PadColor = new Rgba32(114, 114, 114, 255),
                Sampler = Sampler.Bilinear,
                Center = true,
            },
            out LetterboxInfo info);

        byte[] output = letterboxed.CopyPixels(PixelFormat.Rgb24);

        Assert.Equal(1f, info.Scale, 3);
        Assert.Equal(0f, info.PadLeft, 3);
        Assert.Equal(80f, info.PadTop, 3);
        Assert.Equal((114, 114, 114), (output[0], output[1], output[2]));

        int firstWhiteRow = -1;
        for (int y = 0; y < Target && firstWhiteRow < 0; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                int i = ((y * Target) + x) * 3;
                if (output[i] > 250 && output[i + 1] > 250 && output[i + 2] > 250)
                {
                    firstWhiteRow = y;
                    break;
                }
            }
        }

        // 内容贴在图上方时，letterbox 之后必须从 pad 之后的第一行开始（居中而非左上角对齐）
        Assert.Equal(80, firstWhiteRow);
    }

    /// <summary>
    /// 推理端把 640x480 的 BGR 缓冲喂进去：几何必须同样是 scale=1 / padY=80，
    /// 且通道要按 BGR 解读（纯红的 c0=R=1、c2=B=0），与训练端 RGB 平面一致。
    /// </summary>
    [Fact]
    public void Inference_Letterbox_Matches_Training_Geometry_And_Channels()
    {
        byte[] bgr = new byte[Width * Height * 3];
        for (int i = 0; i < bgr.Length; i += 3)
        {
            bgr[i + 2] = 255;
        }

        (OnnxSharp.Tensors.Tensor input, OnnxSharp.Vision.Letterbox lb) = ImagePreprocessor.Letterbox(
            bgr,
            Width,
            Height,
            Target,
            Target,
            bgr: true,
            rowStride: Width * 3);

        Assert.Equal(1f, (float)lb.ScaleX, 3);
        Assert.Equal(0f, (float)lb.PadX, 3);
        Assert.Equal(80f, (float)lb.PadY, 3);

        float[] data = input.AsSpan<float>().ToArray();
        int planeSize = Target * Target;
        int sample = (320 * Target) + 320;

        Assert.Equal(1f, data[sample], 2);
        Assert.Equal(0f, data[planeSize + sample], 2);
        Assert.Equal(0f, data[(2 * planeSize) + sample], 2);

        LetterboxTransform expected = LetterboxTransform.Compute(Width, Height, Target, Target);
        Assert.Equal(expected.ScaleX, (float)lb.ScaleX, 3);
        Assert.Equal(expected.PadY, (float)lb.PadY, 3);
    }
}
