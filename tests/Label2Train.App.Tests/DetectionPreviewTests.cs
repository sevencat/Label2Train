using Avalonia;
using Avalonia.Headless.XUnit;
using Label2Train.App.Controls;
using Label2Train.App.ViewModels;
using Label2Train.Core.AutoLabel;
using Label2Train.Core.Model;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>检测结果叠框：原图像素框要按图片的实际缩放与偏移落在图上。</summary>
public class DetectionPreviewTests
{
    [Fact]
    public void Image_Is_Fitted_And_Centered()
    {
        (Rect imageRect, double scale) = DetectionPreview.FitToBounds(new Size(1280, 720), new Size(640, 640));

        Assert.Equal(0.5, scale, 3);
        Assert.Equal(new Rect(0, 140, 640, 360), imageRect);
    }

    [Fact]
    public void Box_Is_Mapped_With_The_Same_Scale_And_Offset()
    {
        (Rect imageRect, double scale) = DetectionPreview.FitToBounds(new Size(1280, 720), new Size(640, 640));

        Rect screen = DetectionPreview.MapBox(imageRect, scale, new RectD(128, 72, 256, 144));

        Assert.Equal(new Rect(64, 176, 128, 72), screen);
    }

    [Fact]
    public void Degenerate_Size_Draws_Nothing()
    {
        (Rect imageRect, double scale) = DetectionPreview.FitToBounds(new Size(0, 0), new Size(640, 640));

        Assert.Equal(0, scale);
        Assert.True(imageRect.Width <= 0);
    }

    [AvaloniaFact]
    public void Overlay_Keeps_Candidates_To_Draw()
    {
        var preview = new DetectionPreview();
        IReadOnlyList<AutoLabelCandidate> candidates =
        [
            new AutoLabelCandidate(RectD.FromLtrb(10, 20, 110, 220), "person", 0, 0.9),
        ];

        preview.Candidates = candidates;
        Assert.Same(candidates, preview.Candidates);

        preview.Candidates = null!;
        Assert.Empty(preview.Candidates);
    }
}
