using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Label2Train.Core.AutoLabel;
using Label2Train.Core.Model;

namespace Label2Train.App.Controls;

/// <summary>
/// 检测结果预览：把原图等比居中画进控件，再把候选框按同一缩放叠上去。
/// 框的坐标是原图像素，控件大小随窗口变，所以缩放只在这里算一次。
/// </summary>
public sealed class DetectionPreview : Control
{
    public static readonly StyledProperty<IImage?> SourceProperty =
        AvaloniaProperty.Register<DetectionPreview, IImage?>(nameof(Source));

    /// <summary>控件底色：浅色，图片以外留白。</summary>
    public static readonly StyledProperty<IBrush?> BackgroundProperty =
        AvaloniaProperty.Register<DetectionPreview, IBrush?>(
            nameof(Background),
            new SolidColorBrush(Color.FromRgb(0xF2, 0xF3, 0xF7)));

    private IReadOnlyList<AutoLabelCandidate> _candidates = [];

    static DetectionPreview()
    {
        AffectsRender<DetectionPreview>(SourceProperty, BackgroundProperty);
    }

    public DetectionPreview()
    {
        ClipToBounds = true;
    }

    public IImage? Source
    {
        get => GetValue(SourceProperty);
        set => SetValue(SourceProperty, value);
    }

    public IBrush? Background
    {
        get => GetValue(BackgroundProperty);
        set => SetValue(BackgroundProperty, value);
    }

    /// <summary>要叠在图上画出来的框（原图像素坐标）。</summary>
    public IReadOnlyList<AutoLabelCandidate> Candidates
    {
        get => _candidates;
        set
        {
            _candidates = value ?? [];
            InvalidateVisual();
        }
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);

        context.FillRectangle(Background ?? Brushes.Transparent, new Rect(Bounds.Size));

        if (Source is not { } image)
        {
            return;
        }

        (Rect imageRect, double scale) = FitToBounds(image.Size, Bounds.Size);

        if (imageRect.Width <= 0 || imageRect.Height <= 0)
        {
            return;
        }

        context.DrawImage(image, new Rect(image.Size), imageRect);

        foreach (AutoLabelCandidate candidate in _candidates)
        {
            DrawCandidate(context, MapBox(imageRect, scale, candidate.Box), candidate);
        }
    }

    /// <summary>图片在控件里等比居中的目标矩形，以及原图像素到控件像素的缩放。</summary>
    public static (Rect ImageRect, double Scale) FitToBounds(Size imageSize, Size bounds)
    {
        if (imageSize.Width <= 0 || imageSize.Height <= 0 || bounds.Width <= 0 || bounds.Height <= 0)
        {
            return (default, 0);
        }

        double scale = Math.Min(bounds.Width / imageSize.Width, bounds.Height / imageSize.Height);
        double width = imageSize.Width * scale;
        double height = imageSize.Height * scale;

        return (new Rect((bounds.Width - width) / 2, (bounds.Height - height) / 2, width, height), scale);
    }

    /// <summary>原图像素框 → 控件里的屏幕矩形。</summary>
    public static Rect MapBox(Rect imageRect, double scale, RectD box)
        => new(
            imageRect.X + (box.X * scale),
            imageRect.Y + (box.Y * scale),
            box.Width * scale,
            box.Height * scale);

    private static void DrawCandidate(DrawingContext context, Rect screen, AutoLabelCandidate candidate)
    {
        IBrush brush = LabelPalette.BrushFor(candidate.Label);

        context.DrawRectangle(null, new Pen(brush, 2), screen);

        var text = new FormattedText(
            $"{candidate.Label} {candidate.ScoreText}",
            CultureInfo.CurrentCulture,
            FlowDirection.LeftToRight,
            Typeface.Default,
            12,
            Brushes.White);

        // 标签贴在框的上沿，贴顶时改挂在框内侧，别跑到画面外
        var tag = new Rect(
            screen.X,
            Math.Max(0, screen.Y - 18),
            text.Width + 8,
            18);

        context.FillRectangle(brush, tag);
        context.DrawText(text, new Point(tag.X + 4, tag.Y + 2));
    }
}
