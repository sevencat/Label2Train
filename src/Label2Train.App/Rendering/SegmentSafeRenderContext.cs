using OxyPlot;

namespace Label2Train.App.Rendering;

/// <summary>
/// OxyPlot.Avalonia 画多条线段时，把每一段都 BeginFigure 进同一个 <c>StreamGeometry</c>，
/// 中途不 EndFigure，靠 Dispose 收尾。原生 Skia 后端容忍这种写法，纯托管的 SimdSkia
/// 渲染后端不容忍：第二个 BeginFigure 就抛 InvalidOperationException("End the preceding
/// figure.")，异常被 OxyPlot 接住后整张图只剩错误文字，坐标轴和曲线一条都画不出来。
/// 这里把线段拆成一条条 <c>DrawLine</c>（每条新开一个 geometry），绕开那条路径。
/// </summary>
internal sealed class SegmentSafeRenderContext(IRenderContext inner) : IRenderContext
{
    public bool RendersToScreen => inner.RendersToScreen;

    public int ClipCount => inner.ClipCount;

    public void DrawLineSegments(
        IList<ScreenPoint> points,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
    {
        for (int i = 0; i + 1 < points.Count; i += 2)
        {
            inner.DrawLine([points[i], points[i + 1]], stroke, thickness, edgeRenderingMode, dashArray, lineJoin);
        }
    }

    public void DrawLine(
        IList<ScreenPoint> points,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
        => inner.DrawLine(points, stroke, thickness, edgeRenderingMode, dashArray, lineJoin);

    public void DrawEllipse(OxyRect extents, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
        => inner.DrawEllipse(extents, fill, stroke, thickness, edgeRenderingMode);

    public void DrawEllipses(IList<OxyRect> extents, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
        => inner.DrawEllipses(extents, fill, stroke, thickness, edgeRenderingMode);

    public void DrawPolygon(
        IList<ScreenPoint> points,
        OxyColor fill,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
        => inner.DrawPolygon(points, fill, stroke, thickness, edgeRenderingMode, dashArray, lineJoin);

    public void DrawPolygons(
        IList<IList<ScreenPoint>> polygons,
        OxyColor fill,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
        => inner.DrawPolygons(polygons, fill, stroke, thickness, edgeRenderingMode, dashArray, lineJoin);

    public void DrawRectangle(OxyRect rectangle, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
        => inner.DrawRectangle(rectangle, fill, stroke, thickness, edgeRenderingMode);

    public void DrawRectangles(IList<OxyRect> rectangles, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
        => inner.DrawRectangles(rectangles, fill, stroke, thickness, edgeRenderingMode);

    public void DrawText(
        ScreenPoint p,
        string text,
        OxyColor fill,
        string? fontFamily = null,
        double fontSize = 10,
        double fontWeight = FontWeights.Normal,
        double rotation = 0,
        HorizontalAlignment horizontalAlignment = HorizontalAlignment.Left,
        VerticalAlignment verticalAlignment = VerticalAlignment.Top,
        OxySize? maxSize = null)
        => inner.DrawText(p, text, fill, fontFamily, fontSize, fontWeight, rotation, horizontalAlignment, verticalAlignment, maxSize);

    public OxySize MeasureText(string text, string? fontFamily = null, double fontSize = 10, double fontWeight = 500)
        => inner.MeasureText(text, fontFamily, fontSize, fontWeight);

    public void SetToolTip(string text) => inner.SetToolTip(text);

    public void CleanUp() => inner.CleanUp();

    public void DrawImage(
        OxyImage source,
        double srcX,
        double srcY,
        double srcWidth,
        double srcHeight,
        double destX,
        double destY,
        double destWidth,
        double destHeight,
        double opacity,
        bool interpolate)
        => inner.DrawImage(source, srcX, srcY, srcWidth, srcHeight, destX, destY, destWidth, destHeight, opacity, interpolate);

    public void PushClip(OxyRect clippingRectangle) => inner.PushClip(clippingRectangle);

    public void PopClip() => inner.PopClip();
}
