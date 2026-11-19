using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.VisualTree;
using Label2Train.App.ViewModels;
using Label2Train.App.Views;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Localization;
using OxyPlot;
using OxyPlot.Avalonia;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>训练工作台：图表、参数行、模型下拉都在（训练页现在是独立控件）。</summary>
public class TrainViewTests
{
    [AvaloniaFact]
    public async Task Renders_Chart_And_Parameters()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);

        // 曲线区（mAP 左轴 / loss 右轴）和空闲占位都在
        PlotView chart = Assert.Single(view.GetVisualDescendants().OfType<PlotView>());
        Assert.Same(host.ViewModel.Train.Plot, chart.Model);

        // PlotView 是模板控件：OxyPlot 的主题没加载时它没有模板，训练中图表一片空白
        Assert.NotNull(chart.Template);

        HashSet<string> texts = [.. view
            .GetVisualDescendants()
            .OfType<TextBlock>()
            .Select(block => block.Text ?? string.Empty)];

        Assert.Contains("epochs", texts);
        Assert.Contains("batch", texts);
        Assert.Contains("imgsz", texts);
        Assert.Contains(Localizer.Instance.Get("Train.ProductFile"), texts);
    }

    [AvaloniaFact]
    public async Task Model_Picker_Is_Populated_From_ViewModel()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);
        List<ComboBox> pickers = [.. view.GetVisualDescendants().OfType<ComboBox>()];

        // 页面上不止一个下拉（数据集 / 模型），按候选项数认出模型那个
        ComboBox modelPicker = Assert.Single(
            pickers,
            picker => picker.ItemCount == host.ViewModel.Train.ModelFamilies.Count);

        Assert.Same(host.ViewModel.Train.SelectedModel, modelPicker.SelectedItem);
    }

    /// <summary>参数对话框要能起来：x:Name 写错、提示控件缺失都会在这里炸，而不是在用户点「修改」的时候。</summary>
    [AvaloniaFact]
    public void Params_Dialog_Opens_With_Detr_Hints()
    {
        var dialog = new TrainingParamsDialog(100m, 4m, 640m, "rfdetr");

        Assert.NotNull(dialog);
    }

    /// <summary>三个检测单选的绑定：点哪个就换哪个模型，检测时按它的前缀取 onnx。</summary>
    [AvaloniaFact]
    public async Task Detect_Model_Radios_Switch_Selection()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);
        List<RadioButton> radios =
        [
            .. view.GetVisualDescendants()
                .OfType<RadioButton>()
                .Where(radio => radio.Content is "YOLOv8" or "YOLOv11" or "RF-DETR")
        ];

        Assert.Equal(3, radios.Count);
        Assert.Equal("yolov8", host.ViewModel.Train.SelectedDetectModel.FilePrefix);

        radios.Single(radio => (string?)radio.Content == "RF-DETR").IsChecked = true;
        Assert.Equal("rfdetr", host.ViewModel.Train.SelectedDetectModel.FilePrefix);
        Assert.True(host.ViewModel.Train.IsDetectRfDetr);

        radios.Single(radio => (string?)radio.Content == "YOLOv11").IsChecked = true;
        Assert.Equal("yolov11", host.ViewModel.Train.SelectedDetectModel.FilePrefix);
        Assert.True(host.ViewModel.Train.IsDetectYolov11);
    }

    [AvaloniaFact]
    public async Task Chart_Renders_Before_During_And_After_Training()
    {
        await using AppTestHost host = AppTestHost.Create();
        TrainViewModel train = host.ViewModel.Train;

        // 空图：轴范围/步长是 NaN 时，OxyPlot 算刻度会对着 NaN 取符号而抛
        // ArithmeticException，训练页的曲线区就整块空白
        Assert.False(host.ViewModel.Train.HasChartData);
        RenderChart(train);

        train.PushEpoch(loss: 195.33, map50: 0.0, map5095: 0.0);
        train.PushEpoch(loss: 42.5, map50: 0.31, map5095: 0.12);

        // 指标算不出来时会是 NaN（首轮没有 mAP、loss 溢出都可能），别让它把轴范围带成 NaN
        train.PushEpoch(loss: double.NaN, map50: double.NaN, map5095: double.NaN);
        RenderChart(train);

        train.ClearChart();
        RenderChart(train);
    }

    /// <summary>用 SVG 导出真跑一遍 <c>PlotModel.Render</c>：轴范围的哨兵值有问题这里就会抛。</summary>
    private static void RenderChart(TrainViewModel train)
    {
        string svg = new OxyPlot.SvgExporter { Width = 480, Height = 320 }.ExportToString(train.Plot);

        Assert.Contains("<svg", svg, StringComparison.Ordinal);
        Assert.All(
            train.Plot.Axes,
            axis =>
            {
                Assert.False(double.IsNaN(axis.Minimum));
                Assert.False(double.IsNaN(axis.Maximum));

                // 关键：OxyPlot 结尾做 ActualMajorStep = Math.Max(步长, MinimumMajorStep)，
                // 这里留 NaN 会让步长变 NaN，CreateTickValues 里 Math.Sign(NaN) 直接抛
                Assert.False(double.IsNaN(axis.MinimumMajorStep), $"{axis.Key} 的主刻度步长下限是 NaN");
                Assert.True(double.IsFinite(axis.MinimumMinorStep), $"{axis.Key} 的次刻度步长下限是 {axis.MinimumMinorStep}");
            });
    }

    [AvaloniaFact]
    public async Task Chart_Draws_Segments_One_By_One()
    {
        await using AppTestHost host = AppTestHost.Create();
        PlotModel plot = host.ViewModel.Train.Plot;

        // OxyPlot.Avalonia 一次 BeginFigure 画多条线段的写法会被 SimdSkia 拒掉
        // （"End the preceding figure."），渲染上下文会把它们拆成单条线
        var recorder = new LineRecordingRenderContext();
        IRenderContext decorated = Assert.IsAssignableFrom<IRenderContext>(plot.RenderingDecorator?.Invoke(recorder));

        decorated.DrawLineSegments(
            [
                new ScreenPoint(0, 0), new ScreenPoint(10, 0),
                new ScreenPoint(0, 5), new ScreenPoint(10, 5),
            ],
            OxyColors.Black,
            1,
            EdgeRenderingMode.Automatic);

        Assert.Empty(recorder.SegmentBatches);
        Assert.Equal(2, recorder.LineCount);
    }

    [AvaloniaFact]
    public async Task TinyLabel_Uses_The_Annotation_Database()
    {
        await using AppTestHost host = AppTestHost.Create();

        // 标注端生成一个项目，训练端应当从同一个库里直接看到它，不用再选库文件
        await host.ViewModel.GenerateSyntheticProjectAsync(
            new SyntheticProjectDraft("训练用模拟数据", 3, 64, 48, 1, 2, 0.0));

        Assert.Equal(host.ViewModel.DatabasePath, host.ViewModel.Train.LibraryFile);

        IReadOnlyList<TinyLabelProjectSummary> projects = await host.ViewModel.Train.ListLibraryProjectsAsync();

        Assert.Contains(projects, project => project.Name == "训练用模拟数据");
    }
}

/// <summary>只记录调用，用来确认渲染上下文把批量线段拆成了单条线。</summary>
internal sealed class LineRecordingRenderContext : IRenderContext
{
    public List<int> SegmentBatches { get; } = [];

    public int LineCount { get; private set; }

    public bool RendersToScreen => false;

    public int ClipCount => 0;

    public void DrawLineSegments(
        IList<ScreenPoint> points,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
        => SegmentBatches.Add(points.Count);

    public void DrawLine(
        IList<ScreenPoint> points,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
        => LineCount++;

    public void DrawEllipse(OxyRect extents, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
    {
    }

    public void DrawEllipses(IList<OxyRect> extents, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
    {
    }

    public void DrawPolygon(
        IList<ScreenPoint> points,
        OxyColor fill,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
    {
    }

    public void DrawPolygons(
        IList<IList<ScreenPoint>> polygons,
        OxyColor fill,
        OxyColor stroke,
        double thickness,
        EdgeRenderingMode edgeRenderingMode,
        double[]? dashArray = null,
        LineJoin lineJoin = LineJoin.Miter)
    {
    }

    public void DrawRectangle(OxyRect rectangle, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
    {
    }

    public void DrawRectangles(IList<OxyRect> rectangles, OxyColor fill, OxyColor stroke, double thickness, EdgeRenderingMode edgeRenderingMode)
    {
    }

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
    {
    }

    public OxySize MeasureText(string text, string? fontFamily = null, double fontSize = 10, double fontWeight = 500)
        => OxySize.Empty;

    public void SetToolTip(string text)
    {
    }

    public void CleanUp()
    {
    }

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
    {
    }

    public void PushClip(OxyRect clippingRectangle)
    {
    }

    public void PopClip()
    {
    }
}
