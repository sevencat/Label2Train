using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
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

    /// <summary>
    /// 训练目标入口只剩一个「训练目标设置…」：项目、模型、规模、参数都在对话框里配。
    /// 页面上不该再有 YOLO / RF-DETR 数据集按钮，也不该再有检测模型单选。
    /// </summary>
    [AvaloniaFact]
    public async Task Target_Setup_Replaces_Dataset_Shortcuts()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);

        List<Button> buttons = [.. view.GetVisualDescendants().OfType<Button>()];

        Assert.DoesNotContain(buttons, button => button.Content is "YOLO" or "RF-DETR");
        Assert.Contains(
            buttons,
            button => button.Content as string == Localizer.Instance.Get("Train.TargetSetup"));

        // 检测只用当前训练配置的模型，页面上不再有模型单选
        Assert.DoesNotContain(
            view.GetVisualDescendants().OfType<RadioButton>(),
            radio => radio.Content is "YOLOv8" or "YOLOv11" or "RF-DETR");
    }

    /// <summary>run 目录跟着库走（&lt;库根&gt;/runs），不再读「上次数据集目录」。</summary>
    [AvaloniaFact]
    public async Task Output_Directory_Defaults_To_Library_Runs()
    {
        await using AppTestHost host = AppTestHost.Create();

        string output = host.ViewModel.Train.OutputDirectory;

        Assert.StartsWith(host.Root, output, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("runs", Path.GetFileName(output));
    }

    /// <summary>模型与规模现在是只读文本：训练页一个下拉都没有，值跟着训练目标设置变。</summary>
    [AvaloniaFact]
    public async Task Model_And_Scale_Are_Read_Only()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);
        TrainViewModel train = host.ViewModel.Train;

        Assert.Empty(view.GetVisualDescendants().OfType<ComboBox>());

        Assert.Equal(train.SelectedModel!.Name, train.ModelNameText);
        Assert.Equal(train.SelectedScale!.Name, train.ScaleText);

        // RF-DETR 不看规模，只读显示成「不适用」
        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "rfdetr");
        Assert.Equal(Localizer.Instance.Get("Train.ScaleNA"), train.ScaleText);
    }

    /// <summary>训练目标设置对话框：项目 / 模型 / 规模三个下拉都在，RF-DETR 时规模置灰。</summary>
    [AvaloniaFact]
    public void Target_Dialog_Has_Project_Model_Scale()
    {
        var dialog = new TrainingTargetDialog(
            [new TinyLabelProjectSummary(1, "demo", "", 10, 8, 2, DateTime.UnixEpoch, null, [])],
            1,
            [new ModelOption("v8", "YOLOv8"), new ModelOption("rfdetr", "RF-DETR")],
            [new ScaleOption("n", "n · nano"), new ScaleOption("s", "s · small")],
            "v8",
            "n",
            100m,
            4m,
            640m);

        ComboBox projectBox = Assert.IsType<ComboBox>(dialog.FindControl<Control>("ProjectBox"));
        ComboBox modelBox = Assert.IsType<ComboBox>(dialog.FindControl<Control>("ModelBox"));
        ComboBox scaleBox = Assert.IsType<ComboBox>(dialog.FindControl<Control>("ScaleBox"));
        CheckBox augmentBox = Assert.IsType<CheckBox>(dialog.FindControl<Control>("AugmentBox"));

        Assert.Equal(1, projectBox.ItemCount);
        Assert.True(scaleBox.IsEnabled);

        // 增强开关只有 RF-DETR 认：YOLO 的 hflip / mosaic 在它自己的 pipeline 里
        Assert.False(augmentBox.IsEnabled);

        modelBox.SelectedIndex = 1;
        Assert.False(scaleBox.IsEnabled);
        Assert.True(augmentBox.IsEnabled);
        Assert.True(augmentBox.IsChecked);
    }

    /// <summary>
    /// 增上一轮实测里「同种子关增强更准」的结论要能让用户自己复现：开关落在训练目标设置里，
    /// 值一路传到 run options；主界面只在 RF-DETR 下显示 aug 开 / 关。
    /// </summary>
    [AvaloniaFact]
    public async Task Augment_Toggle_Is_RfDetr_Only()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);
        TrainViewModel train = host.ViewModel.Train;

        Assert.False(train.ShowAugment);
        Assert.DoesNotContain(
            view.GetVisualDescendants().OfType<TextBlock>(),
            block => block.Text == "aug" && block.IsVisible);

        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "rfdetr");
        Dispatcher.UIThread.RunJobs();

        Assert.True(train.ShowAugment);
        Assert.Equal(Localizer.Instance.Get("Train.AugmentOn"), train.AugmentText);

        HashSet<string> texts = [.. view
            .GetVisualDescendants()
            .OfType<TextBlock>()
            .Where(block => block.IsVisible)
            .Select(block => block.Text ?? string.Empty)];

        Assert.Contains("aug", texts);
        Assert.Contains(Localizer.Instance.Get("Train.AugmentOn"), texts);

        train.Augment = false;
        Dispatcher.UIThread.RunJobs();

        Assert.Equal(Localizer.Instance.Get("Train.AugmentOff"), train.AugmentText);
    }

    /// <summary>
    /// 检测只认当前训练配置的模型：<c>models/{模型名}_{规模}.onnx</c>，换规模就换文件，
    /// 不拿上一档的权重顶上去（结构不同，框会整体错）。
    /// </summary>
    [AvaloniaFact]
    public async Task Detect_Uses_Current_Training_Config()
    {
        await using AppTestHost host = AppTestHost.Create();

        await host.ViewModel.GenerateSyntheticProjectAsync(
            new SyntheticProjectDraft("检测用", 3, 64, 48, 1, 2, 0.0));

        TrainViewModel train = host.ViewModel.Train;
        IReadOnlyList<TinyLabelProjectSummary> projects = await train.ListLibraryProjectsAsync();
        await train.OpenLibraryProjectAsync(projects[0].Id, projects[0].Name);

        string directory = train.ExportDirectory;
        Directory.CreateDirectory(directory);

        File.WriteAllBytes(Path.Combine(directory, "yolov8_n.onnx"), [0]);
        Assert.Equal(Path.Combine(directory, "yolov8_n.onnx"), train.ResolveDetectModelPath());

        train.SelectedScale = train.ModelScales.Single(scale => scale.Id == "s");
        Assert.Equal(string.Empty, train.ResolveDetectModelPath());

        File.WriteAllBytes(Path.Combine(directory, "yolov8_s.onnx"), [0]);
        Assert.Equal(Path.Combine(directory, "yolov8_s.onnx"), train.ResolveDetectModelPath());
    }

    /// <summary>没有打开数据集时「开始训练」必须是灰的：按钮的 IsEnabled 绑定也要真生效，而不只是 VM 属性为 false。</summary>
    [AvaloniaFact]
    public async Task Start_Button_Is_Disabled_Without_Dataset()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);
        string start = Localizer.Instance.Get("Train.Start");

        Button button = view.GetVisualDescendants()
            .OfType<Button>()
            .Single(item => item.Content as string == start);

        Assert.False(button.IsEnabled);
    }

    [AvaloniaFact]
    public async Task Detr_Pretrained_Weights_Are_Shown()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        TrainView view = TestUi.Train(window);
        TrainViewModel train = host.ViewModel.Train;

        // YOLO 家族不用预训练权重，这一行不该出现
        Assert.False(train.IsDetrFamily);
        Assert.False(train.ShowPretrainedWeightsMissing);

        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "rfdetr");
        Dispatcher.UIThread.RunJobs();

        Assert.True(train.IsDetrFamily);

        // 找到权重显示文件名、路径进 ToolTip；没找到明说随机初始化——两种都不留空
        Assert.False(string.IsNullOrWhiteSpace(train.PretrainedWeights));
        Assert.Equal(
            train.HasPretrainedWeights
                ? Path.GetFileName(train.PretrainedWeightsPath)
                : Localizer.Instance.Get("Train.PretrainedMissing"),
            train.PretrainedWeights);

        HashSet<string> texts = [.. view
            .GetVisualDescendants()
            .OfType<TextBlock>()
            .Select(block => block.Text ?? string.Empty)];

        Assert.Contains(Localizer.Instance.Get("Train.Pretrained"), texts);
        Assert.Contains(train.PretrainedWeights, texts);
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
