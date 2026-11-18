using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Label2Train.App.Views;
using Label2Train.Core.Localization;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>主窗口外壳：模式切换、自绘标题栏、两个页面互斥显示。</summary>
public class MainWindowTests
{
    [AvaloniaFact]
    public async Task Starts_On_Annotation_Page()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);

        AnnotationView annotation = TestUi.Annotation(window);
        TrainView train = TestUi.Train(window);

        Assert.True(annotation.IsVisible);
        Assert.False(train.IsVisible);
        Assert.True(TestUi.Tab(window, "AnnotateTab").IsChecked);
        Assert.False(TestUi.Tab(window, "TrainTab").IsChecked);
    }

    [AvaloniaFact]
    public async Task Switching_To_Training_Shows_Only_Train_Page()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);
        TestUi.SwitchToTraining(window);

        AnnotationView annotation = TestUi.Annotation(window);
        TrainView train = TestUi.Train(window);

        Assert.True(host.ViewModel.IsTrainingMode);
        Assert.False(annotation.IsVisible);
        Assert.True(train.IsVisible);

        // 训练页自己的 DataContext 是 Train 子 VM，标注页还是主 VM（两套上下文不能串）
        Assert.Same(host.ViewModel.Train, train.DataContext);
        Assert.Same(host.ViewModel, annotation.DataContext);

        TestUi.Click(window, TestUi.Tab(window, "AnnotateTab"));

        Assert.False(host.ViewModel.IsTrainingMode);
        Assert.True(annotation.IsVisible);
        Assert.False(train.IsVisible);
    }

    [AvaloniaFact]
    public async Task Custom_Title_Bar_Has_Caption_Buttons()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);

        List<Button> captionButtons = [.. window
            .GetVisualDescendants()
            .OfType<Button>()
            .Where(button => button.Classes.Contains("caption"))];

        Assert.Equal(3, captionButtons.Count);
        Assert.True(TestUi.Named(window, "MaximizeIcon").IsVisible);
        Assert.False(TestUi.Named(window, "RestoreIcon").IsVisible);
    }

    [AvaloniaFact]
    public async Task Maximize_Button_Toggles_Window_State_And_Icon()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);

        Button maximize = window
            .GetVisualDescendants()
            .OfType<Button>()
            .Single(button => button.Classes.Contains("caption")
                && ToolTip.GetTip(button) is string tip
                && tip.Contains("最大化", StringComparison.Ordinal));

        TestUi.Click(window, maximize);

        Assert.Equal(WindowState.Maximized, window.WindowState);
        Assert.False(TestUi.Named(window, "MaximizeIcon").IsVisible);
        Assert.True(TestUi.Named(window, "RestoreIcon").IsVisible);

        TestUi.Click(window, maximize);

        Assert.Equal(WindowState.Normal, window.WindowState);
        Assert.True(TestUi.Named(window, "MaximizeIcon").IsVisible);
        Assert.False(TestUi.Named(window, "RestoreIcon").IsVisible);
    }

    [AvaloniaFact]
    public async Task Minimize_And_Close_Buttons_Work()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);

        TestUi.Click(window, Caption(window, "Common.Minimize"));
        Assert.Equal(WindowState.Minimized, window.WindowState);

        window.WindowState = WindowState.Normal;
        Dispatcher.UIThread.RunJobs();

        bool closed = false;
        window.Closed += (_, _) => closed = true;

        TestUi.Click(window, Caption(window, "Common.Close"));
        Assert.True(closed);
    }

    private static Button Caption(Window window, string toolTipKey)
    {
        string tip = Localizer.Instance.Get(toolTipKey);

        return window
            .GetVisualDescendants()
            .OfType<Button>()
            .Single(button => button.Classes.Contains("caption")
                && ToolTip.GetTip(button) as string == tip);
    }
}
