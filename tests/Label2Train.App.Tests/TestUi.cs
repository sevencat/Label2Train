using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Headless;
using Avalonia.Input;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Label2Train.App.Views;

namespace Label2Train.App.Tests;

/// <summary>headless 下的界面操作小工具。</summary>
internal static class TestUi
{
    /// <summary>显示窗口并把排队的布局 / 绑定任务跑完。</summary>
    public static void Show(Window window)
    {
        window.Show();
        Dispatcher.UIThread.RunJobs();
    }

    /// <summary>切到训练页：训练页在显示前不会 apply template，内部控件只有这时候才建出来。</summary>
    public static void SwitchToTraining(Window window)
        => Click(window, Tab(window, "TrainTab"));

    /// <summary>headless 鼠标点一下控件中心（走真实的命中测试与事件路由）。</summary>
    public static void Click(Window window, Control target)
    {
        Dispatcher.UIThread.RunJobs();

        Point center = target.TranslatePoint(
            new Point(target.Bounds.Width / 2, target.Bounds.Height / 2), window)
            ?? throw new InvalidOperationException($"控件 {target.Name} 不在可视树里");

        window.MouseDown(center, MouseButton.Left);
        window.MouseUp(center, MouseButton.Left);
        Dispatcher.UIThread.RunJobs();
    }

    public static ToggleButton Tab(Window window, string name)
        => window.FindControl<ToggleButton>(name)
            ?? throw new InvalidOperationException($"找不到 {name}");

    public static Control Named(Window window, string name)
        => window.FindControl<Control>(name)
            ?? throw new InvalidOperationException($"找不到 {name}");

    public static AnnotationView Annotation(Window window)
        => window.GetVisualDescendants().OfType<AnnotationView>().Single();

    public static TrainView Train(Window window)
        => window.GetVisualDescendants().OfType<TrainView>().Single();
}
