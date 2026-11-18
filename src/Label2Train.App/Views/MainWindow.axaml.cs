using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Injectio.Attributes;
using Label2Train.App.ViewModels;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Localization;

namespace Label2Train.App.Views;

/// <summary>
/// 主窗口：只负责外壳——标题栏、顶栏模式切换、底栏。
/// 标注页在 <see cref="AnnotationView"/>，训练页在 <see cref="TrainView"/>。
/// 注册由 Injectio 生成（容器里单例，见 <see cref="App"/> 的注册链）。
/// </summary>
[RegisterSingleton<MainWindow>]
public partial class MainWindow : Window
{
    private LogWindow? _logWindow;

    /// <summary>XAML 预览用。</summary>
    public MainWindow() => InitializeComponent();

    public MainWindow(MainWindowViewModel viewModel, StartupOptions startup) : this()
    {
        DataContext = viewModel;

        // 命令行指定了目录就打开它（--annotate 再进首图标注），否则回到上次的数据目录
        Opened += async (_, _) => await OpenStartupWorkspaceAsync(viewModel, startup);
    }

    private static Localizer Loc => Localizer.Instance;

    // ── 自绘标题栏（窗口 SystemDecorations=BorderOnly，标题栏在 XAML 里）────────

    /// <summary>标题栏拖动：按住移动窗口，双击在最大化 / 还原之间切换。</summary>
    private void OnTitleBarPressed(object? sender, PointerPressedEventArgs e)
    {
        if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
        {
            return;
        }

        if (e.ClickCount == 2)
        {
            ToggleMaximize();
            return;
        }

        BeginMoveDrag(e);
    }

    private void OnMinimizeClick(object? sender, RoutedEventArgs e) => WindowState = WindowState.Minimized;

    private void OnMaximizeClick(object? sender, RoutedEventArgs e) => ToggleMaximize();

    private void OnCloseClick(object? sender, RoutedEventArgs e) => Close();

    private void ToggleMaximize() =>
        WindowState = WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;

    /// <summary>最大化 / 还原时换图标（XAML 里两个图标叠放，用可见性切换）。</summary>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);

        if (change.Property != WindowStateProperty)
        {
            return;
        }

        bool maximized = WindowState == WindowState.Maximized;
        MaximizeIcon.IsVisible = !maximized;
        RestoreIcon.IsVisible = maximized;
    }

    private async Task OpenStartupWorkspaceAsync(MainWindowViewModel viewModel, StartupOptions startup)
    {
        if (startup.AnnotateFolder is { } annotateFolder)
        {
            await viewModel.OpenWorkspaceAsync(annotateFolder);
            await AnnotationPanel.AnnotateSelectedAsync();
        }
        else if (startup.OpenFolder is { } folder)
        {
            await viewModel.OpenWorkspaceAsync(folder);
        }
        else
        {
            await viewModel.OpenLastWorkspaceAsync();
        }
    }

    // ── 顶栏：标注 / 训练 切换 ───────────────────────────────────────────────

    /// <summary>「标注」：回到标注工作台（启动默认就在这一页）。</summary>
    private void OnShowAnnotation(object? sender, RoutedEventArgs e) => SwitchMode(training: false);

    /// <summary>「训练」：切到训练工作台；第一次进入时去列库里的标注项目。</summary>
    private async void OnShowTraining(object? sender, RoutedEventArgs e)
    {
        SwitchMode(training: true);

        if (DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.Train.InitializeAsync();
        }
    }

    /// <summary>两个 ToggleButton 的选中态手写同步：点同一个不会把已选中的那个取消掉。</summary>
    private void SwitchMode(bool training)
    {
        if (DataContext is not MainWindowViewModel viewModel)
        {
            return;
        }

        viewModel.IsTrainingMode = training;
        AnnotateTab.IsChecked = !training;
        TrainTab.IsChecked = training;
    }

    // ── 顶栏：数据目录 / 关于 / 日志 ──────────────────────────────────────────

    /// <summary>「日志」：弹独立窗口（同一次只开一个），跟着日志实时滚到底。</summary>
    private void OnOpenLog(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel viewModel)
        {
            return;
        }

        if (_logWindow is { IsVisible: true })
        {
            _logWindow.Activate();
            return;
        }

        _logWindow = new LogWindow(viewModel);
        _logWindow.Closed += (_, _) => _logWindow = null;
        _logWindow.Show(this);
    }

    /// <summary>
    /// 「打开数据目录」：在资源管理器里打开**库文件所在目录**，只是看一眼，
    /// 不选目录、不改库位置（库路径启动时就定了）。
    /// </summary>
    private void OnOpenDataFolder(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel viewModel)
        {
            return;
        }

        string folder = Path.GetDirectoryName(viewModel.DatabasePath) ?? string.Empty;

        if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
        {
            viewModel.Report(Loc.Get("Label.NoDataDir"));
            return;
        }

        if (!OpenInExplorer(folder))
        {
            viewModel.Report(Loc.Format("Label.OpenFolderFailed", folder));
        }
    }

    private async void OnAbout(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel viewModel)
        {
            return;
        }

        var dialog = new AboutWindow(viewModel.DataDirectory, viewModel.DatabasePath);
        await dialog.ShowDialog(this);
    }

    // ── 内部 ─────────────────────────────────────────────────────────────────

    private static bool OpenInExplorer(string folder)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = $"\"{folder}\"",
                UseShellExecute = true,
            });

            return true;
        }
        catch (Exception)
        {
            // 打不开资源管理器不影响程序
            return false;
        }
    }
}
