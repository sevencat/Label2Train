using System.ComponentModel;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;
using Label2Train.App.ViewModels;
using Label2Train.Core.Infrastructure.Logging;

namespace Label2Train.App.Views;

/// <summary>运行日志：右上角「日志」按钮弹出，跟着 <see cref="MainWindowViewModel.LogText"/> 实时刷新。</summary>
public partial class LogWindow : Window
{
    private readonly MainWindowViewModel? _viewModel;

    /// <summary>XAML 预览用。</summary>
    public LogWindow() => InitializeComponent();

    public LogWindow(MainWindowViewModel viewModel, LogBuffer buffer) : this()
    {
        _viewModel = viewModel;
        DataContext = viewModel;

        Opened += (_, _) =>
        {
            // 打开时先把窗口关着期间攒下的行补上，再开始逐条追加
            viewModel.RefreshLog();
            buffer.EntryAdded += OnEntryAdded;

            LogScroll.ScrollToEnd();
            viewModel.PropertyChanged += OnViewModelChanged;
        };

        Closed += (_, _) =>
        {
            buffer.EntryAdded -= OnEntryAdded;
            viewModel.PropertyChanged -= OnViewModelChanged;
        };
    }

    /// <summary>NLog 在后台线程写缓冲，这里切回 UI 线程再改绑定的文本。</summary>
    private void OnEntryAdded(LogEntry entry)
    {
        if (_viewModel is null)
        {
            return;
        }

        Dispatcher.UIThread.Post(() => _viewModel.AppendLog(entry));
    }

    private void OnViewModelChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(MainWindowViewModel.LogText))
        {
            LogScroll.ScrollToEnd();
        }
    }

    private void OnClear(object? sender, RoutedEventArgs e) => _viewModel?.ClearLog();

    private void OnClose(object? sender, RoutedEventArgs e) => Close();
}
