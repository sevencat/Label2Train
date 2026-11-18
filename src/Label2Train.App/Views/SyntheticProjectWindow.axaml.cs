using Avalonia.Controls;
using Avalonia.Interactivity;
using Label2Train.App.ViewModels;
using Label2Train.Core.Localization;

namespace Label2Train.App.Views;

/// <summary>
/// 「生成模拟数据」的参数：项目名、图片数量 / 尺寸、每张形状数、验证集比例。
/// 生成物直接进当前库与项目目录，不落外部数据集文件，所以这里没有输出路径。
/// </summary>
public partial class SyntheticProjectWindow : Window
{
    public SyntheticProjectWindow()
    {
        InitializeComponent();

        // 项目名带时间戳：连着生成几次也不会重名
        NameBox.Text = $"{Loc.Get("Synthetic.DefaultName")} {DateTime.Now:yyyy-MM-dd HH:mm:ss}";
    }

    private static Localizer Loc => Localizer.Instance;

    private void OnConfirm(object? sender, RoutedEventArgs e)
    {
        string name = NameBox.Text?.Trim() ?? string.Empty;

        if (name.Length == 0)
        {
            ErrorText.Text = Loc.Get("Project.NameRequired");
            NameBox.Focus();
            return;
        }

        int minShapes = (int)(MinShapesBox.Value ?? 1m);
        int maxShapes = (int)(MaxShapesBox.Value ?? 3m);

        Close(new SyntheticProjectDraft(
            name,
            (int)(CountBox.Value ?? 30m),
            (int)(WidthBox.Value ?? 640m),
            (int)(HeightBox.Value ?? 480m),
            Math.Max(1, Math.Min(minShapes, maxShapes)),
            Math.Max(1, Math.Max(minShapes, maxShapes)),
            (double)(ValRatioBox.Value ?? 20m) / 100d));
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);
}
