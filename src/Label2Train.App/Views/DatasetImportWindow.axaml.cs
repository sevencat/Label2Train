using Avalonia.Controls;
using Avalonia.Interactivity;
using Label2Train.Core.Dataset;
using Label2Train.Core.Dataset.Import;
using Label2Train.Core.Localization;

namespace Label2Train.App.Views;

/// <summary>
/// 数据集目录校验通过后确认项目名：目录已经选好、形态已经认出，
/// 这里只让用户确认（或改）项目名，取消返回 null。
/// </summary>
public partial class DatasetImportWindow : Window
{
    public DatasetImportWindow()
    {
        InitializeComponent();
    }

    public DatasetImportWindow(DatasetFormat format, string sourcePath, DatasetProbe probe)
        : this()
    {
        ProbeText.Text = Loc.Format("Import.Verified", format.ToDisplayName(), probe.Message);
        SourceText.Text = Loc.Format("Import.SourcePath", sourcePath);
        NameBox.Text = SuggestedName(sourcePath);
    }

    private static Localizer Loc => Localizer.Instance;

    private void OnConfirm(object? sender, RoutedEventArgs e)
    {
        string name = NameBox.Text?.Trim() ?? string.Empty;

        if (name.Length == 0)
        {
            ErrorText.Text = Loc.Get("Import.ProjectNameRequired");
            NameBox.Focus();
            return;
        }

        Close(name);
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);

    /// <summary>目录名当默认项目名（根目录这种取不到名字的就留空，让用户自己填）。</summary>
    private static string SuggestedName(string sourcePath)
    {
        try
        {
            return Path.GetFileName(Path.TrimEndingDirectorySeparator(sourcePath));
        }
        catch (ArgumentException)
        {
            return string.Empty;
        }
    }
}
