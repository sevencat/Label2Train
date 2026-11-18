using Avalonia.Controls;
using Avalonia.Interactivity;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Localization;

namespace Label2Train.App.Views;

/// <summary>
/// 从标注库里挑一个项目来训练：库固定是标注端正在用的那个，只能选项目，不能换库。
/// </summary>
public partial class ProjectPickerDialog : Window
{
    private readonly IReadOnlyList<TinyLabelProjectSummary> _projects;

    public ProjectPickerDialog()
        : this([], string.Empty, 0)
    {
    }

    public ProjectPickerDialog(
        IReadOnlyList<TinyLabelProjectSummary> projects,
        string databaseFile,
        long lastProjectId)
    {
        InitializeComponent();

        _projects = projects;

        LibraryText.Text = Localizer.Instance.Format("Train.LibraryFile", databaseFile);

        ProjectList.ItemsSource = projects
            .Select(project => Localizer.Instance.Format(
                "Train.ProjectRow", project.Name, project.Id, project.ImageCount, project.LabeledCount))
            .ToList();

        // 默认选中上次训练的那个项目；找不到就选第一个
        int index = projects.ToList().FindIndex(project => project.Id == lastProjectId);
        ProjectList.SelectedIndex = index >= 0 ? index : 0;
    }

    private void OnOk(object? sender, RoutedEventArgs e)
    {
        if (Selected is { } project)
        {
            Close(project);
        }
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);

    private TinyLabelProjectSummary? Selected
    {
        get
        {
            int index = ProjectList.SelectedIndex;
            return index >= 0 && index < _projects.Count ? _projects[index] : null;
        }
    }
}
