using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Label2Train.App.Views;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Localization;
using Label2Train.Core.Storage;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>标注工作台的结构与绑定：项目列表、图片清单表头、分页条。</summary>
public class AnnotationViewTests
{
    private static Localizer Loc => Localizer.Instance;

    [AvaloniaFact]
    public async Task Renders_Project_And_Image_Lists()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        window.Show();
        Dispatcher.UIThread.RunJobs();

        AnnotationView view = window.GetVisualDescendants().OfType<AnnotationView>().Single();

        // 左边项目列表 + 右边图片清单
        Assert.Equal(2, view.GetVisualDescendants().OfType<ListBox>().Count());
    }

    [AvaloniaFact]
    public async Task Image_List_Has_All_Column_Headers()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        window.Show();
        Dispatcher.UIThread.RunJobs();

        AnnotationView view = window.GetVisualDescendants().OfType<AnnotationView>().Single();
        HashSet<string> texts = [.. view
            .GetVisualDescendants()
            .OfType<TextBlock>()
            .Select(block => block.Text ?? string.Empty)];

        string[] columns =
        [
            "Label.ColumnThumb",
            "Label.ColumnPath",
            "Label.ColumnLabel",
            "Label.ColumnSplit",
            "Label.ColumnAnnotator",
            "Label.ColumnAnnotatedAt",
            "Label.ColumnRemark",
            "Label.ColumnActions",
        ];

        foreach (string column in columns)
        {
            Assert.Contains(Loc.Get(column), texts);
        }
    }

    [AvaloniaFact]
    public async Task Paging_Buttons_Are_Disabled_Without_Project()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        window.Show();
        Dispatcher.UIThread.RunJobs();

        AnnotationView view = window.GetVisualDescendants().OfType<AnnotationView>().Single();

        // 没选项目时分页不可点（CanPrevPage / CanNextPage 都是 false）
        Assert.False(FindByToolTip(view, "Label.PrevPage").IsEnabled);
        Assert.False(FindByToolTip(view, "Label.NextPage").IsEnabled);
    }

    [AvaloniaFact]
    public async Task Creating_Project_Shows_It_In_List_And_Summary()
    {
        await using AppTestHost host = AppTestHost.Create();

        // 直接调服务建项目，等于走完「新建项目」对话框之后那一步
        var projects = host.Services.GetRequiredService<TinyLabelProjectService>();
        await projects.CreateAsync(StorageLocation.FromPath(host.Root), "演示项目");
        await host.ViewModel.OpenWorkspaceAsync(host.Root);

        MainWindow window = host.CreateMainWindow();
        window.Show();
        Dispatcher.UIThread.RunJobs();

        AnnotationView view = window.GetVisualDescendants().OfType<AnnotationView>().Single();
        ListBox projectList = view.GetVisualDescendants().OfType<ListBox>().First();

        Assert.Single(host.ViewModel.VisibleProjects);
        Assert.Equal(1, projectList.ItemCount);

        // 概览卡按「有项目」显示，标题是项目名
        host.ViewModel.SelectedProject = host.ViewModel.VisibleProjects[0];
        await host.ViewModel.RefreshCurrentProjectAsync(null);
        Dispatcher.UIThread.RunJobs();

        Assert.True(host.ViewModel.HasProject);
        HashSet<string> texts = [.. view
            .GetVisualDescendants()
            .OfType<TextBlock>()
            .Select(block => block.Text ?? string.Empty)];

        Assert.Contains("演示项目", texts);
    }

    private static Button FindByToolTip(Control root, string key)
    {
        string tip = Loc.Get(key);

        return root
            .GetVisualDescendants()
            .OfType<Button>()
            .Single(button => ToolTip.GetTip(button) as string == tip);
    }
}
