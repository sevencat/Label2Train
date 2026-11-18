using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using FreeSql;
using Label2Train.App.ViewModels;
using Label2Train.App.Views;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Localization;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>「生成模拟数据」：建项目、图片进项目目录、标注进数据库。</summary>
public class SyntheticProjectTests
{
    private static Localizer Loc => Localizer.Instance;

    [AvaloniaFact]
    public async Task Generating_Synthetic_Project_Fills_Database_And_Project_Folder()
    {
        await using AppTestHost host = AppTestHost.Create();

        await host.ViewModel.GenerateSyntheticProjectAsync(
            new SyntheticProjectDraft(
                Name: "模拟项目",
                ImageCount: 6,
                ImageWidth: 96,
                ImageHeight: 72,
                MinShapes: 1,
                MaxShapes: 2,
                ValRatio: 0.34));

        // 列表里出现且被选中
        ProjectSummaryViewModel project = Assert.Single(host.ViewModel.VisibleProjects);

        Assert.Equal("模拟项目", project.Name);
        Assert.Equal(6, project.ImageCount);
        Assert.Equal(6, project.LabeledCount);
        Assert.Equal(project.Id, host.ViewModel.SelectedProject?.Id);

        // 图片按 projects/{id}/... 落盘，且没有写到别处
        Assert.Equal(6, Directory.GetFiles(host.Root, "*.png", SearchOption.AllDirectories).Length);

        // 数据库：6 条图片（全部已标注）、6 份标注
        IFreeSql orm = host.Services.GetRequiredService<IFreeSql>();

        List<ImageEntity> images = await orm.Select<ImageEntity>()
            .Where(item => item.ProjectId == project.Id)
            .ToListAsync();

        Assert.Equal(6, images.Count);
        Assert.All(images, image => Assert.Equal("labeled", image.State));
        Assert.All(images, image => Assert.True(File.Exists(image.FilePath), $"{image.FilePath} 不存在"));

        Assert.Equal(6, await orm.Select<AnnotationEntity>().Where(item => item.ProjectId == project.Id).CountAsync());
    }

    [AvaloniaFact]
    public void Dialog_Prefills_Name_With_Timestamp()
    {
        var window = new SyntheticProjectWindow();
        string name = window.FindControl<TextBox>("NameBox")!.Text ?? string.Empty;

        Assert.StartsWith($"{Loc.Get("Synthetic.DefaultName")} ", name, StringComparison.Ordinal);
        Assert.Matches(@"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", name);
    }

    [AvaloniaFact]
    public async Task Generating_Synthetic_Project_With_Zero_Images_Fails_Softly()
    {
        await using AppTestHost host = AppTestHost.Create();

        await host.ViewModel.GenerateSyntheticProjectAsync(
            new SyntheticProjectDraft("空项目", 0, 96, 72, 1, 1, 0.2));

        Assert.Empty(host.ViewModel.VisibleProjects);
        Assert.StartsWith(Loc.Get("Label.SyntheticFailed").Split('{')[0], host.ViewModel.Status, StringComparison.Ordinal);
        Assert.False(host.ViewModel.IsBusy);
    }
}
