using System.Globalization;
using System.Text;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Headless.XUnit;
using Avalonia.VisualTree;
using Label2Train.App.ViewModels;
using Label2Train.App.Views;
using Label2Train.Core.Annotation;
using Label2Train.Core.Dataset;
using Label2Train.Core.Dataset.Import;
using Label2Train.Core.Localization;
using Label2Train.Core.Simulation;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>从数据集建项目：项目管理处的下拉入口、目录校验、确认后的导入。</summary>
public class DatasetImportFlowTests
{
    private static Localizer Loc => Localizer.Instance;

    [AvaloniaFact]
    public async Task Project_Header_Has_Dropdown_With_Import_And_Synthetic()
    {
        await using AppTestHost host = AppTestHost.Create();
        MainWindow window = host.CreateMainWindow();

        TestUi.Show(window);

        // 项目卡片右上角的下拉：导入两种数据集 + 一条分隔线 + 生成模拟数据
        Button dropdown = window
            .GetVisualDescendants()
            .OfType<Button>()
            .Single(button => ToolTip.GetTip(button) as string == Loc.Get("Label.ImportMenu"));

        MenuFlyout flyout = Assert.IsType<MenuFlyout>(dropdown.Flyout);
        List<MenuItem> items = [.. flyout.Items.OfType<MenuItem>()];

        Assert.Equal(3, items.Count);
        Assert.Equal(Loc.Get("Label.ImportYolo"), items[0].Header as string);
        Assert.Equal(Loc.Get("Label.ImportRfDetr"), items[1].Header as string);
        Assert.Equal(Loc.Get("Label.GenerateSynthetic"), items[2].Header as string);
        Assert.Contains(flyout.Items, item => item is Separator);
    }

    [AvaloniaFact]
    public async Task CheckImport_Accepts_Matching_Format_And_Rejects_Other()
    {
        await using AppTestHost host = AppTestHost.Create();
        string dataset = Path.Combine(host.Root, "yolo-source");
        WriteYoloDataset(dataset);

        DatasetImportCheck yolo = await host.ViewModel.CheckImportAsync(dataset, DatasetFormat.Yolo);

        Assert.True(yolo.IsValid);
        Assert.Equal(DatasetFormat.Yolo, yolo.Probe?.Format);

        // 同一个 YOLO 目录按 RF-DETR（COCO）导入要被拦下来
        DatasetImportCheck coco = await host.ViewModel.CheckImportAsync(dataset, DatasetFormat.Coco);

        Assert.False(coco.IsValid);
        Assert.Contains(Loc.Get("Dataset.Format.Coco"), coco.Message, StringComparison.Ordinal);
    }

    [AvaloniaFact]
    public async Task CheckImport_Rejects_Empty_Folder()
    {
        await using AppTestHost host = AppTestHost.Create();
        string empty = Path.Combine(host.Root, "empty");
        Directory.CreateDirectory(empty);

        DatasetImportCheck check = await host.ViewModel.CheckImportAsync(empty, DatasetFormat.Yolo);

        Assert.False(check.IsValid);
        Assert.NotEmpty(check.Message);
    }

    [AvaloniaFact]
    public async Task Confirming_Import_Creates_Project_With_Images_And_Annotations()
    {
        await using AppTestHost host = AppTestHost.Create();
        string dataset = Path.Combine(host.Root, "yolo-source");
        int boxes = WriteYoloDataset(dataset);

        // 对话框点了「导入」之后走的就是这一步
        await host.ViewModel.ImportAsync(new ImportRequest(dataset, "导入的项目", DatasetFormat.Yolo));

        ProjectSummaryViewModel project = Assert.Single(host.ViewModel.VisibleProjects);

        Assert.Equal("导入的项目", project.Name);
        Assert.Equal(4, project.ImageCount);
        Assert.Equal(4, project.LabeledCount);
        Assert.True(boxes >= 4, $"合成数据集应至少产生 4 个框，实际 {boxes}");
    }

    /// <summary>造一个 4 张 96x72 的 YOLO 目录（2 train / 2 val，每张都有标注）。</summary>
    private static int WriteYoloDataset(string root)
    {
        var generator = new SyntheticImageGenerator(new SyntheticOptions(4, 96, 72, 1, 2, 20261118));
        IReadOnlyList<SyntheticImage> images = generator.Generate();

        Directory.CreateDirectory(root);
        File.WriteAllText(
            Path.Combine(root, "data.yaml"),
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: square\n  1: triangle\n  2: circle\n");

        int boxes = 0;

        for (int index = 0; index < images.Count; index++)
        {
            SyntheticImage image = images[index];
            string split = index < 2 ? "train" : "val";

            WriteFile(Path.Combine(root, "images", split, image.FileName), image.Png);

            var labels = new StringBuilder();

            foreach (SyntheticShape shape in image.Shapes)
            {
                (double cx, double cy, double width, double height) =
                    BoxConvert.ToYolo(shape.Bounds, image.Width, image.Height);

                labels.Append(ClassId(shape.Label).ToString(CultureInfo.InvariantCulture)).Append(' ')
                    .Append(cx.ToString("F6", CultureInfo.InvariantCulture)).Append(' ')
                    .Append(cy.ToString("F6", CultureInfo.InvariantCulture)).Append(' ')
                    .Append(width.ToString("F6", CultureInfo.InvariantCulture)).Append(' ')
                    .Append(height.ToString("F6", CultureInfo.InvariantCulture)).Append('\n');

                boxes++;
            }

            WriteFile(
                Path.Combine(root, "labels", split, Path.GetFileNameWithoutExtension(image.FileName) + ".txt"),
                Encoding.UTF8.GetBytes(labels.ToString()));
        }

        return boxes;
    }

    private static int ClassId(string label)
    {
        for (int index = 0; index < SyntheticImageGenerator.ClassNames.Count; index++)
        {
            if (SyntheticImageGenerator.ClassNames[index] == label)
            {
                return index;
            }
        }

        return 0;
    }

    private static void WriteFile(string path, byte[] content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, content);
    }
}
