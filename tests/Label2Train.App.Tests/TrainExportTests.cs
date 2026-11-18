using System.Globalization;
using Avalonia.Headless.XUnit;
using Label2Train.App.ViewModels;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Storage;
using Label2Train.Train.Export;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training;
using TorchSharp;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 训练产物导出：标注项目训练出来的模型要落进项目目录的 <c>models/</c>（跟着项目走），
/// 外部目录数据集才退回输出目录的 <c>export/</c>；训练结束会自动导一次。
/// </summary>
public class TrainExportTests
{
    private const string Architecture = "yolov8n-detect-2";

    [AvaloniaFact]
    public async Task Export_Lands_In_Project_Models_Directory()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, long projectId) = await OpenSyntheticProjectAsync(host);

        train.OutputDirectory = SeedRun(host.Root);

        string expected = TinyLabelWorkspace.ModelDirectory(StorageLocation.FromPath(host.Root), projectId);

        Assert.Equal(expected, train.ExportDirectory);
        Assert.True(await train.ExportAsync(withOnnx: true));

        Assert.True(File.Exists(Path.Combine(expected, TrainingExportService.WeightsFileName)));
        Assert.True(File.Exists(Path.Combine(expected, TrainingExportService.ConfigFileName)));
        Assert.True(File.Exists(Path.Combine(expected, TrainingExportService.OnnxFileName)));

        // 导出的 ONNX 立刻成为检测用的模型
        Assert.Equal(Path.Combine(expected, TrainingExportService.OnnxFileName), train.LastOnnxPath);
    }

    [AvaloniaFact]
    public async Task Training_Finishes_With_Model_In_Project_Directory()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, _) = await OpenSyntheticProjectAsync(host);

        train.OutputDirectory = SeedRun(host.Root);

        await train.StartTrainingAsync();

        string onnx = Path.Combine(train.ExportDirectory, TrainingExportService.OnnxFileName);

        Assert.True(File.Exists(onnx), $"训练结束应自动导出到 {onnx}");
        Assert.Contains(TinyLabelWorkspace.ModelsFolderName, train.ExportDirectory, StringComparison.Ordinal);
    }

    [AvaloniaFact]
    public async Task External_Dataset_Falls_Back_To_Output_Export()
    {
        await using AppTestHost host = AppTestHost.Create();
        TrainViewModel train = host.ViewModel.Train;

        Assert.Equal(Path.Combine(train.OutputDirectory, "export"), train.ExportDirectory);
    }

    /// <summary>造一个「已经训练过」的输出目录：有元数据齐全的 best.safetensors 才有东西可导。</summary>
    private static string SeedRun(string root)
    {
        string run = Path.Combine(root, "runs");
        Directory.CreateDirectory(run);

        using YoloDetectionModel model = YoloArchitectures.CreateFromArchitecture(torch.CPU, Architecture);

        Checkpoint.Save(
            Path.Combine(run, TrainingExportService.WeightsFileName),
            model,
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["architecture"] = Architecture,
                ["classes"] = "2",
                ["image_size"] = "640",
            });

        return run;
    }

    private static async Task<(TrainViewModel Train, long ProjectId)> OpenSyntheticProjectAsync(AppTestHost host)
    {
        await host.ViewModel.GenerateSyntheticProjectAsync(
            new SyntheticProjectDraft("导出用模拟数据", 3, 64, 48, 1, 2, 0.0));

        TrainViewModel train = host.ViewModel.Train;
        IReadOnlyList<TinyLabelProjectSummary> projects = await train.ListLibraryProjectsAsync();
        TinyLabelProjectSummary project = Assert.Single(projects);

        await train.OpenLibraryProjectAsync(
            project.Id,
            $"{project.Name} · #{project.Id.ToString(CultureInfo.InvariantCulture)}");

        return (train, project.Id);
    }
}
