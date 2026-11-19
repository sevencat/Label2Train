using System.Globalization;
using Avalonia.Headless.XUnit;
using Microsoft.Extensions.DependencyInjection;
using Label2Train.App.ViewModels;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure;
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

    /// <summary>导出文件名前缀：按架构自动取，yolov8* → <c>yolov8</c>，不再叫 best。</summary>
    private const string ExportPrefix = "yolov8";

    [AvaloniaFact]
    public async Task Export_Lands_In_Project_Models_Directory()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, long projectId) = await OpenSyntheticProjectAsync(host);

        train.OutputDirectory = SeedRun(host.Root);

        string expected = TinyLabelWorkspace.ModelDirectory(StorageLocation.FromPath(host.Root), projectId);

        Assert.Equal(expected, train.ExportDirectory);
        Assert.True(await train.ExportAsync(withOnnx: true));

        Assert.True(File.Exists(Path.Combine(expected, ExportPrefix + ".safetensors")));
        Assert.True(File.Exists(Path.Combine(expected, ExportPrefix + ".json")));
        Assert.True(File.Exists(Path.Combine(expected, ExportPrefix + ".onnx")));

        // 导出的 ONNX 立刻成为检测用的模型，检测单选也跟着切到对应家族
        Assert.Equal(Path.Combine(expected, ExportPrefix + ".onnx"), train.LastOnnxPath);
        Assert.Equal(ExportPrefix, train.SelectedDetectModel.FilePrefix);
    }

    [AvaloniaFact]
    public async Task Training_Finishes_With_Model_In_Project_Directory()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, _) = await OpenSyntheticProjectAsync(host);

        train.OutputDirectory = SeedRun(host.Root);

        await train.StartTrainingAsync();

        string onnx = Path.Combine(train.ExportDirectory, ExportPrefix + ".onnx");

        Assert.True(File.Exists(onnx), $"训练结束应自动导出到 {onnx}");
        Assert.Contains(TinyLabelWorkspace.ModelsFolderName, train.ExportDirectory, StringComparison.Ordinal);
    }

    /// <summary>导出文件名按架构取，不再统一叫 best——三个模型要能同时在 models 目录里活着。</summary>
    [Theory]
    [InlineData("yolov8n-detect-2", "yolov8")]
    [InlineData("yolo11n-detect-2", "yolov11")]
    [InlineData("rfdetr-detect-2", "rfdetr")]
    [InlineData("simple-detr-detect-2", "simple-detr")]
    public void Export_Prefix_Follows_Architecture(string architecture, string expected)
    {
        Assert.Equal(expected, TrainingExportService.PrefixFor(architecture));
    }

    /// <summary>没选项目或没选输出路径时「开始训练」必须是灰的。</summary>
    [AvaloniaFact]
    public async Task Start_Is_Grey_Until_Dataset_And_Output_Are_Set()
    {
        await using AppTestHost host = AppTestHost.Create();
        TrainViewModel train = host.ViewModel.Train;

        Assert.False(train.CanStart);

        await OpenSyntheticProjectAsync(host);
        Assert.True(train.CanStart);

        train.OutputDirectory = "   ";
        Assert.False(train.CanStart);
    }

    /// <summary>
    /// RF-DETR 的 onnx 要同时过 patch 14 与下采样 16，边长得是 112 的倍数；
    /// 训完才发现导不出来太亏，所以参数不合法时直接不让开始。
    /// </summary>
    [AvaloniaFact]
    public async Task Start_Is_Grey_When_RfDetr_ImageSize_Cannot_Export()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, _) = await OpenSyntheticProjectAsync(host);

        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "rfdetr");

        Assert.False(train.IsImageSizeAllowed);
        Assert.False(train.CanStart);

        train.ImageSize = TrainingPresets.RfDetr.ImageSize;

        Assert.True(train.IsImageSizeAllowed);
        Assert.True(train.CanStart);

        // YOLO 家族不受这条约束
        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "v8");
        train.ImageSize = 640;

        Assert.True(train.CanStart);
    }

    [Theory]
    [InlineData(336, true)]
    [InlineData(448, true)]
    [InlineData(560, true)]
    [InlineData(640, false)]
    [InlineData(518, false)]
    public void Detr_ImageSize_Must_Be_Multiple_Of_112(int size, bool valid)
    {
        Assert.Equal(valid, TrainingPresets.IsValidDetrImageSize(size));
    }

    [Theory]
    [InlineData(640, 672)]
    [InlineData(518, 560)]
    [InlineData(100, 112)]
    public void Nearest_Valid_Detr_ImageSize_Is_Used_As_Suggestion(int input, int expected)
    {
        Assert.Equal(expected, TrainingPresets.NearestValidDetrImageSize(input));
    }

    /// <summary>
    /// 检测单选必须严格对应前缀：选 YOLOv8 只认 <c>yolov8.onnx</c>，没有就返回空——
    /// 拿 yolov11 / rfdetr 的权重顶上，框会整体错，DETR 还会直接崩在输出名上。
    /// </summary>
    [AvaloniaFact]
    public async Task Detect_Model_Path_Follows_Selection_Strictly()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, _) = await OpenSyntheticProjectAsync(host);

        string directory = train.ExportDirectory;
        Directory.CreateDirectory(directory);

        // 只导出了 v11
        File.WriteAllBytes(Path.Combine(directory, "yolov11.onnx"), [0]);

        train.SelectedDetectModel = train.DetectModels.Single(model => model.FilePrefix == "yolov11");
        Assert.Equal(Path.Combine(directory, "yolov11.onnx"), train.ResolveDetectModelPath());

        train.SelectedDetectModel = train.DetectModels.Single(model => model.FilePrefix == "yolov8");
        Assert.Equal(string.Empty, train.ResolveDetectModelPath());

        train.SelectedDetectModel = train.DetectModels.Single(model => model.FilePrefix == "rfdetr");
        Assert.Equal(string.Empty, train.ResolveDetectModelPath());

        // rfdetr 导出后，选 RF-DETR 才拿到它；选 YOLO 依然是空
        File.WriteAllBytes(Path.Combine(directory, "rfdetr.onnx"), [0]);
        Assert.Equal(Path.Combine(directory, "rfdetr.onnx"), train.ResolveDetectModelPath());

        train.SelectedDetectModel = train.DetectModels.Single(model => model.FilePrefix == "yolov8");
        Assert.Equal(string.Empty, train.ResolveDetectModelPath());

        // 旧版 best.onnx 只给 YOLOv8 兼容，v11 不猜
        File.Delete(Path.Combine(directory, "yolov11.onnx"));
        File.WriteAllBytes(Path.Combine(directory, TrainingExportService.OnnxFileName), [0]);

        train.SelectedDetectModel = train.DetectModels.Single(model => model.FilePrefix == "yolov11");
        Assert.Equal(string.Empty, train.ResolveDetectModelPath());

        train.SelectedDetectModel = train.DetectModels.Single(model => model.FilePrefix == "yolov8");
        Assert.Equal(
            Path.Combine(directory, TrainingExportService.OnnxFileName),
            train.ResolveDetectModelPath());
    }

    [AvaloniaFact]
    public async Task External_Dataset_Falls_Back_To_Output_Export()
    {
        await using AppTestHost host = AppTestHost.Create();
        TrainViewModel train = host.ViewModel.Train;

        Assert.Equal(Path.Combine(train.OutputDirectory, "export"), train.ExportDirectory);
    }

    /// <summary>
    /// 启动时不自动打开上次的项目：数据集得由用户点一次「选择项目」确认，
    /// 否则一进训练页「开始训练」就是亮的，看起来像「没选中也能训练」。
    /// </summary>
    [AvaloniaFact]
    public async Task Initialize_Does_Not_Auto_Open_Last_Project()
    {
        await using AppTestHost host = AppTestHost.Create();

        await host.ViewModel.GenerateSyntheticProjectAsync(
            new SyntheticProjectDraft("自动恢复用模拟数据", 3, 64, 48, 1, 2, 0.0));

        TrainViewModel train = host.ViewModel.Train;
        TinyLabelProjectSummary project = Assert.Single(await train.ListLibraryProjectsAsync());

        // 模拟「上次训练用过它」：记下 id，再走一次首次进入训练页的初始化
        host.Services.GetRequiredService<AppSettings>().LastTrainLibraryProjectId = project.Id;

        await train.InitializeAsync();

        Assert.False(train.HasDataset);
        Assert.False(train.CanStart);
        Assert.Contains(project.Name, train.Status, StringComparison.Ordinal);
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
