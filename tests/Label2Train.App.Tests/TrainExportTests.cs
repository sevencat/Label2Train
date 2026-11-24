using System.Globalization;
using Avalonia.Headless.XUnit;
using Microsoft.Extensions.DependencyInjection;
using Label2Train.App.ViewModels;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Storage;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Export;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training;
using TorchSharp;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 训练产物导出：标注项目训练出来的模型要落进项目目录的 <c>models/</c>（跟着项目走），
/// 还没选项目时才退回输出目录的 <c>export/</c>；训练结束与用户停止都会自动导一次。
/// </summary>
public class TrainExportTests
{
    private const string Architecture = "yolov8n-detect-2";

    /// <summary>导出文件名：按权重里的架构自动取，yolov8n → <c>yolov8_n</c>，不再叫 best。</summary>
    private const string ExportPrefix = "yolov8_n";

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

        // 导出的 ONNX 立刻成为检测用的模型
        Assert.Equal(Path.Combine(expected, ExportPrefix + ".onnx"), train.LastOnnxPath);
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

    /// <summary>导出文件名按「架构 + 规模」取：yolov8_s / yolov11_m / rfdetr 各占一份，互不覆盖。</summary>
    [Theory]
    [InlineData("yolov8n-detect-2", "yolov8_n")]
    [InlineData("yolov8s-detect-2", "yolov8_s")]
    [InlineData("yolo11m-detect-2", "yolov11_m")]
    [InlineData("yolo26n-detect-2", "yolo26_n")]
    [InlineData("rfdetr-detect-2", "rfdetr")]
    [InlineData("simple-detr-detect-2", "simple-detr")]
    public void Export_Prefix_Follows_Architecture(string architecture, string expected)
    {
        Assert.Equal(expected, TrainingExportService.DefaultPrefixFor(architecture));
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
    /// 回归：336 是 112 的倍数（DETR 合法）却不是 32 的倍数（YOLO 合法），
    /// 预处理层曾用 YOLO 的 32 校验，导致 DETR 训练在构造加载器时就崩（界面上填 352 → 纠正成 336 → 报错）。
    /// </summary>
    [Theory]
    [InlineData(336)]
    [InlineData(560)]
    public void Detr_ImageSize_Passes_Preprocess_With_Detr_Stride(int size)
    {
        var options = new YoloPreprocessOptions
        {
            Width = size,
            Height = size,
            SizeStride = TrainingPresets.DetrImageSizeStride,
        };

        Assert.Same(options, options.Validate());
        Assert.Throws<ArgumentException>(() => new YoloPreprocessOptions
        {
            Width = size,
            Height = size,
        }.Validate());
    }

    /// <summary>
    /// 训练入口按界面 imgsz 走：合法值原样传递（以前 runner 硬编码 224，界面填 560 也不生效），
    /// 非法值就近纠正。
    /// </summary>
    [Theory]
    [InlineData(224, 224)]
    [InlineData(560, 560)]
    [InlineData(672, 672)]
    [InlineData(640, 672)]
    public void Resolve_Detr_ImageSize_Honours_Valid_Values(int input, int expected)
    {
        Assert.Equal(expected, TrainingPresets.ResolveDetrImageSize(input));
    }

    /// <summary>
    /// 输出目录 = <c>projects/{id}/models/{模型名}_{规模}</c>：切规模、切家族都跟着换，
    /// 新权重不能写进旧模型目录，否则导出会把两份产物混在一起。
    /// </summary>
    [AvaloniaFact]
    public async Task Output_Directory_Follows_Project_Model_And_Scale()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, long projectId) = await OpenSyntheticProjectAsync(host);

        string models = TinyLabelWorkspace.ModelDirectory(StorageLocation.FromPath(host.Root), projectId);

        Assert.Equal(Path.Combine(models, "yolov8_n"), train.OutputDirectory);
        Assert.Equal("yolov8_n", train.ModelKey);

        train.SelectedScale = train.ModelScales.Single(scale => scale.Id == "s");
        Assert.Equal(Path.Combine(models, "yolov8_s"), train.OutputDirectory);

        // RF-DETR 的体量由 queries / 层数决定，没有规模字母
        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "rfdetr");
        Assert.Equal(Path.Combine(models, "rfdetr"), train.OutputDirectory);

        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "v11");
        Assert.Equal(Path.Combine(models, "yolov11_s"), train.OutputDirectory);

        train.SelectedModel = train.ModelFamilies.Single(model => model.Id == "v26");
        Assert.Equal(Path.Combine(models, "yolo26_s"), train.OutputDirectory);
    }

    [AvaloniaFact]
    public async Task No_Project_Falls_Back_To_Output_Export()
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

    /// <summary>
    /// YOLO26 的导出按官方默认走无 NMS：ONNX 输出 <c>[1, 300, 6]</c>，<c>.json</c> 里的 output.shape 也要跟着变，
    /// 免得下游按「4+类别数」去解读。
    /// </summary>
    [AvaloniaFact]
    public async Task Export_Yolo26_Writes_NmsFree_Onnx()
    {
        await using AppTestHost host = AppTestHost.Create();
        (TrainViewModel train, _) = await OpenSyntheticProjectAsync(host);

        train.OutputDirectory = SeedRun(host.Root, "yolo26n-detect-2");

        Assert.True(await train.ExportAsync(withOnnx: true));

        string onnx = Path.Combine(train.ExportDirectory, "yolo26_n.onnx");
        Assert.True(File.Exists(onnx));

        using var document = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(train.ExportDirectory, "yolo26_n.json")));
        string[] shape = document.RootElement
            .GetProperty("output")
            .GetProperty("shape")
            .EnumerateArray()
            .Select(value => value.ToString())
            .ToArray();
        Assert.Equal(["1", "300", "6"], shape);

        // 无 NMS 图能在检测端跑通（端到端布局由输出形状自动识别）
        using var runner = new Label2Train.Core.AutoLabel.YoloOnnxRunner(onnx);
        Assert.NotNull(runner.Predict(new byte[640 * 640 * 4], 640, 640, new Label2Train.Core.AutoLabel.AutoLabelOptions(onnx, 0.25, ClassNames: null)));
    }

    /// <summary>造一个「已经训练过」的输出目录：有元数据齐全的 best.safetensors 才有东西可导。</summary>
    private static string SeedRun(string root, string? architecture = null)
    {
        string run = Path.Combine(root, "runs");
        Directory.CreateDirectory(run);

        string target = architecture ?? Architecture;
        using YoloDetectionModel model = YoloArchitectures.CreateFromArchitecture(torch.CPU, target);

        Checkpoint.Save(
            Path.Combine(run, TrainingExportService.WeightsFileName),
            model,
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["architecture"] = target,
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
