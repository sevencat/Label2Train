using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Microsoft.Extensions.Logging;
using Label2Train.App.Controls;
using Label2Train.App.ViewModels;
using Label2Train.Core.AutoLabel;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Localization;
using Label2Train.Core.Storage;
using Label2Train.Train.Training;

namespace Label2Train.App.Views;

/// <summary>
/// 训练工作台（主窗口的「训练」页）。数据在 <see cref="TrainViewModel"/> 里，
/// 这里只做文件选择、对话框与按钮转发。
/// </summary>
public partial class TrainView : UserControl
{
    public TrainView() => InitializeComponent();

    private TrainViewModel? ViewModel => DataContext as TrainViewModel;

    /// <summary>对话框的宿主窗口（本控件嵌在主窗口里，不是窗口本身）。</summary>
    private Window? Owner => TopLevel.GetTopLevel(this) as Window;

    private static Localizer Loc => Localizer.Instance;

    /// <summary>
    /// 「训练目标设置」：项目、模型、规模、epochs / batch / imgsz 一次配好，主界面只读展示。
    /// <para>
    /// 训练入口只有这一条：外部 YOLO / COCO 数据集必须先在导入流程里转成库项目，
    /// 否则训练集会绕开库，导出的模型和项目也对不上号。
    /// </para>
    /// </summary>
    private async void OnSetupTarget(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        IReadOnlyList<TinyLabelProjectSummary> projects = await viewModel.ListLibraryProjectsAsync();

        if (projects.Count == 0)
        {
            viewModel.Report(Loc.Format("Train.NoProjectsInLibrary", viewModel.LibraryFile));
            return;
        }

        var dialog = new TrainingTargetDialog(
            projects,
            viewModel.ProjectId,
            viewModel.ModelFamilies,
            viewModel.ModelScales,
            viewModel.SelectedModel?.Id ?? "v8",
            viewModel.SelectedScale?.Id ?? "n",
            viewModel.Epochs,
            viewModel.BatchSize,
            viewModel.ImageSize);

        if (await dialog.ShowDialog<TrainingTarget?>(owner) is not { } target)
        {
            return;
        }

        bool resized = (int)viewModel.ImageSize != target.ImageSize;
        bool projectChanged = target.Project.Id != viewModel.ProjectId;

        // 先落模型与规模：OutputDirectory 跟着 ModelKey 走，随后打开项目时会按新项目 + 新模型再派生一次；
        // 顺序反了就会先用旧模型名建目录，白跑一趟
        viewModel.SelectedModel = viewModel.ModelFamilies
            .FirstOrDefault(model => model.Id == target.FamilyId) ?? viewModel.SelectedModel;
        viewModel.SelectedScale = viewModel.ModelScales
            .FirstOrDefault(scale => scale.Id == target.ScaleId) ?? viewModel.SelectedScale;
        viewModel.Epochs = target.Epochs;
        viewModel.BatchSize = target.BatchSize;
        viewModel.ImageSize = target.ImageSize;

        // 换项目 = 重新拉图片清单与标注，顺手记住这个项目
        if (projectChanged)
        {
            await viewModel.OpenLibraryProjectAsync(
                target.Project.Id,
                $"{target.Project.Name} · #{target.Project.Id}");
        }

        // imgsz 被纠正过就说一声，别让用户以为参数没保存
        if (resized && TrainingPresets.IsDetrFamily(target.FamilyId))
        {
            viewModel.Report(Loc.Format("Train.ImageSizeAdjusted", target.ImageSize));
        }
    }

    /// <summary>
    /// 「导出 ONNX」：把最优权重导成 <c>{模型名}_{规模}.onnx</c> 放进项目 models 目录，
    /// 之后「检测」直接用它。
    /// </summary>
    private async void OnExportOnnx(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await viewModel.ExportAsync(withOnnx: true);
        }
    }

    /// <summary>「导出 safetensors」：权重 + model.json（架构 / 类别 / 输入尺寸）。</summary>
    private async void OnExportWeights(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await viewModel.ExportAsync(withOnnx: false);
        }
    }

    /// <summary>「打开」：在资源管理器里打开输出目录（不存在就先建出来，免得报错）。</summary>
    private void OnOpenOutput(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(viewModel.OutputDirectory);
            Process.Start(new ProcessStartInfo("explorer.exe", $"\"{viewModel.OutputDirectory}\"")
            {
                UseShellExecute = true,
            });
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.ComponentModel.Win32Exception)
        {
            viewModel.Report(Loc.Format("Train.OpenOutputFailed", viewModel.OutputDirectory));
        }
    }

    private async void OnStart(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await viewModel.StartTrainingAsync();
        }
    }

    private void OnStop(object? sender, RoutedEventArgs e) => ViewModel?.StopTraining();

    private void OnPause(object? sender, RoutedEventArgs e) => ViewModel?.PauseTraining();

    private void OnResume(object? sender, RoutedEventArgs e) => ViewModel?.ResumeTraining();

    /// <summary>「模拟数据」：随机画圆 / 正方形 / 三角形连同标注一起生成，存成 YOLO、RF-DETR 或 TinyLabel。</summary>
    private async void OnSimulate(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } main || Owner is not { } owner)
        {
            return;
        }

        var simulation = new SimulationViewModel(
            BeanFactory.Require<ILoggerFactory>().CreateLogger<SimulationViewModel>(),
            BeanFactory.Require<BlobStoreRegistry>(),
            BeanFactory.Require<TinyLabelDatabase>());

        var dialog = new SimulationWindow(simulation);

        await dialog.ShowDialog(owner);

        // 生成到当前库里的话，项目下拉要能立刻看见
        if (simulation.HasGenerated)
        {
            await main.RefreshProjectsAsync();
        }
    }

    /// <summary>
    /// 检测验证：选图片 → 用 ONNX 模型推理 → 弹窗把结果叠在图上。
    /// 不占页面空间，只在需要时开一个对话框。
    /// </summary>
    private async void OnDetect(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        // 选文件、读图、推理、弹窗都在里面：任何一步失败都要落到状态行与日志里，
        // 不能只是「点了没反应」
        try
        {
            await DetectAsync(viewModel, owner);
        }
        catch (Exception error)
        {
            viewModel.Report($"检测失败：{error.Message}");
            BeanFactory.Resolve<ILoggerFactory>()?.CreateLogger<TrainView>().LogError(error, "检测失败");
        }
    }

    private async Task DetectAsync(TrainViewModel viewModel, Window owner)
    {
        AppSettings settings = BeanFactory.Require<AppSettings>();

        string? imagePath = await PickFileAsync(
            "选择要检测的图片",
            [new FilePickerFileType("图片") { Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"] }]);

        if (imagePath is null)
        {
            return;
        }

        // 只用当前训练配置的模型：{模型名}_{规模}.onnx。
        // 找不到就明确报缺文件，绝不悄悄换成别的模型（换了框会整体错、甚至崩在输出名上）
        string modelPath = viewModel.ResolveDetectModelPath();

        if (!File.Exists(modelPath))
        {
            viewModel.Report(viewModel.DescribeMissingDetectModel());
            return;
        }

        viewModel.Status = "检测中…";

        using IAutoLabelRunner runner = AutoLabelRunnerFactory.Create(modelPath);
        using var bitmap = new Bitmap(imagePath);

        byte[] bgra = BitmapPixels.ToBgra(bitmap, out int width, out int height);

        var options = new AutoLabelOptions(
            modelPath,
            settings.AutoLabelConfidence,
            ClassNames: viewModel.Classes.Count > 0 ? viewModel.Classes : null);

        IReadOnlyList<AutoLabelCandidate> candidates =
            await Task.Run(() => runner.Predict(bgra, width, height, options));

        viewModel.Status = $"检测完成：{candidates.Count} 个框";

        var window = new DetectWindow(
            bitmap,
            candidates,
            Path.GetFileName(imagePath),
            Path.GetFileName(modelPath));

        await window.ShowDialog(owner);
    }

    private static void SaveSettings(SettingsStore settingsStore, AppSettings settings)
    {
        try
        {
            settingsStore.Save(settings);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // 配置写不进去不影响本次检测
        }
    }

    private async Task<string?> PickFileAsync(
        string title,
        FilePickerFileType[] filters,
        string? startFolder = null)
    {
        IStorageProvider? storage = TopLevel.GetTopLevel(this)?.StorageProvider;

        if (storage is null)
        {
            return null;
        }

        IStorageFolder? start = null;

        if (!string.IsNullOrWhiteSpace(startFolder) && Directory.Exists(startFolder))
        {
            start = await storage.TryGetFolderFromPathAsync(startFolder);
        }

        IReadOnlyList<IStorageFile> files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = filters,
            SuggestedStartLocation = start,
        });

        return files.Count == 0 ? null : files[0].TryGetLocalPath();
    }
}
