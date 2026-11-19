using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Microsoft.Extensions.Logging;
using Label2Train.App.Controls;
using Label2Train.App.ViewModels;
using Label2Train.Core.AutoLabel;
using Label2Train.Core.Dataset;
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

    /// <summary>「训练参数」：epochs / batch / imgsz 收进对话框改，界面只做只读展示。</summary>
    private async void OnEditParams(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        var dialog = new TrainingParamsDialog(
            viewModel.Epochs,
            viewModel.BatchSize,
            viewModel.ImageSize,
            viewModel.SelectedModel?.Id ?? "v8");

        if (await dialog.ShowDialog<TrainParams?>(owner) is { } updated)
        {
            bool resized = (int)viewModel.ImageSize != updated.ImageSize;

            viewModel.Epochs = updated.Epochs;
            viewModel.BatchSize = updated.BatchSize;
            viewModel.ImageSize = updated.ImageSize;

            // imgsz 被纠正过就说一声，别让用户以为参数没保存
            if (resized && TrainingPresets.IsDetrFamily(viewModel.SelectedModel?.Id))
            {
                viewModel.Report(Loc.Format("Train.ImageSizeAdjusted", updated.ImageSize));
            }
        }
    }

    /// <summary>「YOLO」：选目录，按 YOLO 格式打开。</summary>
    private async void OnOpenYolo(object? sender, RoutedEventArgs e)
        => await OpenDatasetFolderAsync(DatasetFormat.Yolo, Loc.Get("Train.OpenYoloTip"));

    /// <summary>「RF-DETR」：选目录，按 COCO（RF-DETR）格式打开。</summary>
    private async void OnOpenRfDetr(object? sender, RoutedEventArgs e)
        => await OpenDatasetFolderAsync(DatasetFormat.Coco, Loc.Get("Train.OpenRfDetrTip"));

    private async Task OpenDatasetFolderAsync(DatasetFormat format, string title)
    {
        if (ViewModel is not { } viewModel)
        {
            return;
        }

        string? folder = await PickFolderAsync(title);

        if (folder is null)
        {
            return;
        }

        await viewModel.OpenDatasetAsync(folder, format);
    }

    /// <summary>
    /// 「TinyLabel」：库固定用标注端当前那个，只让用户挑库里的项目，
    /// 再把该项目的图片与标注读成训练数据集。
    /// </summary>
    private async void OnOpenTinyLabel(object? sender, RoutedEventArgs e)
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

        TinyLabelProjectSummary? project = await new ProjectPickerDialog(
                projects, viewModel.LibraryFile, viewModel.LastLibraryProjectId)
            .ShowDialog<TinyLabelProjectSummary?>(owner);

        if (project is null)
        {
            return;
        }

        // 打开项目 = 拉取它的图片清单与标注，顺手记住这个项目
        await viewModel.OpenLibraryProjectAsync(project.Id, $"{project.Name} · #{project.Id}");
    }

    private async void OnPickOutput(object? sender, RoutedEventArgs e)
    {
        string? folder = await PickFolderAsync(Loc.Get("Train.OutputDir"));

        if (folder is not null && ViewModel is { } viewModel)
        {
            viewModel.OutputDirectory = folder;
        }
    }

    /// <summary>「导出 ONNX」：把最优权重导成 best.onnx，之后「检测」直接用它。</summary>
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

        // 单选定的哪个模型就用哪个：yolov8.onnx / yolov11.onnx / rfdetr.onnx。
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

    private async Task<string?> PickFolderAsync(string title)
    {
        IStorageProvider? storage = TopLevel.GetTopLevel(this)?.StorageProvider;

        if (storage is null)
        {
            return null;
        }

        IReadOnlyList<IStorageFolder> folders = await storage.OpenFolderPickerAsync(
            new FolderPickerOpenOptions { Title = title, AllowMultiple = false });

        return folders.Count == 0 ? null : folders[0].TryGetLocalPath();
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
