using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using FreeSql;
using Label2Train.App.ViewModels;
using Label2Train.Core.Dataset;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Localization;
using Label2Train.Core.Storage;
using Label2Train.Label;

namespace Label2Train.App.Views;

/// <summary>
/// 标注工作台（主窗口的「标注」页）：左边项目列表，右边项目概览 + 图片清单。
/// 数据在 <see cref="MainWindowViewModel"/> 里，这里只做文件选择、对话框与标注窗口的转发。
/// </summary>
public partial class AnnotationView : UserControl
{
    // 由容器提供；预览 / 未初始化时都是 null，对应操作直接不做事
    private readonly LabelDatasetImportService? _importer = BeanFactory.Resolve<LabelDatasetImportService>();
    private readonly TinyLabelClassService? _classService = BeanFactory.Resolve<TinyLabelClassService>();
    private readonly AppSettings? _settings = BeanFactory.Resolve<AppSettings>();
    private readonly SettingsStore? _settingsStore = BeanFactory.Resolve<SettingsStore>();
    private readonly BlobStoreRegistry? _registry = BeanFactory.Resolve<BlobStoreRegistry>();
    private readonly IFreeSql? _orm = BeanFactory.Resolve<IFreeSql>();

    public AnnotationView() => InitializeComponent();

    private MainWindowViewModel? ViewModel => DataContext as MainWindowViewModel;

    /// <summary>对话框 / 标注窗口的宿主窗口（本控件嵌在主窗口里，不是窗口本身）。</summary>
    private Window? Owner => TopLevel.GetTopLevel(this) as Window;

    private static Localizer Loc => Localizer.Instance;

    // ── 刷新 ─────────────────────────────────────────────────────────────────

    private async void OnRefresh(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await viewModel.RefreshProjectsAsync();
        }
    }

    // ── 项目管理 ─────────────────────────────────────────────────────────────

    private async void OnCreateProject(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        // 库在启动时就建好了（空目录也有一个只有表的库），这里直接建项目
        var dialog = new ProjectEditorWindow();
        ProjectDraft? draft = await dialog.ShowDialog<ProjectDraft?>(owner);

        if (draft is not null)
        {
            await viewModel.CreateProjectAsync(draft);
        }
    }

    // ── 从数据集建项目（YOLO / RF-DETR） ─────────────────────────────────────

    private async void OnImportYoloDataset(object? sender, RoutedEventArgs e)
        => await ImportDatasetAsProjectAsync(DatasetFormat.Yolo);

    private async void OnImportRfDetrDataset(object? sender, RoutedEventArgs e)
        => await ImportDatasetAsProjectAsync(DatasetFormat.Coco);

    /// <summary>选目录 → 校验是不是目标数据集 → 输项目名 → 建项目并导入图片与标注。</summary>
    private async Task ImportDatasetAsProjectAsync(DatasetFormat format)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        string? source = await PickFolderAsync(Loc.Format("Import.PickTitleFor", format.ToDisplayName()));

        if (source is null)
        {
            return;
        }

        viewModel.Status = Loc.Get("Import.Verifying");

        DatasetImportCheck check = await viewModel.CheckImportAsync(source, format);

        if (!check.IsValid || check.Probe is null)
        {
            viewModel.Report(check.Message);

            await new ConfirmWindow(
                Loc.Get("Import.NotImportableTitle"),
                check.Message,
                Loc.Get("Confirm.Notice")).ShowDialog<bool>(owner);

            return;
        }

        var dialog = new DatasetImportWindow(format, source, check.Probe);
        string? name = await dialog.ShowDialog<string?>(owner);

        if (name is not null)
        {
            await viewModel.ImportAsync(new ImportRequest(source, name, format));
        }
    }

    /// <summary>生成模拟数据：结果直接进当前库与项目目录，没有外部文件要选。</summary>
    private async void OnGenerateSyntheticProject(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        var dialog = new SyntheticProjectWindow();
        SyntheticProjectDraft? draft = await dialog.ShowDialog<SyntheticProjectDraft?>(owner);

        if (draft is not null)
        {
            await viewModel.GenerateSyntheticProjectAsync(draft);
        }
    }

    private async void OnEditProject(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        if (viewModel.SelectedProject is not { } project)
        {
            viewModel.Status = Loc.Get("Label.PickProjectFirst");
            return;
        }

        var dialog = new ProjectEditorWindow(project.Name, project.Summary.Description);
        ProjectDraft? draft = await dialog.ShowDialog<ProjectDraft?>(owner);

        if (draft is not null)
        {
            await viewModel.UpdateProjectAsync(draft);
        }
    }

    private async void OnDeleteProject(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        if (viewModel.SelectedProject is not { } project)
        {
            viewModel.Status = Loc.Get("Label.PickProjectFirst");
            return;
        }

        var confirm = new ConfirmWindow(
            Loc.Get("Confirm.DeleteTitle"),
            Loc.Format("Confirm.DeleteMessage", project.Name),
            Loc.Get("Common.Delete"));

        if (await confirm.ShowDialog<bool>(owner))
        {
            await viewModel.DeleteProjectAsync();
        }
    }

    private async void OnToggleClosed(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { SelectedProject: not null } viewModel)
        {
            await viewModel.ToggleProjectClosedAsync();
        }
    }

    // ── 图片入库 ─────────────────────────────────────────────────────────────

    private async void OnImportImages(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel)
        {
            return;
        }

        IStorageProvider? storage = TopLevel.GetTopLevel(this)?.StorageProvider;

        if (storage is null)
        {
            return;
        }

        IReadOnlyList<IStorageFile> files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = Loc.Get("Label.ImportImagesTitle"),
            AllowMultiple = true,
            FileTypeFilter =
            [
                new FilePickerFileType(Loc.Get("Label.ImagesFileType"))
                {
                    Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"],
                },
            ],
        });

        string[] paths = [.. files
            .Select(file => file.TryGetLocalPath())
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Select(path => path!)];

        if (paths.Length > 0)
        {
            await viewModel.ImportImagesAsync(paths);
        }
    }

    private async void OnImportDataset(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner || _importer is not { } importer)
        {
            return;
        }

        var dialog = new ImportDialog(viewModel.DataDirectory, importer);
        ImportRequest? request = await dialog.ShowDialog<ImportRequest?>(owner);

        if (request is not null)
        {
            await viewModel.ImportAsync(request);
        }
    }

    private async void OnImportFolder(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel)
        {
            return;
        }

        string? folder = await PickFolderAsync(Loc.Get("Label.ImportFolderTitle"));

        if (folder is not null)
        {
            await viewModel.ImportFolderAsync(folder, copy: true);
        }
    }

    private async void OnScanFolder(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel)
        {
            return;
        }

        string? folder = await PickFolderAsync(Loc.Get("Label.ScanFolderTitle"));

        if (folder is not null)
        {
            await viewModel.ImportFolderAsync(folder, copy: false);
        }
    }

    // ── 产出 ─────────────────────────────────────────────────────────────────

    private async void OnExportDataset(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        if (viewModel.SelectedProject is null)
        {
            viewModel.Status = Loc.Get("Label.PickProjectFirst");
            return;
        }

        var dialog = new ExportDatasetWindow(viewModel.DataDirectory);
        ExportRequest? request = await dialog.ShowDialog<ExportRequest?>(owner);

        if (request is not null)
        {
            await viewModel.ExportAsync(request.Format, request.Destination);
        }
    }

    private void OnOpenProjectFolder(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel)
        {
            return;
        }

        if (viewModel.SelectedProject is not { } project)
        {
            viewModel.Status = Loc.Get("Label.PickProjectFirst");
            return;
        }

        string folder = TinyLabelWorkspace.ProjectDirectory(
            StorageLocation.FromPath(viewModel.DataDirectory),
            project.Id);

        if (!Directory.Exists(folder))
        {
            folder = viewModel.DataDirectory;
        }

        OpenInExplorer(folder);
    }

    private async void OnManageClasses(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner || _classService is not { } service)
        {
            return;
        }

        if (viewModel.SelectedProject is not { } project)
        {
            viewModel.Status = Loc.Get("Label.PickProjectFirst");
            return;
        }

        var dialog = new ClassEditorWindow(service, StorageLocation.FromPath(viewModel.DataDirectory), project.Id);
        bool changed = await dialog.ShowDialog<bool>(owner);

        if (changed)
        {
            await viewModel.RefreshCurrentProjectAsync(viewModel.SelectedImage?.Id);
        }
    }

    // ── 标注 ─────────────────────────────────────────────────────────────────

    /// <summary>「开始标注」：标当前选中行，没有选中就标第一张。</summary>
    private async void OnStartAnnotation(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await AnnotateAsync(viewModel.SelectedImage ?? viewModel.Images.FirstOrDefault());
        }
    }

    /// <summary>行尾 ⋮ 菜单：Flyout 里的 MenuItem 拿不到行的 DataContext，点按钮时先记下这一行。</summary>
    private ProjectImageViewModel? _actionRow;

    private void OnImageActionMenu(object? sender, RoutedEventArgs e)
    {
        _actionRow = (sender as Control)?.Tag as ProjectImageViewModel;
    }

    private async void OnAnnotateImageAction(object? sender, RoutedEventArgs e)
        => await AnnotateAsync(_actionRow ?? ViewModel?.SelectedImage);

    /// <summary>菜单「重置为未标注」：只改状态，标注数据留着（与 TinyLabel 一致）。</summary>
    private async void OnResetImageState(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || (_actionRow ?? viewModel.SelectedImage) is not { } row)
        {
            return;
        }

        await viewModel.ResetImageStateAsync(row);
    }

    /// <summary>菜单「清除标注」：删标注数据，先确认。</summary>
    private async void OnClearAnnotation(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner
            || (_actionRow ?? viewModel.SelectedImage) is not { } row)
        {
            return;
        }

        var confirm = new ConfirmWindow(
            Loc.Get("Confirm.ClearAnnotationTitle"),
            Loc.Format("Label.ConfirmClearAnnotation", row.DisplayPath),
            Loc.Get("Common.Clear"));

        if (await confirm.ShowDialog<bool>(owner))
        {
            await viewModel.ClearAnnotationForAsync(row);
        }
    }

    /// <summary>菜单「删除图片」：库里的行 + 导入模式下的原图，先确认。</summary>
    private async void OnDeleteImage(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner
            || (_actionRow ?? viewModel.SelectedImage) is not { } row)
        {
            return;
        }

        var confirm = new ConfirmWindow(
            Loc.Get("Confirm.DeleteImageTitle"),
            Loc.Format("Label.ConfirmDeleteImage", row.DisplayPath),
            Loc.Get("Common.Delete"));

        if (await confirm.ShowDialog<bool>(owner))
        {
            await viewModel.DeleteImageForAsync(row);
        }
    }

    /// <summary>双击清单行直接进标注，跟 TinyLabel 的行单击选中、双击打开一致。</summary>
    private async void OnRowDoubleTapped(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await AnnotateAsync(viewModel.SelectedImage);
        }
    }

    /// <summary>供命令行入口使用：对当前选中行（没有就第一张）打开标注对话框。</summary>
    public Task AnnotateSelectedAsync()
        => ViewModel is { } viewModel
            ? AnnotateAsync(viewModel.SelectedImage ?? viewModel.Images.FirstOrDefault())
            : Task.CompletedTask;

    private async Task AnnotateAsync(ProjectImageViewModel? item)
    {
        if (ViewModel is not { } viewModel || Owner is not { } owner)
        {
            return;
        }

        if (item is null)
        {
            viewModel.Report("先选一行图片再标注");
            return;
        }

        if (viewModel.SelectedProject is null || string.IsNullOrWhiteSpace(viewModel.DatabasePath))
        {
            viewModel.Report(Loc.Get("Label.PickProjectFirst"));
            return;
        }

        if (_settings is null || _settingsStore is null || _registry is null || _orm is null)
        {
            viewModel.Report("服务容器不可用，标注窗口打不开");
            return;
        }

        try
        {
            long projectId = viewModel.SelectedProject.Id;

            // 一次开窗：当前这张 + 本页后面还没标注的（保存并继续在同窗口换图，不关窗重开）
            List<AnnotationTarget> targets = BuildAnnotationQueue(viewModel, item, _registry);

            var window = new AnnotationWindow(
                _orm,
                projectId,
                targets,
                viewModel.Classes,
                _settings,
                _settingsStore);

            await window.ShowDialog<AnnotationOutcome>(owner);

            // 关窗后刷新当前页，并落回最后标的那张
            await viewModel.RefreshCurrentProjectAsync(window.LastImageId ?? item.Id);
        }
        catch (Exception error)
        {
            viewModel.Report($"打开标注窗口失败：{error.Message}");
        }
    }

    /// <summary>连续标注队列：起点这张在前，其后是同页还没标注的图（不绕回已标注的）。</summary>
    private static List<AnnotationTarget> BuildAnnotationQueue(
        MainWindowViewModel viewModel,
        ProjectImageViewModel start,
        BlobStoreRegistry registry)
    {
        var order = new List<ProjectImageViewModel> { start };

        order.AddRange(viewModel.Images.Where(image => !ReferenceEquals(image, start) && image.IsPending));

        return
        [
            .. order.Select(image => new AnnotationTarget(
                image.Row,
                image.Image,
                registry.Get(image.Image.Location))),
        ];
    }

    // ── 分页 ─────────────────────────────────────────────────────────────────

    private async void OnPrevPage(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await viewModel.TurnPageAsync(-1);
        }
    }

    private async void OnNextPage(object? sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await viewModel.TurnPageAsync(1);
        }
    }

    // ── 内部 ─────────────────────────────────────────────────────────────────

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

    private static bool OpenInExplorer(string folder)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = $"\"{folder}\"",
                UseShellExecute = true,
            });

            return true;
        }
        catch (Exception)
        {
            // 打不开资源管理器不影响程序
            return false;
        }
    }
}
