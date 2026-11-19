using System.Collections.ObjectModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using Injectio.Attributes;
using Microsoft.Extensions.Logging;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Legends;
using OxyPlot.Series;
using Label2Train.App.Rendering;
using Label2Train.Core;
using Label2Train.Core.Dataset;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Infrastructure.Logging;
using Label2Train.Core.Localization;
using Label2Train.Core.Simulation;
using Label2Train.Core.Storage;
using Label2Train.Train.Export;
using Label2Train.Train.Training;

namespace Label2Train.App.ViewModels;

/// <summary>
/// 训练工作台：上面工具栏、中间曲线、下面状态与日志，作为主窗口的第二个页面使用。
/// 检测验证不占页面，收在工具栏的按钮里：选一张图 → 推理 → 弹对话框看叠加结果。
/// </summary>
[RegisterSingleton<TrainViewModel>]
public partial class TrainViewModel : ObservableObject
{
    private readonly ILogger<TrainViewModel> _logger;
    private readonly DatasetStoreResolver _resolver;
    private readonly IDatasetTrainingRunner _runner;
    private readonly AppSettings _settings;
    private readonly SettingsStore _settingsStore;
    private readonly TinyLabelProjectCatalog _catalog;
    private readonly TinyLabelDatabase _database;
    private readonly TrainingExportService _exporter;

    private IDatasetStore? _store;

    /// <summary>当前数据集里的图片张数：0 张时不让开始训练（空数据集只会训出 NaN）。</summary>
    private int _imageCount;

    /// <summary>当前数据集来自哪个标注项目；0 表示还没选项目（此时导出退回输出目录）。</summary>
    private long _projectId;

    private CancellationTokenSource? _training;
    private TrainControl? _control;

    /// <summary>第一次切到训练页时才去列项目，不在启动时白跑一趟。</summary>
    private bool _initialized;

    [ObservableProperty]
    private string _status = Localizer.Instance.Get("Train.Ready");

    [ObservableProperty]
    private string _logText = string.Empty;

    [ObservableProperty]
    private string _datasetPath = string.Empty;

    [ObservableProperty]
    private string _datasetSummary = Localizer.Instance.Get("Train.NoDataset");

    [ObservableProperty]
    private string _datasetStats = string.Empty;

    [ObservableProperty]
    private decimal _epochs = TrainingPresets.Yolo.Epochs;

    [ObservableProperty]
    private decimal _batchSize = TrainingPresets.Yolo.BatchSize;

    [ObservableProperty]
    private decimal _imageSize = TrainingPresets.Yolo.ImageSize;

    /// <summary>训练成哪个家族：YOLOv8 / YOLOv11 / RF-DETR。</summary>
    public IReadOnlyList<ModelOption> ModelFamilies { get; } =
    [
        new ModelOption("v8", "YOLOv8"),
        new ModelOption("v11", "YOLOv11"),
        new ModelOption("rfdetr", "RF-DETR"),
    ];

    /// <summary>当前选的模型家族；训练时原样传给 run option 的 Family。</summary>
    [ObservableProperty]
    private ModelOption? _selectedModel;

    /// <summary>
    /// YOLO 的规模（ultralytics 的 n / s / m / l / x，对应 yolov8n … yolov8x）。
    /// 只对 YOLO 家族生效：RF-DETR 的体量由它自己的 queries / 层数决定，不看这个。
    /// 换规模等于换网络结构，旧权重不能接着训，只能重新训练。
    /// </summary>
    public IReadOnlyList<ScaleOption> ModelScales { get; } =
    [
        new ScaleOption("n", "n · nano"),
        new ScaleOption("s", "s · small"),
        new ScaleOption("m", "m · medium"),
        new ScaleOption("l", "l · large"),
        new ScaleOption("x", "x · xlarge"),
    ];

    /// <summary>当前规模；训练时原样传给 run option 的 Scale（默认 n，与之前固定值一致）。</summary>
    [ObservableProperty]
    private ScaleOption? _selectedScale;

    /// <summary>规模下拉只在 YOLO 家族下有用，选到 RF-DETR 就置灰。</summary>
    public bool IsScaleEnabled => !TrainingPresets.IsDetrFamily(SelectedModel?.Id);

    /// <summary>主界面「模型」只读显示：家族名。</summary>
    public string ModelNameText => SelectedModel?.Name ?? string.Empty;

    /// <summary>主界面「规模」只读显示：RF-DETR 不看规模，显示为「不适用」。</summary>
    public string ScaleText => IsScaleEnabled
        ? SelectedScale?.Name ?? string.Empty
        : Loc.Get("Train.ScaleNA");

    /// <summary>「模型名 规模」拼一段：摘要、检测提示都按这个说法。</summary>
    public string ModelText => IsScaleEnabled
        ? $"{SelectedModel?.Name} {SelectedScale?.Id}"
        : SelectedModel?.Name ?? string.Empty;

    /// <summary>主界面「训练目标」一行摘要：项目 · 模型 规模；参数在右边单独只读展示。</summary>
    public string TargetSummary => Loc.Format(
        "Train.TargetSummary",
        string.IsNullOrWhiteSpace(DatasetPath) ? Loc.Get("Train.NoDataset") : DatasetPath,
        ModelText);

    /// <summary>最近一次导出的 ONNX 路径（只用于展示与日志；检测走的是单选对应的前缀文件）。</summary>
    public string LastOnnxPath { get; private set; } = string.Empty;

    /// <summary>epochs / batch / imgsz 在界面上只读展示，改值走「修改」对话框。</summary>
    public string EpochsText => ((int)Epochs).ToString(CultureInfo.InvariantCulture);

    public string BatchSizeText => ((int)BatchSize).ToString(CultureInfo.InvariantCulture);

    public string ImageSizeText => ((int)ImageSize).ToString(CultureInfo.InvariantCulture);

    partial void OnEpochsChanged(decimal value)
    {
        OnPropertyChanged(nameof(EpochsText));
        OnPropertyChanged(nameof(TargetSummary));
    }

    partial void OnBatchSizeChanged(decimal value)
    {
        OnPropertyChanged(nameof(BatchSizeText));
        OnPropertyChanged(nameof(TargetSummary));
    }

    partial void OnImageSizeChanged(decimal value)
    {
        OnPropertyChanged(nameof(ImageSizeText));
        OnPropertyChanged(nameof(TargetSummary));
        RaiseRunState();
    }

    partial void OnDatasetPathChanged(string value) => OnPropertyChanged(nameof(TargetSummary));

    partial void OnSelectedModelChanged(ModelOption? value)
    {
        RaiseRunState();
        OnPropertyChanged(nameof(IsScaleEnabled));
        OnPropertyChanged(nameof(ModelKey));
        OnPropertyChanged(nameof(ModelNameText));
        OnPropertyChanged(nameof(ScaleText));
        OnPropertyChanged(nameof(ModelText));
        OnPropertyChanged(nameof(TargetSummary));
        SyncOutputDirectory();
        ReportImageSizeIfBlocked();
    }

    /// <summary>换规模就是换网络结构：产物目录跟着换成 <c>models/yolov8_m</c> 这一档。</summary>
    partial void OnSelectedScaleChanged(ScaleOption? value)
    {
        OnPropertyChanged(nameof(ModelKey));
        OnPropertyChanged(nameof(ScaleText));
        OnPropertyChanged(nameof(ModelText));
        OnPropertyChanged(nameof(TargetSummary));
        SyncOutputDirectory();
    }

    /// <summary>
    /// 当前家族的 imgsz 能不能用：RF-DETR 的 ONNX 导出要求边长同时被 patch 14 与下采样 16
    /// 整除，也就是 112 的倍数。这里就拦住，别等训练完才发现导不出来。
    /// </summary>
    public bool IsImageSizeAllowed =>
        !TrainingPresets.IsDetrFamily(SelectedModel?.Id)
        || TrainingPresets.IsValidDetrImageSize((int)ImageSize);

    /// <summary>切到 DETR 家族而 imgsz 不合法时，状态行直接说清怎么改。</summary>
    private void ReportImageSizeIfBlocked()
    {
        if (IsTraining || IsImageSizeAllowed)
        {
            return;
        }

        Report(Loc.Format(
            "Train.DetrImageSizeBlocked",
            TrainingPresets.NearestValidDetrImageSize((int)ImageSize)));
    }

    [ObservableProperty]
    private string _outputDirectory = Path.Combine(AppDataPaths.DataDirectory, "runs");

    [ObservableProperty]
    private bool _isTraining;

    /// <summary>是否处于暂停状态（训练线程停在批次边界）。</summary>
    [ObservableProperty]
    private bool _isPaused;

    [ObservableProperty]
    private double _progress;

    [ObservableProperty]
    private string _runInfo = string.Empty;

    // ── 训练曲线（OxyPlot）：x 轴按 epoch 序号，mAP 走左轴、loss 走右轴 ──

    private const string AxisTextColor = "#37474F";
    private const string GridColor = "#CFD8DC";

    // 三条线要在白底上一眼分清：以前 mAP50 绿、loss 青绿、mAP50-95 深紫，
    // 绿和青绿几乎同色，曲线叠在一起时看不出哪条是哪条。改成蓝 / 品红 / 橙三个相差大的色相。
    private const string Map50Color = "#1565C0";
    private const string Map5095Color = "#D81B60";
    private const string LossColor = "#EF6C00";

    private readonly LinearAxis _epochAxis;
    private readonly LinearAxis _mapAxis;
    private readonly LinearAxis _lossAxis;
    private readonly LineSeries _map50Series;
    private readonly LineSeries _map5095Series;
    private readonly LineSeries _lossSeries;

    /// <summary>曲线模型：两个 mAP 挂左轴（0~1），loss 挂右轴。</summary>
    public PlotModel Plot { get; }

    /// <summary>还没有任何点：曲线区盖一层提示文字。</summary>
    [ObservableProperty]
    private bool _hasChartData;

    /// <summary>
    /// 库（<c>label2train.db</c>）里的标注项目：训练的输入就是标注端写进去的那些项目，
    /// 不用再让用户去挑目录。
    /// </summary>
    public ObservableCollection<ProjectOption> Projects { get; } = [];

    /// <summary>当前选中的标注项目。</summary>
    [ObservableProperty]
    private ProjectOption? _selectedProject;

    /// <summary>数据集类别（检测验证时用来给框命名）。</summary>
    public IReadOnlyList<string> Classes { get; private set; } = [];

    /// <summary>当前打开的数据集是否可用于训练。</summary>
    public bool HasDataset => _store is not null;

    // ── 运行状态（按钮可用性与顶栏徽标都从这里取）──────────────────────────────

    /// <summary>
    /// 没选数据集 / 没选输出路径时「开始训练」置灰：训练既要知道数据在哪，
    /// 也要知道权重往哪写，缺一个都会在跑起来之后才报错。
    /// 数据集里一张图都没有时同样置灰——空数据集跑出来的是一堆 NaN，不如直接拦住。
    /// imgsz 跟当前家族不兼容（RF-DETR 不是 112 的倍数）时也置灰，因为训完也导不出 ONNX。
    /// </summary>
    public bool CanStart => !IsTraining
        && _store is not null
        && _imageCount > 0
        && !string.IsNullOrWhiteSpace(OutputDirectory)
        && IsImageSizeAllowed;

    public bool CanPause => IsTraining && !IsPaused;

    public bool CanResume => IsTraining && IsPaused;

    public bool CanFinish => IsTraining;

    public bool ShowRunningBadge => IsTraining && !IsPaused;

    public bool ShowPausedBadge => IsPaused;

    public bool ShowReadyBadge => !IsTraining;

    partial void OnIsTrainingChanged(bool value) => RaiseRunState();

    partial void OnIsPausedChanged(bool value) => RaiseRunState();

    partial void OnOutputDirectoryChanged(string value) => RaiseRunState();

    private void RaiseRunState()
    {
        OnPropertyChanged(nameof(IsImageSizeAllowed));
        OnPropertyChanged(nameof(CanStart));
        OnPropertyChanged(nameof(CanPause));
        OnPropertyChanged(nameof(CanResume));
        OnPropertyChanged(nameof(CanFinish));
        OnPropertyChanged(nameof(ShowRunningBadge));
        OnPropertyChanged(nameof(ShowPausedBadge));
        OnPropertyChanged(nameof(ShowReadyBadge));
    }

    private static Localizer Loc => Localizer.Instance;

    public TrainViewModel(
        ILogger<TrainViewModel> logger,
        LogBuffer logBuffer,
        DatasetStoreResolver resolver,
        IDatasetTrainingRunner runner,
        AppSettings settings,
        SettingsStore settingsStore,
        TinyLabelProjectCatalog catalog,
        TinyLabelDatabase database,
        TrainingExportService exporter)
    {
        _logger = logger;
        _resolver = resolver;
        _runner = runner;
        _settings = settings;
        _settingsStore = settingsStore;
        _catalog = catalog;
        _database = database;
        _exporter = exporter;
        _selectedModel = ModelFamilies[0];
        _selectedScale = ModelScales[0];

        // 轴范围一律给实数：OxyPlot 2.x 在算刻度时会对 NaN 取符号，
        // 一旦某根轴的 Minimum/Maximum 是 NaN 就抛 ArithmeticException，整张图都画不出来
        _epochAxis = CreateAxis("epoch", AxisPosition.Bottom, minimum: 0, maximum: 10, minimumStep: 1);
        _mapAxis = CreateAxis("mAP", AxisPosition.Left, minimum: 0, maximum: 1, minimumStep: 0.05);
        _lossAxis = CreateAxis("loss", AxisPosition.Right, minimum: 0, maximum: 1);

        _map50Series = CreateSeries("mAP50", Map50Color, _mapAxis.Key);
        _map5095Series = CreateSeries("mAP50-95", Map5095Color, _mapAxis.Key);
        _lossSeries = CreateSeries("loss", LossColor, _lossAxis.Key);

        Plot = new PlotModel
        {
            DefaultFontSize = 12,
            TextColor = OxyColor.Parse(AxisTextColor),
            PlotAreaBorderColor = OxyColor.Parse(GridColor),
            PlotAreaBorderThickness = new OxyThickness(0, 0, 0, 1),
            IsLegendVisible = true,
            // SimdSkia 后端不容忍 OxyPlot.Avalonia 画线段的方式，不包一层图表只会渲染出报错文字
            RenderingDecorator = context => new SegmentSafeRenderContext(context),
        };

        Plot.Legends.Add(new Legend
        {
            LegendPosition = LegendPosition.TopRight,
            LegendPlacement = LegendPlacement.Inside,
            LegendBackground = OxyColor.Parse("#F2FFFFFF"),
            LegendBorder = OxyColor.Parse(GridColor),
            LegendBorderThickness = 1,
        });

        Plot.Axes.Add(_epochAxis);
        Plot.Axes.Add(_mapAxis);
        Plot.Axes.Add(_lossAxis);

        Plot.Series.Add(_map50Series);
        Plot.Series.Add(_map5095Series);
        Plot.Series.Add(_lossSeries);

        // 训练产物跟着库里的项目走：选过项目就是 projects/{id}/runs，没选就是库目录下的 runs。
        OutputDirectory = RunDirectoryFor(projectId: 0);

        logger.LogInformation("训练工作台已创建，等待选择数据集");

        LogText = string.Join(Environment.NewLine, logBuffer.Snapshot().Select(entry => entry.ToString()));
    }

    /// <summary>
    /// 第一次切到训练页时跑一次：列出库里的项目。
    /// 这里<b>不</b>自动把上次的项目打开——数据集没被用户明确选中时「开始训练」必须是灰的，
    /// 否则启动后就出现一个可点的按钮，看起来像「没选也能训练」。
    /// 上次用的项目只在「选择项目」对话框里预选，点一下确定就打开了。
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        if (_initialized)
        {
            return;
        }

        _initialized = true;

        await RefreshProjectsAsync(ct).ConfigureAwait(true);

        if (_store is null
            && Projects.FirstOrDefault(item => item.Id == _settings.LastTrainLibraryProjectId) is { } last)
        {
            Status = Loc.Format("Train.LastProjectHint", last.Name);
        }
    }

    /// <summary>切换界面语言后重算不依赖数据的固定文案。</summary>
    public void RefreshLocalizedTexts()
    {
        if (_store is null)
        {
            Status = Loc.Get("Train.Ready");
            DatasetSummary = Loc.Get("Train.NoDataset");
        }
    }

    // ── 数据集 ────────────────────────────────────────────────────────────────

    /// <summary>
    /// 列出库（<c>label2train.db</c>）里的标注项目：训练的输入就是标注端写进去的这些项目，
    /// 不用再让用户挑目录。
    /// </summary>
    public async Task RefreshProjectsAsync(CancellationToken ct = default)
    {
        string? folder = Path.GetDirectoryName(_database.Path);

        if (string.IsNullOrWhiteSpace(folder))
        {
            return;
        }

        Status = Localizer.Instance.Get("Train.LoadingProjects");

        try
        {
            IReadOnlyList<TinyLabelProjectSummary> projects = await _catalog
                .ListAsync(StorageLocation.FromPath(folder), ct)
                .ConfigureAwait(true);

            Projects.Clear();

            foreach (TinyLabelProjectSummary project in projects)
            {
                Projects.Add(new ProjectOption(
                    project.Id, project.Name, project.ImageCount, project.LabeledCount));
            }

            if (Projects.Count == 0)
            {
                // 库里没有项目就不提示，状态回到就绪即可
                if (!IsTraining)
                {
                    Status = Localizer.Instance.Get("Train.Ready");
                }

                _logger.LogInformation("库里没有标注项目");
                return;
            }

            if (!IsTraining)
            {
                Status = Localizer.Instance.Get("Train.Ready");
            }

            _logger.LogInformation("库里有 {Count} 个标注项目", Projects.Count);
        }
        catch (Exception error)
        {
            _logger.LogError(error, "读取标注项目失败");
            Status = Localizer.Instance.Format("Train.OpenProjectFailed", error.Message);
            AppendLog(Status);
        }
    }

    /// <summary>打开库里的一个项目当训练数据集：数据目录 + 项目 Id 交给统一解析入口。</summary>
    public async Task OpenProjectAsync(ProjectOption? project, CancellationToken ct = default)
    {
        if (project is null)
        {
            return;
        }

        string? folder = Path.GetDirectoryName(_database.Path);

        if (string.IsNullOrWhiteSpace(folder))
        {
            return;
        }

        Status = Localizer.Instance.Get("Train.Opening");

        try
        {
            DatasetOpenResult opened = await _resolver
                .OpenAsync(
                    StorageLocation.FromPath(folder),
                    DatasetFormat.TinyLabelDb,
                    project.Id.ToString(CultureInfo.InvariantCulture),
                    ct)
                .ConfigureAwait(true);

            await ApplyOpenedAsync(opened, $"{project.Name} · #{project.Id}", ct, project.Id)
                .ConfigureAwait(true);
        }
        catch (Exception error)
        {
            _logger.LogError(error, "打开标注项目失败");
            Status = Localizer.Instance.Format("Train.OpenProjectFailed", error.Message);
            AppendLog(Status);
        }
    }

    /// <summary>
    /// 列出标注端当前库（<c>label2train.db</c>）里的项目：训练直接用标注库，不再让用户挑库文件。
    /// 用独立只读连接查，避免和标注页的工作连接互相影响。
    /// </summary>
    public async Task<IReadOnlyList<TinyLabelProjectSummary>> ListLibraryProjectsAsync(CancellationToken ct = default)
    {
        try
        {
            return await Task.Run(
                    () => TinyLabelProjectCatalog.ReadFile(_database.Path, _logger, ct),
                    ct)
                .ConfigureAwait(true);
        }
        catch (Exception error)
        {
            _logger.LogError(error, "读取标注项目失败");
            Status = Localizer.Instance.Format("Train.OpenProjectFailed", error.Message);
            AppendLog(Status);
            return [];
        }
    }

    /// <summary>训练要用的库：固定为标注端当前库。</summary>
    public string LibraryFile => _database.Path;

    /// <summary>上次在标注库里选来训练的项目 Id；0 表示没记过。</summary>
    public long LastLibraryProjectId => _settings.LastTrainLibraryProjectId;

    /// <summary>当前数据集所属项目 Id；0 表示还没选项目（「训练目标设置」据此判断要不要重新拉数据）。</summary>
    public long ProjectId => _projectId;

    /// <summary>打开标注库里的某个项目当训练数据集，并记住项目 Id（下次启动自动恢复）。</summary>
    public async Task OpenLibraryProjectAsync(
        long projectId,
        string source,
        CancellationToken ct = default)
    {
        string databaseFile = _database.Path;
        string? folder = Path.GetDirectoryName(databaseFile);

        if (string.IsNullOrWhiteSpace(folder))
        {
            return;
        }

        Status = Localizer.Instance.Get("Train.Opening");

        try
        {
            DatasetOpenResult opened = await _resolver
                .OpenAsync(
                    StorageLocation.FromPath(folder),
                    DatasetFormat.TinyLabelDb,
                    Path.GetFileName(databaseFile),
                    projectId.ToString(CultureInfo.InvariantCulture),
                    ct)
                .ConfigureAwait(true);

            await ApplyOpenedAsync(opened, source, ct, projectId).ConfigureAwait(true);

            _settings.LastTrainLibraryProjectId = projectId;
            SaveSettings();
        }
        catch (Exception error)
        {
            _logger.LogError(error, "打开标注项目失败");
            Status = Localizer.Instance.Format("Train.OpenProjectFailed", error.Message);
            AppendLog(Status);
        }
    }

    /// <summary>
    /// 打开成功后统一收尾：换掉旧数据集、算摘要与统计、记类别。
    /// <paramref name="projectId"/> 是数据集所属的标注项目，run 目录与导出目录都跟着它走。
    /// </summary>
    private async Task ApplyOpenedAsync(
        DatasetOpenResult opened,
        string source,
        CancellationToken ct,
        long projectId = 0)
    {
        (_store as IDisposable)?.Dispose();
        _store = opened.Store;
        _projectId = projectId;

        // 换项目就把 run 目录切到该项目下，产物不会攒在共用的 runs 里分不清是谁的。
        OutputDirectory = RunDirectoryFor(projectId);

        DatasetPath = source;
        DatasetSummary = $"{opened.Store.Format.ToDisplayName()} · split {string.Join(" / ", opened.Store.Splits)} · " +
                         Localizer.Instance.Format("Train.Classes", opened.Store.Classes.Count);

        int images = 0;
        int boxes = 0;

        foreach (string split in opened.Store.Splits)
        {
            IReadOnlyList<DatasetSample> samples = await opened.Store
                .OpenSplit(split)
                .LoadSamplesAsync(ct)
                .ConfigureAwait(true);

            images += samples.Count;
            boxes += samples.Sum(sample => sample.Boxes.Count);
        }

        Classes = [.. opened.Store.Classes.Names];
        _imageCount = images;
        DatasetStats = Localizer.Instance.Format("Train.DatasetStats", images, boxes);
        Status = images > 0
            ? Localizer.Instance.Format("Train.DatasetOpened", DatasetStats)
            : Localizer.Instance.Get("Train.DatasetEmpty");
        OnPropertyChanged(nameof(HasDataset));
        OnPropertyChanged(nameof(CanStart));
    }

    // ── 训练 ──────────────────────────────────────────────────────────────────

    /// <summary>
    /// 当前训练配置的产物名：<c>yolov8_s</c> / <c>yolov11_m</c>；RF-DETR 没有规模字母，就叫 <c>rfdetr</c>。
    /// run 目录与导出的模型文件名都用它，一个「模型 + 规模」一套产物。
    /// </summary>
    public string ModelKey
    {
        get
        {
            string family = SelectedModel?.Id switch
            {
                "v11" => "yolov11",
                "rfdetr" => "rfdetr",
                _ => "yolov8",
            };

            return IsScaleEnabled ? $"{family}_{SelectedScale?.Id ?? "n"}" : family;
        }
    }

    /// <summary>
    /// 训练 run 目录：<c>projects/{id}/models/{模型名_规模}</c>，例如
    /// <c>data/projects/2/models/yolov8_s</c>。权重、优化器状态、train.json 都写在这里，
    /// 导出时从上一层的 <c>models</c> 取 <c>best.safetensors</c>，落成 <c>yolov8_s.onnx</c>。
    /// 还没选项目时退回 <c>{库根}/runs</c>，库路径异常时再用数据目录兜底。
    /// </summary>
    private string RunDirectoryFor(long projectId)
    {
        string? root = Path.GetDirectoryName(_database.Path);

        if (string.IsNullOrWhiteSpace(root) || projectId <= 0)
        {
            return string.IsNullOrWhiteSpace(root)
                ? Path.Combine(AppDataPaths.DataDirectory, "runs")
                : Path.Combine(root, "runs");
        }

        return Path.Combine(
            TinyLabelWorkspace.ModelDirectory(StorageLocation.FromPath(root), projectId),
            ModelKey);
    }

    /// <summary>
    /// 模型 / 规模下拉一变就重算输出目录：换家族或换规模等于换一套产物，
    /// 新权重不能写进旧模型目录。训练中不动——跑着的任务已经拿着旧路径。
    /// </summary>
    private void SyncOutputDirectory()
    {
        if (IsTraining)
        {
            return;
        }

        OutputDirectory = RunDirectoryFor(_projectId);
    }

    /// <summary>
    /// 导出目的地：训练数据来自标注库里的项目时进 <c>projects/{id}/models</c>（模型跟着项目走），
    /// 还没选项目时退回 <c>{输出目录}/export</c>。
    /// YOLO / RF-DETR / TinyLabel 产物都走同一个目录，只是文件名由架构决定。
    /// </summary>
    public string ExportDirectory
    {
        get
        {
            string? folder = Path.GetDirectoryName(_database.Path);

            if (_projectId > 0 && !string.IsNullOrWhiteSpace(folder))
            {
                return TinyLabelWorkspace.ModelDirectory(StorageLocation.FromPath(folder), _projectId);
            }

            return Path.Combine(OutputDirectory, "export");
        }
    }

    /// <summary>
    /// 检测只用当前训练配置的模型：<c>models/{模型名}_{规模}.onnx</c>，对应哪个文件由
    /// <see cref="ModelKey"/> 决定。不做跨规模 / 跨家族回退——ONNX 的输出层跟网络结构绑定，
    /// 拿别的权重顶上去只会得到乱框或直接崩，比「提示缺模型」糟得多。
    /// </summary>
    public string ResolveDetectModelPath()
    {
        string path = Path.Combine(ExportDirectory, ModelKey + ".onnx");

        return File.Exists(path) ? path : string.Empty;
    }

    /// <summary>当前训练配置还没导出模型时给用户看的话：缺哪个文件、怎么生成。</summary>
    public string DescribeMissingDetectModel() => Loc.Format(
        "Train.DetectModelMissing",
        ModelText,
        ModelKey + ".onnx");

    /// <summary>
    /// 导出当前训练产物到 <see cref="ExportDirectory"/>：<c>{模型名}_{规模}.safetensors</c> +
    /// <c>{模型名}_{规模}.json</c>，<paramref name="withOnnx"/> 为真时再导出 <c>{模型名}_{规模}.onnx</c>；
    /// 文件名由权重里的 architecture 决定（yolov8_s / yolov11_m / rfdetr），不是读界面下拉，
    /// 免得训练完改了下拉就把 A 的权重导成 B 的名字。导出的 ONNX 之后直接给「检测」用。
    /// 返回是否成功（训练结束 / 用户停止后的自动导出据此决定状态行怎么写）。
    /// </summary>
    public async Task<bool> ExportAsync(bool withOnnx, CancellationToken ct = default)
    {
        Status = Loc.Get("Train.Exporting");

        try
        {
            string target = ExportDirectory;

            TrainingExportResult result = await Task
                .Run(() => _exporter.Export(OutputDirectory, target, withOnnx), ct)
                .ConfigureAwait(true);

            if (result.OnnxPath is { } onnx)
            {
                LastOnnxPath = onnx;

                // 检测直接拿这个文件；AI 标注也从这个设置取模型，重启不用再挑
                _settings.AutoLabelModelPath = onnx;
                SaveSettings();
            }

            Status = result.OnnxPath is null
                ? Loc.Format("Train.ExportWeightsDone", result.SafetensorsPath)
                : Loc.Format("Train.ExportOnnxDone", result.OnnxPath);

            AppendLog(Status);
            return true;
        }
        catch (Exception error)
        {
            _logger.LogError(error, "导出模型失败");
            Status = Loc.Format("Train.ExportFailed", error.Message);
            AppendLog(Status);
            return false;
        }
    }

    /// <summary>
    /// 训练结束或用户停止后把 best 导到项目目录（含 ONNX），检测/AI 标注立刻能用；
    /// 一轮都还没评估过（没有 best.safetensors）时静默跳过。
    /// </summary>
    private async Task AutoExportAsync(CancellationToken ct)
    {
        if (!File.Exists(Path.Combine(OutputDirectory, TrainingExportService.WeightsFileName)))
        {
            _logger.LogInformation("输出目录里没有 {Weights}，跳过自动导出", TrainingExportService.WeightsFileName);
            return;
        }

        if (await ExportAsync(withOnnx: true, ct).ConfigureAwait(true))
        {
            Status = Loc.Format("Train.AutoExportDone", ExportDirectory);
            AppendLog(Status);
        }
    }

    public async Task StartTrainingAsync()
    {
        if (IsTraining || _store is null)
        {
            Status = Localizer.Instance.Get(_store is null ? "Train.NeedDataset" : "Train.AlreadyRunning");
            return;
        }

        _training = new CancellationTokenSource();
        _control = new TrainControl(_training.Token);
        _control.PausedChanged += OnPausedChanged;

        IsTraining = true;
        IsPaused = false;
        Progress = 0;
        ClearChart();

        // 规模只对 YOLO 有意义；RF-DETR 走自己的结构，不拼这个后缀
        string scale = IsScaleEnabled ? SelectedScale?.Id ?? "n" : string.Empty;
        RunInfo = $"epochs {Epochs:0} · batch {BatchSize:0} · imgsz {ImageSize:0}"
            + (scale.Length > 0 ? $" · scale {scale}" : string.Empty);

        Status = Localizer.Instance.Get("Train.Training");
        AppendLog(Localizer.Instance.Format("Train.Started", DatasetPath));

        // 进度与日志回调都在训练线程上触发：Progress<T> 会回到创建它的 UI 线程。
        var progress = new Progress<TrainingProgress>(item =>
        {
            Progress = item.Fraction;
            PushEpoch(item.Loss, item.Map50, item.Map50_95);
            Status = $"epoch {item.Epoch + 1}/{item.TotalEpochs} · loss {item.Loss:F4} · mAP50 {item.Map50:F3}";
        });

        IProgress<string> log = new Progress<string>(AppendLog);

        try
        {
            // 设备不在这里传：训练侧用 DeviceFactory.Default（启动时探测过，有 CUDA 就用 CUDA）
            var options = new TrainingRunOptions(
                StorageLocation.Parse(OutputDirectory),
                Epochs: (int)Epochs,
                BatchSize: (int)BatchSize,
                ImageSize: (int)ImageSize,
                Scale: scale.Length > 0 ? scale : "n",
                Family: SelectedModel?.Id ?? "v8")
            {
                Progress = progress,
                Log = message => log.Report(message),
                Control = _control,
            };

            TrainingRunSummary summary = await Task
                .Run(() => _runner.TrainAsync(_store, options, _training.Token), _training.Token)
                .ConfigureAwait(true);

            Progress = 1;
            Status = summary.Summary;
            AppendLog(summary.Summary);

            foreach (string warning in summary.Warnings.Take(10))
            {
                AppendLog($"  ! {warning}");
            }

            // 训练结束自动导出：{模型名}_{规模}.onnx 落进项目目录 models/
            await AutoExportAsync(_training.Token).ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            Status = Localizer.Instance.Get("Train.Stopped");
            AppendLog(Status);

            // 用户主动停止也要出模型：runner 每轮评估都会留 best，把已有的那份导出去。
            // 这里必须传 None——_training.Token 已经取消，再传进去导出会立刻被中止。
            await AutoExportAsync(CancellationToken.None).ConfigureAwait(true);
        }
        catch (Exception error)
        {
            _logger.LogError(error, "训练失败");
            Status = Localizer.Instance.Format("Train.Failed", error.Message);
            AppendLog(Status);
        }
        finally
        {
            IsTraining = false;
            IsPaused = false;

            if (_control is { } control)
            {
                control.PausedChanged -= OnPausedChanged;
                control.Dispose();
                _control = null;
            }

            _training?.Dispose();
            _training = null;
        }
    }

    /// <summary>暂停：训练线程在下一个批次边界停住，参数、优化器、EMA 全部原样保留。</summary>
    public void PauseTraining()
    {
        if (!IsTraining || _control is null || _control.IsPaused)
        {
            return;
        }

        Status = Localizer.Instance.Get("Train.Pausing");
        _control.Pause();
    }

    /// <summary>继续：从暂停处接着跑，不重新加载权重。</summary>
    public void ResumeTraining()
    {
        if (!IsTraining || _control is null || !_control.IsPaused)
        {
            return;
        }

        Status = Localizer.Instance.Get("Train.Resuming");
        _control.Resume();
    }

    /// <summary>结束训练：取消令牌，训练线程在批次边界退出。</summary>
    public void StopTraining()
    {
        if (!IsTraining || _training is null)
        {
            return;
        }

        // 暂停状态下也允许结束：先放开暂停，否则训练线程听不到取消。
        _control?.Resume();

        _training.Cancel();
        Status = Localizer.Instance.Get("Train.Stopping");
    }

    private void OnPausedChanged(bool paused)
    {
        IsPaused = paused;

        if (paused)
        {
            Status = Localizer.Instance.Get("Train.Paused");
        }
    }

    /// <summary>给界面用：把提示信息同时写进状态行与日志。</summary>
    public void Report(string message)
    {
        Status = message;
        AppendLog(message);
    }

    /// <summary>追加一轮指标（进度回调在 UI 线程上触发）。</summary>
    public void PushEpoch(double loss, double map50, double map5095)
    {
        double epoch = _map50Series.Points.Count + 1;

        _map50Series.Points.Add(new DataPoint(epoch, map50));
        _map5095Series.Points.Add(new DataPoint(epoch, map5095));
        _lossSeries.Points.Add(new DataPoint(epoch, loss));

        // 范围给实数而不是 NaN（NaN 会让 OxyPlot 算刻度时崩）：epoch 轴跟着轮数走，
        // loss 轴只往上放，loss 掉下去时曲线仍在框里
        _epochAxis.Maximum = Math.Max(10d, epoch + 1);

        if (double.IsFinite(loss))
        {
            _lossAxis.Maximum = Math.Max(_lossAxis.Maximum, loss * 1.2);
        }

        // mAP 早期经常只有 0.00x：轴固定 0~1 的话线贴着底边，看着就像「没有数据」。
        // 按当前最好成绩把轴放大一点（至少留 0.1，封顶 1），小幅进步也能看见
        double bestMap = Math.Max(double.IsFinite(map50) ? map50 : 0, double.IsFinite(map5095) ? map5095 : 0);

        if (bestMap > 0)
        {
            _mapAxis.Maximum = Math.Max(0.1, Math.Min(1d, Math.Ceiling(bestMap * 10) / 10 * 1.2));
        }

        HasChartData = true;
        Plot.InvalidatePlot(updateData: true);
    }

    /// <summary>开始新一轮训练前清空曲线。</summary>
    public void ClearChart()
    {
        _map50Series.Points.Clear();
        _map5095Series.Points.Clear();
        _lossSeries.Points.Clear();

        _epochAxis.Maximum = 10;
        _mapAxis.Maximum = 1;
        _lossAxis.Maximum = 1;
        HasChartData = false;
        Plot.InvalidatePlot(updateData: true);
    }

    private static LinearAxis CreateAxis(
        string title,
        AxisPosition position,
        double minimum = double.NaN,
        double maximum = double.NaN,
        double minimumStep = double.NaN)
        => new()
        {
            Key = title,
            Title = title,
            Position = position,
            // 兜底：谁忘了传范围也不留 NaN，默认给 0~1
            Minimum = double.IsNaN(minimum) ? 0 : minimum,
            Maximum = double.IsNaN(maximum) ? 1 : maximum,
            // 步长下限必须是 0（Axis 的默认值），不能是 NaN：OxyPlot 结尾会做
            // ActualMajorStep = Math.Max(算出来的步长, MinimumMajorStep)，NaN 会把它整成 NaN，
            // 随后 CreateTickValues 里 Math.Sign(NaN) 抛 ArithmeticException，整张图不渲染
            MinimumMajorStep = double.IsNaN(minimumStep) ? 0 : minimumStep,
            MinimumMinorStep = 0,
            MajorGridlineStyle = LineStyle.Solid,
            MajorGridlineColor = OxyColor.Parse(GridColor),
            MinorGridlineStyle = LineStyle.None,
            AxislineColor = OxyColor.Parse(GridColor),
            TicklineColor = OxyColor.Parse(AxisTextColor),
            TitleColor = OxyColor.Parse(AxisTextColor),
        };

    private static LineSeries CreateSeries(string title, string color, string yAxisKey)
        => new()
        {
            Title = title,
            Color = OxyColor.Parse(color),
            // 2px 在低分屏上细得几乎看不见，加粗到 2.5 让三条线更醒目
            StrokeThickness = 2.5,
            MarkerType = MarkerType.None,
            XAxisKey = "epoch",
            YAxisKey = yAxisKey,
        };

    // ── 内部 ──────────────────────────────────────────────────────────────────

    private void SaveSettings()
    {
        try
        {
            _settingsStore.Save(_settings);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            _logger.LogWarning(error, "保存设置失败");
        }
    }

    private void AppendLog(string text)
        => LogText = string.IsNullOrEmpty(LogText) ? text : $"{LogText}{Environment.NewLine}{text}";
}

/// <summary>训练模型家族下拉的一行；Id 直接就是 <c>TrainingRunOptions.Family</c>。</summary>
public sealed record ModelOption(string Id, string Name);

/// <summary>模型规模下拉的一行；Id 直接就是 <c>TrainingRunOptions.Scale</c>（n / s / m / l / x）。</summary>
public sealed record ScaleOption(string Id, string Name);

/// <summary>「标注项目」下拉的一行：项目 Id 就是打开数据集时的 datasetId。</summary>
public sealed record ProjectOption(long Id, string Name, int ImageCount, int LabeledCount)
{
    /// <summary>下拉里显示「项目名 · 已标/总数」。</summary>
    public string DisplayName => $"{Name} · {LabeledCount}/{ImageCount}";
}
