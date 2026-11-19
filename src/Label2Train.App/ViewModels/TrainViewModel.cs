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

    /// <summary>当前数据集来自哪个标注项目；0 表示外部目录（没有项目目录可放模型）。</summary>
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
    /// 检测用的模型（界面上三个单选）：Id 只用于判断家族，<see cref="DetectModelOption.FilePrefix"/>
    /// 就是导出文件名前缀，选中哪个就去 models 目录取 <c>{前缀}.onnx</c>。
    /// </summary>
    public IReadOnlyList<DetectModelOption> DetectModels { get; } =
    [
        new DetectModelOption("yolov8", "YOLOv8", "yolov8"),
        new DetectModelOption("yolov11", "YOLOv11", "yolov11"),
        new DetectModelOption("rfdetr", "RF-DETR", "rfdetr"),
    ];

    /// <summary>检测用哪个模型推理。</summary>
    [ObservableProperty]
    private DetectModelOption _selectedDetectModel = null!;

    public bool IsDetectYolov8
    {
        get => SelectedDetectModel.Id == "yolov8";
        set
        {
            if (value)
            {
                SelectedDetectModel = DetectModels[0];
            }
        }
    }

    public bool IsDetectYolov11
    {
        get => SelectedDetectModel.Id == "yolov11";
        set
        {
            if (value)
            {
                SelectedDetectModel = DetectModels[1];
            }
        }
    }

    public bool IsDetectRfDetr
    {
        get => SelectedDetectModel.Id == "rfdetr";
        set
        {
            if (value)
            {
                SelectedDetectModel = DetectModels[2];
            }
        }
    }

    partial void OnSelectedDetectModelChanged(DetectModelOption value)
    {
        OnPropertyChanged(nameof(IsDetectYolov8));
        OnPropertyChanged(nameof(IsDetectYolov11));
        OnPropertyChanged(nameof(IsDetectRfDetr));
    }

    /// <summary>最近一次导出的 ONNX 路径（只用于展示与日志；检测走的是单选对应的前缀文件）。</summary>
    public string LastOnnxPath { get; private set; } = string.Empty;

    /// <summary>epochs / batch / imgsz 在界面上只读展示，改值走「修改」对话框。</summary>
    public string EpochsText => ((int)Epochs).ToString(CultureInfo.InvariantCulture);

    public string BatchSizeText => ((int)BatchSize).ToString(CultureInfo.InvariantCulture);

    public string ImageSizeText => ((int)ImageSize).ToString(CultureInfo.InvariantCulture);

    partial void OnEpochsChanged(decimal value) => OnPropertyChanged(nameof(EpochsText));

    partial void OnBatchSizeChanged(decimal value) => OnPropertyChanged(nameof(BatchSizeText));

    partial void OnImageSizeChanged(decimal value)
    {
        OnPropertyChanged(nameof(ImageSizeText));
        RaiseRunState();
    }

    partial void OnSelectedModelChanged(ModelOption? value)
    {
        RaiseRunState();
        ReportImageSizeIfBlocked();
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

    private readonly LinearAxis _epochAxis;
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
    /// 没选项目 / 没选输出路径时「开始训练」置灰：训练既要知道数据在哪，
    /// 也要知道权重往哪写，缺一个都会在跑起来之后才报错。
    /// imgsz 跟当前家族不兼容（RF-DETR 不是 112 的倍数）时也置灰，因为训完也导不出 ONNX。
    /// </summary>
    public bool CanStart => !IsTraining
        && _store is not null
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
        _selectedDetectModel = DetectModels[0];

        // 轴范围一律给实数：OxyPlot 2.x 在算刻度时会对 NaN 取符号，
        // 一旦某根轴的 Minimum/Maximum 是 NaN 就抛 ArithmeticException，整张图都画不出来
        _epochAxis = CreateAxis("epoch", AxisPosition.Bottom, minimum: 0, maximum: 10, minimumStep: 1);
        LinearAxis mapAxis = CreateAxis("mAP", AxisPosition.Left, minimum: 0, maximum: 1, minimumStep: 0.1);
        _lossAxis = CreateAxis("loss", AxisPosition.Right, minimum: 0, maximum: 1);

        _map50Series = CreateSeries("mAP50", "#2E7D32", mapAxis.Key);
        _map5095Series = CreateSeries("mAP50-95", "#6A1B9A", mapAxis.Key);
        _lossSeries = CreateSeries("loss", "#00796B", _lossAxis.Key);

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
        Plot.Axes.Add(mapAxis);
        Plot.Axes.Add(_lossAxis);

        Plot.Series.Add(_map50Series);
        Plot.Series.Add(_map5095Series);
        Plot.Series.Add(_lossSeries);

        if (!string.IsNullOrWhiteSpace(settings.LastTrainDataDirectory))
        {
            OutputDirectory = Path.Combine(settings.LastTrainDataDirectory, "runs");
        }

        logger.LogInformation("训练工作台已创建，等待选择数据集");

        LogText = string.Join(Environment.NewLine, logBuffer.Snapshot().Select(entry => entry.ToString()));
    }

    /// <summary>第一次切到训练页时跑一次：列库里的项目，再恢复上次训练的那个项目。</summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        if (_initialized)
        {
            return;
        }

        _initialized = true;

        await RefreshProjectsAsync(ct).ConfigureAwait(true);
        await RestoreLastLibraryAsync(ct).ConfigureAwait(true);
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
    /// 打开外部数据集目录。<paramref name="format"/> 为 <c>null</c> 时交给探测器自己判断，
    /// 指定了（YOLO / COCO / TinyLabel）就按该格式打开。
    /// </summary>
    public async Task OpenDatasetAsync(string path, DatasetFormat? format = null, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }

        Status = Localizer.Instance.Get("Train.Opening");

        try
        {
            DatasetOpenResult opened = await _resolver
                .OpenAsync(StorageLocation.Parse(path), format, datasetId: null, ct)
                .ConfigureAwait(true);

            await ApplyOpenedAsync(opened, path, ct).ConfigureAwait(true);

            _settings.LastTrainDataDirectory = Path.GetDirectoryName(path) ?? string.Empty;
            SaveSettings();
        }
        catch (Exception error)
        {
            _logger.LogError(error, "打开数据集失败");
            Status = Localizer.Instance.Format("Train.OpenFailed", error.Message);
            AppendLog(Status);
        }
    }

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

            _settings.LastTrainDataDirectory = folder;
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
            _settings.LastTrainDataDirectory = folder;
            SaveSettings();
        }
        catch (Exception error)
        {
            _logger.LogError(error, "打开标注项目失败");
            Status = Localizer.Instance.Format("Train.OpenProjectFailed", error.Message);
            AppendLog(Status);
        }
    }

    /// <summary>启动时把上次训练的标注项目重新打开；库固定是标注端当前库，项目没了就静默跳过。</summary>
    public async Task RestoreLastLibraryAsync(CancellationToken ct = default)
    {
        // 没记过项目就不自动打开，避免启动就替用户挑了数据集
        if (_settings.LastTrainLibraryProjectId <= 0)
        {
            return;
        }

        string database = _database.Path;

        if (!File.Exists(database))
        {
            return;
        }

        string? folder = Path.GetDirectoryName(database);

        if (string.IsNullOrWhiteSpace(folder))
        {
            return;
        }

        IReadOnlyList<TinyLabelProjectSummary> projects;

        try
        {
            projects = await _catalog.ListAsync(StorageLocation.FromPath(folder), ct).ConfigureAwait(true);
        }
        catch (Exception error)
        {
            // 启动恢复失败不打扰用户：状态留着，等他自己挑数据集
            _logger.LogWarning(error, "恢复上次的标注项目失败");
            return;
        }

        TinyLabelProjectSummary? project =
            projects.FirstOrDefault(item => item.Id == _settings.LastTrainLibraryProjectId)
            ?? projects.FirstOrDefault();

        if (project is null)
        {
            return;
        }

        _logger.LogInformation("启动恢复标注项目 {Project}（库 {Database}）", project.Name, database);

        await OpenLibraryProjectAsync(project.Id, $"{project.Name} · #{project.Id}", ct)
            .ConfigureAwait(true);
    }

    /// <summary>
    /// 打开成功后统一收尾：换掉旧数据集、算摘要与统计、记类别。
    /// <paramref name="projectId"/> 是数据集所属的标注项目（外部目录传 0），导出时据此决定模型落在哪。
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
        DatasetStats = Localizer.Instance.Format("Train.DatasetStats", images, boxes);
        Status = Localizer.Instance.Format("Train.DatasetOpened", DatasetStats);
        OnPropertyChanged(nameof(HasDataset));
        OnPropertyChanged(nameof(CanStart));
    }

    // ── 训练 ──────────────────────────────────────────────────────────────────

    /// <summary>
    /// 导出目的地：训练数据来自标注库里的项目时进 <c>projects/{id}/models</c>（模型跟着项目走），
    /// 外部目录数据集没有项目目录，退回 <c>{输出目录}/export</c>。
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
    /// 当前检测单选对应的 ONNX 路径，严格按前缀取：选 YOLOv8 就是 <c>yolov8.onnx</c>，
    /// 选 YOLOv11 就是 <c>yolov11.onnx</c>，选 RF-DETR 就是 <c>rfdetr.onnx</c>；找不到返回空串。
    /// 这里不做跨前缀回退——拿别的家族的权重去推理只会得到乱框或者直接崩，
    /// 比「提示缺模型」糟得多。
    /// </summary>
    public string ResolveDetectModelPath()
    {
        string directory = ExportDirectory;
        string preferred = Path.Combine(directory, SelectedDetectModel.FilePrefix + ".onnx");

        if (File.Exists(preferred))
        {
            return preferred;
        }

        // 旧版本导出统一叫 best.onnx，且只有 YOLO 一个家族；v11 的权重长什么样无从得知，
        // 所以只给 YOLOv8 兼容这一份
        if (SelectedDetectModel.Id == "yolov8")
        {
            string legacy = Path.Combine(directory, TrainingExportService.OnnxFileName);

            return File.Exists(legacy) ? legacy : string.Empty;
        }

        return string.Empty;
    }

    /// <summary>选的模型还没导出时给用户看的话：缺哪个文件、怎么生成。</summary>
    public string DescribeMissingDetectModel() => Loc.Format(
        "Train.DetectModelMissing",
        SelectedDetectModel.Name,
        SelectedDetectModel.FilePrefix + ".onnx");

    /// <summary>
    /// 导出当前训练产物到 <see cref="ExportDirectory"/>：<c>{模型名}.safetensors</c> +
    /// <c>{模型名}.json</c>，<paramref name="withOnnx"/> 为真时再导出一份 <c>{模型名}.onnx</c>；
    /// 文件名前缀按架构自动取（yolov8 / yolov11 / rfdetr），导出的 ONNX 之后直接给「检测」用。
    /// 返回是否成功（训练结束后的自动导出据此决定状态行怎么写）。
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

                // 检测单选跟着刚导出的模型走：三个模型共存时，用户不用自己去对文件名
                string prefix = Path.GetFileNameWithoutExtension(onnx);

                if (DetectModels.FirstOrDefault(model => model.FilePrefix == prefix) is { } matched)
                {
                    SelectedDetectModel = matched;
                }

                // AI 标注从这个设置取模型：导出后直接指向新模型，重启也不用再挑文件
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
    /// 训练一结束就把模型导到项目目录（含 ONNX），检测/AI 标注立刻能用；
    /// 摘要 runner 或干跑不写权重时静默跳过。
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
        RunInfo = $"epochs {Epochs:0} · batch {BatchSize:0} · imgsz {ImageSize:0}";

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

            // 训练完自动导出：best.safetensors + model.json + best.onnx 落到项目目录 models/
            await AutoExportAsync(_training.Token).ConfigureAwait(true);

            _settings.LastTrainDataDirectory = Path.GetDirectoryName(DatasetPath) ?? _settings.LastTrainDataDirectory;
            SaveSettings();
        }
        catch (OperationCanceledException)
        {
            Status = Localizer.Instance.Get("Train.Stopped");
            AppendLog(Status);
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
            StrokeThickness = 2,
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

/// <summary>「检测用哪个模型」的三个单选：<see cref="FilePrefix"/> 就是导出文件名前缀。</summary>
public sealed record DetectModelOption(string Id, string Name, string FilePrefix);

/// <summary>「标注项目」下拉的一行：项目 Id 就是打开数据集时的 datasetId。</summary>
public sealed record ProjectOption(long Id, string Name, int ImageCount, int LabeledCount)
{
    /// <summary>下拉里显示「项目名 · 已标/总数」。</summary>
    public string DisplayName => $"{Name} · {LabeledCount}/{ImageCount}";
}
