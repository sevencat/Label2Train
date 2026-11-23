using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;
using Label2Train.App.ViewModels;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Localization;
using Label2Train.Train.Training;

namespace Label2Train.App.Views;

/// <summary>「训练目标设置」的结果；取消时对话框返回 <c>null</c>。</summary>
public sealed record TrainingTarget(
    TinyLabelProjectSummary Project,
    int Epochs,
    int BatchSize,
    int ImageSize,
    string FamilyId,
    string ScaleId,
    bool Augment = true);

/// <summary>项目下拉的一行：<see cref="Display"/> 是给用户看的「名称 · #id」。</summary>
public sealed record ProjectChoice(long Id, string Display, TinyLabelProjectSummary Project);

/// <summary>
/// 训练目标设置：项目、模型家族、模型规模、epochs / batch / imgsz 一次配好。
/// 主界面只读展示这些值，避免训练中手滑改坏；入口只有一个「训练目标设置…」按钮。
/// <para>
/// 「缺省 YOLO / 缺省 RF-DETR」两个按钮一键回到推荐值：两者对 imgsz 的要求不一样
/// （YOLO 32 的倍数随便取，RF-DETR 必须 112 的倍数），手填很容易训完才发现导不出 ONNX。
/// </para>
/// </summary>
public partial class TrainingTargetDialog : Window
{
    private static readonly IBrush WarningBrush = new SolidColorBrush(Color.Parse("#D03A3A"));

    private readonly IBrush? _hintBrush;

    /// <summary>XAML 预览用：空项目列表 + YOLO 缺省。</summary>
    public TrainingTargetDialog()
        : this([], 0, [new ModelOption("v8", "YOLOv8")], [new ScaleOption("n", "n · nano")],
            "v8", "n", TrainingPresets.Yolo.Epochs, TrainingPresets.Yolo.BatchSize,
            TrainingPresets.Yolo.ImageSize, augment: true)
    {
    }

    public TrainingTargetDialog(
        IReadOnlyList<TinyLabelProjectSummary> projects,
        long selectedProjectId,
        IReadOnlyList<ModelOption> families,
        IReadOnlyList<ScaleOption> scales,
        string familyId,
        string scaleId,
        decimal epochs,
        decimal batchSize,
        decimal imageSize,
        bool augment = true)
    {
        InitializeComponent();

        _hintBrush = ImageSizeHint.Foreground;

        List<ProjectChoice> choices =
        [
            .. projects.Select(project => new ProjectChoice(
                project.Id,
                Loc.Format("Train.ProjectRow", project.Name, project.Id, project.ImageCount, project.LabeledCount),
                project)),
        ];

        ProjectBox.ItemsSource = choices;
        ProjectBox.SelectedItem = choices.FirstOrDefault(choice => choice.Id == selectedProjectId)
            ?? choices.FirstOrDefault();

        ModelBox.ItemsSource = families;
        ModelBox.SelectedItem = families.FirstOrDefault(model => model.Id == familyId)
            ?? families.FirstOrDefault();

        ScaleBox.ItemsSource = scales;
        ScaleBox.SelectedItem = scales.FirstOrDefault(scale => scale.Id == scaleId)
            ?? scales.FirstOrDefault();

        EpochsBox.Value = epochs;
        BatchBox.Value = batchSize;
        ImageSizeBox.Value = imageSize;
        AugmentBox.IsChecked = augment;

        ModelBox.SelectionChanged += OnModelSelectionChanged;
        ImageSizeBox.ValueChanged += OnImageSizeValueChanged;

        UpdateScaleState();
        UpdateImageSizeHint();
    }

    private static Localizer Loc => Localizer.Instance;

    /// <summary>当前选的家族是不是 RF-DETR：决定规模是否可选、imgsz 的整除要求。</summary>
    private bool IsDetr => TrainingPresets.IsDetrFamily((ModelBox.SelectedItem as ModelOption)?.Id);

    private void OnModelSelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        UpdateScaleState();
        UpdateImageSizeHint();
    }

    /// <summary>
    /// RF-DETR 的体量由 queries / 层数决定，规模下拉直接置灰；
    /// 增强开关也只有 RF-DETR 认（YOLO 的 hflip / mosaic 在它自己的 pipeline 里）。
    /// </summary>
    private void UpdateScaleState()
    {
        ScaleBox.IsEnabled = !IsDetr;
        AugmentBox.IsEnabled = IsDetr;
    }

    private void OnPresetYolo(object? sender, RoutedEventArgs e) => ApplyPreset(TrainingPresets.Yolo);

    private void OnPresetRfDetr(object? sender, RoutedEventArgs e) => ApplyPreset(TrainingPresets.RfDetr);

    private void ApplyPreset((int Epochs, int BatchSize, int ImageSize) preset)
    {
        EpochsBox.Value = preset.Epochs;
        BatchBox.Value = preset.BatchSize;
        ImageSizeBox.Value = preset.ImageSize;

        UpdateImageSizeHint();
    }

    private void OnImageSizeValueChanged(object? sender, NumericUpDownValueChangedEventArgs e) =>
        UpdateImageSizeHint();

    /// <summary>RF-DETR 家族下把 imgsz 的要求写在提示里；合法就说明白，非法就红字给出建议值。</summary>
    private void UpdateImageSizeHint()
    {
        if (!IsDetr)
        {
            ImageSizeHint.Text = Loc.Get("Train.ImageSizeHint");
            ImageSizeHint.Foreground = _hintBrush;

            return;
        }

        int size = (int)(ImageSizeBox.Value ?? 0m);

        if (TrainingPresets.IsValidDetrImageSize(size))
        {
            ImageSizeHint.Text = Loc.Format("Train.ImageSizeHintDetr", TrainingPresets.DetrImageSizeStride);
            ImageSizeHint.Foreground = _hintBrush;

            return;
        }

        ImageSizeHint.Text = Loc.Format(
            "Train.ImageSizeInvalidDetr",
            TrainingPresets.DetrImageSizeStride,
            TrainingPresets.NearestValidDetrImageSize(size));
        ImageSizeHint.Foreground = WarningBrush;
    }

    private void OnOk(object? sender, RoutedEventArgs e)
    {
        if (ProjectBox.SelectedItem is not ProjectChoice project)
        {
            // 库里一个项目都没有的话页面根本进不来，这里只是兜底，不关闭对话框
            ProjectBox.Focus();

            return;
        }

        int imageSize = (int)(ImageSizeBox.Value ?? TrainingPresets.Yolo.ImageSize);

        // RF-DETR 只接受 112 的倍数：直接就近纠正，别让非法值流回主界面
        if (IsDetr)
        {
            imageSize = TrainingPresets.NearestValidDetrImageSize(imageSize);
        }

        Close(new TrainingTarget(
            project.Project,
            (int)(EpochsBox.Value ?? TrainingPresets.Yolo.Epochs),
            (int)(BatchBox.Value ?? TrainingPresets.Yolo.BatchSize),
            imageSize,
            (ModelBox.SelectedItem as ModelOption)?.Id ?? "v8",
            (ScaleBox.SelectedItem as ScaleOption)?.Id ?? "n",
            AugmentBox.IsChecked ?? true));
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);
}
