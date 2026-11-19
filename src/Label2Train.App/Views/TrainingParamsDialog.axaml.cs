using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;
using Label2Train.Core.Localization;
using Label2Train.Train.Training;

namespace Label2Train.App.Views;

/// <summary>「训练参数」对话框的结果；取消时对话框返回 <c>null</c>。</summary>
public sealed record TrainParams(int Epochs, int BatchSize, int ImageSize);

/// <summary>
/// 训练参数对话框：epochs / batch / imgsz 都在这里改，每个都带一句含义说明。
/// 主界面只读展示这三个值，避免直接手滑把输入框改坏。
/// <para>
/// 「缺省 YOLO / 缺省 RF-DETR」两个按钮一键回到推荐值：两者对 imgsz 的要求不一样
/// （YOLO 32 的倍数随便取，RF-DETR 必须 112 的倍数），手填很容易训完才发现导不出 ONNX。
/// </para>
/// </summary>
public partial class TrainingParamsDialog : Window
{
    private static readonly IBrush WarningBrush = new SolidColorBrush(Color.Parse("#D03A3A"));

    private readonly bool _isDetr;
    private readonly IBrush? _hintBrush;

    /// <summary>XAML 预览用：按 YOLO 缺省填。</summary>
    public TrainingParamsDialog()
    {
        InitializeComponent();

        _isDetr = false;
        _hintBrush = ImageSizeHint.Foreground;

        ImageSizeBox.ValueChanged += OnImageSizeValueChanged;
        ApplyPreset(TrainingPresets.Yolo);
    }

    public TrainingParamsDialog(decimal epochs, decimal batchSize, decimal imageSize, string family)
    {
        InitializeComponent();

        _isDetr = TrainingPresets.IsDetrFamily(family);
        _hintBrush = ImageSizeHint.Foreground;

        EpochsBox.Value = epochs;
        BatchBox.Value = batchSize;
        ImageSizeBox.Value = imageSize;

        ImageSizeBox.ValueChanged += OnImageSizeValueChanged;
        UpdateImageSizeHint();
    }

    private static Localizer Loc => Localizer.Instance;

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
        if (!_isDetr)
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
        int imageSize = (int)(ImageSizeBox.Value ?? TrainingPresets.Yolo.ImageSize);

        // RF-DETR 只接受 112 的倍数：直接就近纠正，别让非法值流回主界面
        if (_isDetr)
        {
            imageSize = TrainingPresets.NearestValidDetrImageSize(imageSize);
        }

        Close(new TrainParams(
            (int)(EpochsBox.Value ?? TrainingPresets.Yolo.Epochs),
            (int)(BatchBox.Value ?? TrainingPresets.Yolo.BatchSize),
            imageSize));
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);
}
