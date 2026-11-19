namespace Label2Train.Train.Data.Images;

/// <summary>YOLO 检测预处理参数：letterbox 尺寸 + 训练增强。</summary>
/// <remarks>
/// 增强概率 / 增益都取 <c>[0,1]</c>：
/// <see cref="HorizontalFlipProbability"/> 是命中概率；
/// <see cref="HueGain"/> 是色相扰动幅度（<c>0.015</c> 表示 ±2.7°，与 ultralytics 默认 <c>hsv_h</c> 一致），
/// <see cref="SaturationGain"/> / <see cref="ValueGain"/> 是饱和度 / 明度的相对增益。
/// </remarks>
public sealed record YoloPreprocessOptions
{
    /// <summary>网络输入宽（必须是 32 的倍数）。</summary>
    public int Width { get; init; } = 640;

    /// <summary>网络输入高（必须是 32 的倍数）。</summary>
    public int Height { get; init; } = 640;

    /// <summary>letterbox 填充灰度值（ultralytics 默认 114）。</summary>
    public byte PadValue { get; init; } = 114;

    /// <summary>水平翻转概率（验证 / 推理时保持 0）。</summary>
    public float HorizontalFlipProbability { get; init; }

    /// <summary>mosaic（4 图拼贴）概率；训练常开，验证 / 推理保持 0。</summary>
    public float MosaicProbability { get; init; }

    /// <summary>
    /// 随机缩放增益：先按 <c>1 ± ScaleGain</c> 把图 letterbox 到变尺寸，再贴回网络尺寸画布。
    /// <c>0</c> 表示不缩放（验证 / 推理必须为 0，否则与导出端不一致）。
    /// </summary>
    public float ScaleGain { get; init; }

    /// <summary>
    /// 随机平移增益：内容在画布上最多偏移 <c>TranslateGain × 边长</c> 像素。
    /// <c>0</c> 表示居中（验证 / 推理必须为 0）。
    /// </summary>
    public float TranslateGain { get; init; }

    /// <summary>色相扰动增益，<c>0</c> 表示不扰动。</summary>
    public float HueGain { get; init; }

    /// <summary>饱和度扰动增益。</summary>
    public float SaturationGain { get; init; }

    /// <summary>明度扰动增益。</summary>
    public float ValueGain { get; init; }

    /// <summary>
    /// 训练用增强：hflip 0.5、mosaic 0.5、scale ±0.5、translate ±0.1、HSV(0.015 / 0.7 / 0.4)。
    /// 小数据集全靠这几项补多样性，不增强的话模型只学会「框住大概位置」。
    /// </summary>
    public static YoloPreprocessOptions TrainDefault(int imageSize) => new()
    {
        Width = imageSize,
        Height = imageSize,
        HorizontalFlipProbability = 0.5f,
        MosaicProbability = 0.5f,
        ScaleGain = 0.5f,
        TranslateGain = 0.1f,
        HueGain = 0.015f,
        SaturationGain = 0.7f,
        ValueGain = 0.4f,
    };

    /// <summary>验证 / 推理用：只 letterbox，不做任何随机增强（必须与导出端逐像素一致）。</summary>
    public static YoloPreprocessOptions InferenceDefault(int imageSize) => new()
    {
        Width = imageSize,
        Height = imageSize,
    };

    /// <summary>校验并返回自身（非法值直接抛错，不做静默修正）。</summary>
    public YoloPreprocessOptions Validate()
    {
        const int stride = 32;

        if (Width <= 0 || Height <= 0 || Width % stride != 0 || Height % stride != 0)
        {
            throw new ArgumentException(
                $"预处理尺寸必须是 {stride} 的正整数倍（P3/P4/P5 下采样），实际 {Width}x{Height}。",
                nameof(Width));
        }

        ValidateProbability(HorizontalFlipProbability, nameof(HorizontalFlipProbability));
        ValidateProbability(MosaicProbability, nameof(MosaicProbability));
        ValidateGain(ScaleGain, nameof(ScaleGain));
        ValidateGain(TranslateGain, nameof(TranslateGain));
        ValidateGain(HueGain, nameof(HueGain));
        ValidateGain(SaturationGain, nameof(SaturationGain));
        ValidateGain(ValueGain, nameof(ValueGain));

        return this;
    }

    private static void ValidateProbability(float value, string name)
    {
        if (value is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(name, value, "概率必须在 [0,1] 内。");
    }

    private static void ValidateGain(float value, string name)
    {
        if (value is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(name, value, "增益必须在 [0,1] 内。");
    }
}
