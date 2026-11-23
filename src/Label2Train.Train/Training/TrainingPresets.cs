using Label2Train.Train.Data.Images;

namespace Label2Train.Train.Training;

/// <summary>
/// 训练参数的缺省值与合法性。参数对话框的「缺省 YOLO / 缺省 RF-DETR」按钮、
/// 「开始训练」的可用性判断、RF-DETR 的 ONNX 导出校验都从这里取，
/// 免得同一个数字（560、112）在三处各写一遍，改一处漏两处。
/// </summary>
public static class TrainingPresets
{
    /// <summary>YOLO 缺省：640 是官方惯例，也是 32 的倍数（三档 stride 都整除）。</summary>
    public static readonly (int Epochs, int BatchSize, int ImageSize) Yolo = (100, 4, 640);

    /// <summary>
    /// RF-DETR 缺省：560。DINOv2 的 patch 是 14、检测头要求 16 的倍数，
    /// 合法边长必须是 lcm(14, 16) = 112 的倍数，560 = 40×14 = 35×16，正好是官方尺寸。
    /// </summary>
    public static readonly (int Epochs, int BatchSize, int ImageSize) RfDetr = (100, 4, 560);

    /// <summary>
    /// DETR 系输入边长的步长：DINOv2 的 patch 14 与检测头下采样 16 的公倍数。
    /// 界面校验、训练、ONNX 导出全都用它，免得三处各写一个数（训练侧曾单独要求 224，
    /// 界面上填的 560 根本到不了训练，日志里永远是 224）。
    /// </summary>
    public const int DetrImageSizeStride = 112;

    /// <summary>一张 ONNX 图能不能建出来，就看边长是不是 112 的倍数。</summary>
    public static bool IsValidDetrImageSize(int imageSize) =>
        imageSize >= DetrImageSizeStride && imageSize % DetrImageSizeStride == 0;

    /// <summary>
    /// RF-DETR 的 query 数缺省值。官方权重是 300 行（COCO 的量级），但那意味着
    /// 「几百个 query 里只有几个正样本」，本机实测在每图 4~5 个目标的数据上比 50 慢一个量级。
    /// </summary>
    public const int DetrQueries = 50;

    /// <summary>
    /// 下拉里可选的 query 数。上限 300 = 官方权重的行数：再往上官方 query_feat 不够切，
    /// 多出来的 query 只能是随机初始化，等于把「有预训练」这个前提废掉。
    /// </summary>
    public static readonly int[] DetrQueryChoices = [50, 100, 150, 300];

    /// <summary>
    /// query 数兜底：至少 1 个，最多 300 个（见 <see cref="DetrQueryChoices"/>）。
    /// 界面只给合法档位，这里挡的是手改代码 / 旧配置。
    /// </summary>
    public static int ResolveDetrQueries(int queries) => Math.Clamp(queries, 1, 300);

    /// <summary>
    /// RF-DETR 训练集的预处理 / 增强。之前训练集与验证集共用「只 letterbox」的参数，
    /// 几十张图几十轮就被完全记住，验证集先到顶——这是继续加轮数的天花板。
    /// <para>
    /// 这里按官方配方开 hflip 0.5、随机缩放 ±0.25、平移 ±0.1、HSV(0.015 / 0.7 / 0.4)，
    /// 但**不开 mosaic**：官方 RF-DETR 配方里没有 mosaic，小数据集拼贴后目标常只剩半截。
    /// </para>
    /// </summary>
    public static YoloPreprocessOptions DetrTrainPreprocess(int imageSize) => new()
    {
        Width = imageSize,
        Height = imageSize,
        SizeStride = DetrImageSizeStride,
        HorizontalFlipProbability = 0.5f,
        ScaleGain = 0.25f,
        TranslateGain = 0.1f,
        HueGain = 0.015f,
        SaturationGain = 0.7f,
        ValueGain = 0.4f,
    };

    /// <summary>合法值原样返回，非法时就近纠正（训练入口兜底用，界面已经纠正过一次）。</summary>
    public static int ResolveDetrImageSize(int imageSize) =>
        IsValidDetrImageSize(imageSize) ? imageSize : NearestValidDetrImageSize(imageSize);

    /// <summary>就近换成合法边长；低于一档时给最小的一档（112）。</summary>
    public static int NearestValidDetrImageSize(int imageSize)
    {
        int steps = (int)Math.Round(imageSize / (double)DetrImageSizeStride, MidpointRounding.AwayFromZero);
        return Math.Max(1, steps) * DetrImageSizeStride;
    }

    /// <summary>家族 id（<c>v8</c> / <c>v11</c> / <c>rfdetr</c> / <c>simple-detr</c>）是不是 DETR 系。</summary>
    public static bool IsDetrFamily(string? family)
    {
        if (string.IsNullOrWhiteSpace(family))
        {
            return false;
        }

        string text = family.Trim();

        return text.StartsWith("rfdetr", StringComparison.OrdinalIgnoreCase)
            || text.StartsWith("simple-detr", StringComparison.OrdinalIgnoreCase)
            || text.StartsWith("detr", StringComparison.OrdinalIgnoreCase);
    }
}
