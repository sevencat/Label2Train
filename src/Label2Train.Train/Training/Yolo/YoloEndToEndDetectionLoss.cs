using Label2Train.Train.Models.Yolo;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Training.Yolo;

/// <summary>
/// YOLO26 的双头损失（官方 <c>E2EDetectLoss</c>）：一条 one-to-many 分支 + 一条 one-to-one 分支，
/// 两条分支各自按同一套 TAL 分配算 box / cls / dfl（reg_max = 1 时第三项退化成 L1），最后相加。
/// <para>
/// <c>prediction</c> 是模型前向的 one-to-many 原始输出 <c>[B, 4*nc+nc, N]</c>；
/// one-to-one 分支的原始输出由模型通过 extras（键 <see cref="OneToOneKey"/>）传进来，
/// 没有 extras 时退化成单头损失（训练器在模型不支持双头时就走这条路）。
/// </para>
/// <para>
/// top-k 与官方一致：one-to-many 用 10、one-to-one 用 1（<see cref="YoloLossOptions.TopK"/> 在这里被忽略），
/// 因此 one-to-one 分支每个 GT 只挑一个最优锚点，正是 YOLO26 免 NMS 的解码前提。
/// </para>
/// </summary>
public sealed class YoloEndToEndDetectionLoss : ILoss
{
    /// <summary>extras 里 one-to-one 分支原始输出的键名（与模型侧的字面量必须一致）。</summary>
    public const string OneToOneKey = YoloV26DetectionModel.OneToOneKey;

    private readonly YoloDetectionLoss _oneToMany;
    private readonly YoloDetectionLoss _oneToOne;

    /// <summary>按输入尺寸与类别数创建双头损失；<paramref name="regressionMax"/> 必须与模型一致（YOLO26 为 1）。</summary>
    public YoloEndToEndDetectionLoss(
        int imageWidth,
        int imageHeight,
        int classCount,
        int regressionMax = 1,
        YoloLossOptions? options = null,
        int oneToManyTopK = 10,
        int oneToOneTopK = 1)
    {
        if (oneToManyTopK <= 0)
            throw new ArgumentOutOfRangeException(nameof(oneToManyTopK), oneToManyTopK, "top-k 必须大于 0。");
        if (oneToOneTopK <= 0)
            throw new ArgumentOutOfRangeException(nameof(oneToOneTopK), oneToOneTopK, "top-k 必须大于 0。");

        YoloLossOptions baseOptions = options ?? new YoloLossOptions();

        _oneToMany = new YoloDetectionLoss(
            imageWidth,
            imageHeight,
            classCount,
            regressionMax,
            baseOptions with { TopK = oneToManyTopK });
        _oneToOne = new YoloDetectionLoss(
            imageWidth,
            imageHeight,
            classCount,
            regressionMax,
            baseOptions with { TopK = oneToOneTopK });
    }

    /// <summary>
    /// 按模型的检测头自动挑损失：<c>end2end</c>（有 one-to-one 分支）用双头损失；
    /// 其余家族保持原来的单头损失。reg_max 一律从检测头取，避免 reg_max = 1 的模型
    /// 被喂进 16-bin 的 DFL 损失。
    /// </summary>
    public static ILoss CreateFor(
        YoloDetectionModel model,
        int imageWidth,
        int imageHeight,
        int classCount,
        YoloLossOptions? options = null,
        bool trackComponents = false)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (model.Head.UsesEndToEnd)
        {
            return new YoloEndToEndDetectionLoss(
                imageWidth, imageHeight, classCount, model.Head.RegressionMax, options)
            {
                TrackComponents = trackComponents,
            };
        }

        return new YoloDetectionLoss(
            imageWidth, imageHeight, classCount, model.Head.RegressionMax, options)
        {
            TrackComponents = trackComponents,
        };
    }

    /// <inheritdoc />
    public string Name => "yolo26-detect";

    /// <summary>每个锚点的输出宽度（<c>4*reg_max + nc</c>，两条分支相同）。</summary>
    public int OutputChannels => _oneToMany.OutputChannels;

    /// <summary>
    /// 是否在每一步读回分项（会同步设备）。两条分支都会打开，
    /// <see cref="LastParts"/> 是两者之和；<see cref="LastOneToManyParts"/> 单独给出 one-to-many 的分项。
    /// </summary>
    public bool TrackComponents
    {
        get => _oneToMany.TrackComponents;
        set
        {
            _oneToMany.TrackComponents = value;
            _oneToOne.TrackComponents = value;
        }
    }

    /// <summary>最近一次 <see cref="Compute(Tensor,Tensor,IReadOnlyDictionary{string,Tensor})"/> 的分项之和。</summary>
    public YoloLossParts LastParts { get; private set; } = new(0f, 0f, 0f, 0f);

    /// <summary>最近一次 one-to-many 分支的分项（one-to-one 的部分 = <see cref="LastParts"/> 减去它）。</summary>
    public YoloLossParts LastOneToManyParts => _oneToMany.LastParts;

    /// <summary>切换输入尺寸（多尺度训练用）：两条分支的锚点网格一起重建。</summary>
    public void ConfigureImageSize(int width, int height)
    {
        _oneToMany.ConfigureImageSize(width, height);
        _oneToOne.ConfigureImageSize(width, height);
    }

    /// <summary>只有 one-to-many 分支的损失（没有 one-to-one 输出时的退化路径）。</summary>
    public Tensor Compute(Tensor prediction, Tensor target)
    {
        Tensor total = _oneToMany.Compute(prediction, target);
        LastParts = _oneToMany.LastParts;
        return total;
    }

    /// <summary>双头损失：one-to-many + one-to-one（官方 E2EDetectLoss 就是两项直接相加）。</summary>
    public Tensor Compute(
        Tensor prediction,
        Tensor target,
        IReadOnlyDictionary<string, Tensor> extras)
    {
        if (!extras.TryGetValue(OneToOneKey, out Tensor? oneToOne))
            return Compute(prediction, target);

        Tensor oneToManyLoss = _oneToMany.Compute(prediction, target);
        Tensor oneToOneLoss = _oneToOne.Compute(oneToOne, target);

        if (TrackComponents)
        {
            YoloLossParts first = _oneToMany.LastParts;
            YoloLossParts second = _oneToOne.LastParts;
            LastParts = new YoloLossParts(
                first.Box + second.Box,
                first.Class + second.Class,
                first.Dfl + second.Dfl,
                first.Total + second.Total);
        }

        return oneToManyLoss + oneToOneLoss;
    }
}
