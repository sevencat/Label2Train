using Label2Train.Train.Models.Yolo;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Training.Yolo;

/// <summary>
/// YOLO26 的双头损失（官方 <c>E2ELoss</c>）：一条 one-to-many 分支 + 一条 one-to-one 分支，
/// 两条分支各自按同一套 TAL 分配算 box / cls / dfl（reg_max = 1 时第三项退化成 L1），再按
/// 随 epoch 衰减的权重加权求和。
/// <para>
/// <c>prediction</c> 是模型前向的 one-to-many 原始输出 <c>[B, 4*nc+nc, N]</c>；
/// one-to-one 分支的原始输出由模型通过 extras（键 <see cref="OneToOneKey"/>）传进来，
/// 没有 extras 时退化成单头损失（训练器在模型不支持双头时就走这条路）。
/// </para>
/// <para>
/// 与 ultralytics 8.4 的 <c>E2ELoss</c> 逐项对齐：
/// one-to-many 用 <c>tal_topk=10</c>，one-to-one 用 <c>tal_topk=7, tal_topk2=1</c>；
/// 两条分支的权重按 <c>decay(x) = max(1 - x/(epochs-1), 0) * (0.8 - 0.1) + 0.1</c> 线性衰减，
/// 其中 <c>x</c> 是已结束的 epoch 数（从 1 起）。训练前期以 one-to-many 为主（0.8 : 0.2），
/// 末期几乎全靠 one-to-one（0.1 : 0.9）——这正是 YOLO26 能免 NMS 的训练策略。
/// </para>
/// <para>
/// 权重只在 epoch 边界变化：训练器每个 epoch 结束调用一次 <see cref="OnEpochEnd"/>。
/// </para>
/// </summary>
public sealed class YoloEndToEndDetectionLoss : ILoss, IEpochAwareLoss
{
    /// <summary>extras 里 one-to-one 分支原始输出的键名（与模型侧的字面量必须一致）。</summary>
    public const string OneToOneKey = YoloV26DetectionModel.OneToOneKey;

    /// <summary>one-to-many 分支的初始权重（官方 <c>o2m = 0.8</c>）。</summary>
    public const float InitialOneToManyWeight = 0.8f;

    /// <summary>one-to-many 分支的终止权重（官方 <c>final_o2m = 0.1</c>）。</summary>
    public const float FinalOneToManyWeight = 0.1f;

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
        int oneToOneTopK = 7,
        int oneToOneTopK2 = 1)
    {
        if (oneToManyTopK <= 0)
            throw new ArgumentOutOfRangeException(nameof(oneToManyTopK), oneToManyTopK, "top-k 必须大于 0。");
        if (oneToOneTopK <= 0)
            throw new ArgumentOutOfRangeException(nameof(oneToOneTopK), oneToOneTopK, "top-k 必须大于 0。");
        if (oneToOneTopK2 <= 0 || oneToOneTopK2 > oneToOneTopK)
        {
            throw new ArgumentOutOfRangeException(
                nameof(oneToOneTopK2), oneToOneTopK2, "topk2 必须在 1 到 top-k 之间。");
        }

        YoloLossOptions baseOptions = options ?? new YoloLossOptions();

        _oneToMany = new YoloDetectionLoss(
            imageWidth,
            imageHeight,
            classCount,
            regressionMax,
            baseOptions with { TopK = oneToManyTopK, TopK2 = 0 });
        _oneToOne = new YoloDetectionLoss(
            imageWidth,
            imageHeight,
            classCount,
            regressionMax,
            baseOptions with { TopK = oneToOneTopK, TopK2 = oneToOneTopK2 });
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
    /// 当前 one-to-many 分支的权重；初始 0.8，每个 epoch 结束按官方公式向 0.1 衰减。
    /// </summary>
    public float OneToManyWeight { get; private set; } = InitialOneToManyWeight;

    /// <summary>当前 one-to-one 分支的权重（<c>1 - OneToManyWeight</c>，0.2 → 0.9）。</summary>
    public float OneToOneWeight => 1f - OneToManyWeight;

    /// <summary>已经结束的 epoch 数（官方 <c>updates</c>）。</summary>
    public int Updates { get; private set; }

    /// <summary>
    /// 是否在每一步读回分项（会同步设备）。两条分支都会打开，
    /// <see cref="LastParts"/> 是两者按当前权重加权后的和；<see cref="LastOneToManyParts"/> 单独给出 one-to-many 的原始分项。
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

    /// <summary>最近一次计算的分项（两条分支按当前权重加权后的和）。</summary>
    public YoloLossParts LastParts { get; private set; } = new(0f, 0f, 0f, 0f);

    /// <summary>最近一次 one-to-many 分支的原始分项（未加权）。</summary>
    public YoloLossParts LastOneToManyParts => _oneToMany.LastParts;

    /// <summary>最近一次 one-to-one 分支的原始分项（未加权）。</summary>
    public YoloLossParts LastOneToOneParts => _oneToOne.LastParts;

    /// <summary>是否记录 TAL 分配统计（转发给两条分支；会同步设备，仅诊断用）。</summary>
    public bool CaptureAssignmentStats
    {
        get => _oneToOne.CaptureAssignmentStats;
        set
        {
            _oneToMany.CaptureAssignmentStats = value;
            _oneToOne.CaptureAssignmentStats = value;
        }
    }

    /// <summary>one-to-many 分支最近一次分配的前景锚点数。</summary>
    public int OneToManyForegroundCount => _oneToMany.LastForegroundCount;

    /// <summary>one-to-one 分支最近一次分配的前景锚点数（topk2 = 1 时理论值等于有效 GT 数）。</summary>
    public int OneToOneForegroundCount => _oneToOne.LastForegroundCount;

    /// <summary>one-to-one 分支最近一次分配的目标分数总和。</summary>
    public float OneToOneTargetScoreSum => _oneToOne.LastTargetScoreSum;

    /// <summary>one-to-one 分支最近一次分配的前 40 个前景锚点（展平索引）。</summary>
    public IReadOnlyList<int> OneToOneForegroundPositions => _oneToOne.LastForegroundPositions;

    /// <summary>切换输入尺寸（多尺度训练用）：两条分支的锚点网格一起重建。</summary>
    public void ConfigureImageSize(int width, int height)
    {
        _oneToMany.ConfigureImageSize(width, height);
        _oneToOne.ConfigureImageSize(width, height);
    }

    /// <inheritdoc />
    public void OnEpochEnd(int epoch, int totalEpochs)
    {
        // 官方 E2ELoss.update()：updates += 1 后按 decay(updates) 重算 o2m；
        // epoch 从 0 起，结束时等价于 updates = epoch + 1。
        Updates = epoch + 1;
        OneToManyWeight = Decay(Updates, totalEpochs);
    }

    /// <summary>官方 <c>E2ELoss.decay</c>：线性衰减到 <see cref="FinalOneToManyWeight"/>，epochs=1 时不除零。</summary>
    internal static float Decay(int updates, int epochs)
    {
        float fraction = MathF.Max(1f - updates / (float)Math.Max(epochs - 1, 1), 0f);
        return fraction * (InitialOneToManyWeight - FinalOneToManyWeight) + FinalOneToManyWeight;
    }

    /// <summary>
    /// 只有 one-to-many 分支的损失（没有 one-to-one 输出时的退化路径）。
    /// <para>此时没有双头加权可言，保持原始尺度，与单头 <see cref="YoloDetectionLoss"/> 的数值可比。</para>
    /// </summary>
    public Tensor Compute(Tensor prediction, Tensor target)
    {
        Tensor total = _oneToMany.Compute(prediction, target);

        if (TrackComponents)
            LastParts = _oneToMany.LastParts;

        return total;
    }

    /// <summary>双头损失：<c>one2many * o2m + one2one * o2o</c>（官方 E2ELoss）。</summary>
    public Tensor Compute(
        Tensor prediction,
        Tensor target,
        IReadOnlyDictionary<string, Tensor> extras)
    {
        if (!extras.TryGetValue(OneToOneKey, out Tensor? oneToOne))
            return Compute(prediction, target);

        Tensor oneToManyLoss = _oneToMany.Compute(prediction, target);
        Tensor oneToOneLoss = _oneToOne.Compute(oneToOne, target);

        float oneToManyWeight = OneToManyWeight;
        float oneToOneWeight = OneToOneWeight;

        if (TrackComponents)
        {
            YoloLossParts first = Scale(_oneToMany.LastParts, oneToManyWeight);
            YoloLossParts second = Scale(_oneToOne.LastParts, oneToOneWeight);
            LastParts = new YoloLossParts(
                first.Box + second.Box,
                first.Class + second.Class,
                first.Dfl + second.Dfl,
                first.Total + second.Total);
        }

        return oneToManyLoss * oneToManyWeight + oneToOneLoss * oneToOneWeight;
    }

    private static YoloLossParts Scale(YoloLossParts parts, float weight) =>
        new(parts.Box * weight, parts.Class * weight, parts.Dfl * weight, parts.Total * weight);
}
