using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Models.Yolo;

/// <summary>
/// ultralytics 的 <c>Detect</c> 头：三个尺度各走一条 <c>cv2</c>（box 回归）与 <c>cv3</c>（分类）分支，
/// 输出未解码的原始预测。
/// <para>
/// 每个尺度输出 <c>[B, 4*reg_max + nc, H*W]</c>，沿锚点维拼接成 <c>[B, 4*reg_max + nc, N]</c>。
/// 训练时该张量直接喂给 TAL 分配与 box / cls / dfl 三项损失；推理解码
/// （DFL 积分 + dist2bbox + sigmoid）由导出 / 预测路径负责，保持与 ultralytics 的
/// <c>output0</c> 布局一致。
/// </para>
/// <para>
/// YOLO26 用 <paramref name="endToEnd"/> = <c>true</c> 打开第二套分支
/// （<c>one2one_cv2</c> / <c>one2one_cv3</c>，键名与官方一致）：两套分支的权重形状相同、
/// 初始值相同，训练时 one-to-one 分支只做 one-to-many 的「1 对 1」版本，推理用它就不需要 NMS。
/// 此时 <c>reg_max = 1</c>，官方不建 DFL（<c>nn.Identity</c>），所以这里也不注册 <c>dfl</c>。
/// </para>
/// <para>
/// 子模块名 <c>cv2</c> / <c>cv3</c> / <c>dfl</c> / <c>one2one_cv2</c> / <c>one2one_cv3</c>
/// 与 ultralytics 一致，键名形如 <c>model.22.cv2.0.0.conv.weight</c>。
/// </para>
/// </summary>
public sealed class DetectHeadModule : Module<Tensor[], Tensor>
{
    /// <summary>ultralytics 默认的 DFL 区间数。</summary>
    public const int DefaultRegressionMax = 16;

    /// <summary>YOLO26 的 DFL 区间数（等于 1 时不建 DFL，直接回归距离）。</summary>
    public const int EndToEndRegressionMax = 1;

    /// <summary>P3 / P4 / P5 的下采样倍数，分类 bias 先验按它换算。</summary>
    private static readonly int[] DefaultStrides = { 8, 16, 32 };

    private readonly ModuleList<Module<Tensor, Tensor>> cv2;
    private readonly ModuleList<Module<Tensor, Tensor>> cv3;
    // 字段名必须与 ultralytics 的键名逐字相同：TorchSharp 0.107 按字段名注册子模块，
    // 写成 one2oneCv2 会让 state_dict 出现 model.23.one2oneCv2.* 这种对不上的键
    private readonly ModuleList<Module<Tensor, Tensor>>? one2one_cv2;
    private readonly ModuleList<Module<Tensor, Tensor>>? one2one_cv3;
    private readonly DflModule? dfl;
    private readonly int _levels;
    private readonly YoloSequential[] _boxBranches;
    private readonly YoloSequential[] _classBranches;
    private readonly YoloSequential[]? _oneToOneBoxBranches;
    private readonly YoloSequential[]? _oneToOneClassBranches;

    /// <summary>按各尺度输入通道创建检测头。</summary>
    /// <param name="inputChannels">P3 / P4 / P5 的输入通道。</param>
    /// <param name="classCount">类别数。</param>
    /// <param name="regressionMax">DFL 区间数（ultralytics 默认 16；YOLO26 为 1）。</param>
    /// <param name="device">计算设备。</param>
    /// <param name="depthwiseClassBranch">
    /// 分类分支是否用「深度可分离 3×3 + 1×1」的两段式（YOLOv11 / v26 的写法，键名多一层：
    /// <c>cv3.0.0.0.conv.weight</c>）；YOLOv8 用普通 3×3 卷积。
    /// </param>
    /// <param name="imageSize">
    /// 训练 / 推理的输入边长，只用于分类 bias 的初始先验（与 ultralytics 的
    /// <c>log(5 / nc / (imgsz / stride)²)</c> 一致）。
    /// </param>
    /// <param name="endToEnd">是否再建一套 one-to-one 分支（YOLO26）。</param>
    public DetectHeadModule(
        IReadOnlyList<int> inputChannels,
        int classCount,
        int regressionMax = DefaultRegressionMax,
        Device? device = null,
        bool depthwiseClassBranch = false,
        int imageSize = 640,
        bool endToEnd = false)
        : base("Detect")
    {
        ArgumentNullException.ThrowIfNull(inputChannels);

        if (inputChannels.Count == 0)
            throw new ArgumentException("检测头至少需要 1 个尺度。", nameof(inputChannels));
        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "类别数必须大于 0。");
        if (regressionMax <= 0)
            throw new ArgumentOutOfRangeException(nameof(regressionMax), regressionMax, "reg_max 必须大于 0。");

        _levels = inputChannels.Count;
        ClassCount = classCount;
        RegressionMax = regressionMax;
        UsesDepthwiseClassBranch = depthwiseClassBranch;
        UsesEndToEnd = endToEnd;

        // 与 ultralytics Detect 的宽度选择一致：回归分支取 max(16, ch/4, 4*reg_max)，
        // 分类分支取 max(ch[0], min(nc, 100))。
        int boxChannels = Math.Max(16, Math.Max(inputChannels[0] / 4, regressionMax * 4));
        int classChannels = Math.Max(inputChannels[0], Math.Min(classCount, 100));

        cv2 = new ModuleList<Module<Tensor, Tensor>>();
        cv3 = new ModuleList<Module<Tensor, Tensor>>();
        _boxBranches = new YoloSequential[inputChannels.Count];
        _classBranches = new YoloSequential[inputChannels.Count];

        YoloSequential BuildBox(int channels) => new(
            "Sequential",
            ("0", new ConvModule(channels, boxChannels, 3, 1, device)),
            ("1", new ConvModule(boxChannels, boxChannels, 3, 1, device)),
            ("2", nn.Conv2d(
                boxChannels,
                4 * regressionMax,
                1,
                1,
                0,
                bias: true,
                device: device,
                dtype: ScalarType.Float32)));

        YoloSequential BuildClass(int channels) => depthwiseClassBranch
            ? new YoloSequential(
                "Sequential",
                ("0", new YoloSequential(
                    "Sequential",
                    ("0", new ConvModule(channels, channels, 3, 1, device, groups: channels)),
                    ("1", new ConvModule(channels, classChannels, 1, 1, device)))),
                ("1", new YoloSequential(
                    "Sequential",
                    ("0", new ConvModule(classChannels, classChannels, 3, 1, device, groups: classChannels)),
                    ("1", new ConvModule(classChannels, classChannels, 1, 1, device)))),
                ("2", nn.Conv2d(
                    classChannels,
                    classCount,
                    1,
                    1,
                    0,
                    bias: true,
                    device: device,
                    dtype: ScalarType.Float32)))
            : new YoloSequential(
                "Sequential",
                ("0", new ConvModule(channels, classChannels, 3, 1, device)),
                ("1", new ConvModule(classChannels, classChannels, 3, 1, device)),
                ("2", nn.Conv2d(
                    classChannels,
                    classCount,
                    1,
                    1,
                    0,
                    bias: true,
                    device: device,
                    dtype: ScalarType.Float32)));

        for (int level = 0; level < inputChannels.Count; level++)
        {
            _boxBranches[level] = BuildBox(inputChannels[level]);
            _classBranches[level] = BuildClass(inputChannels[level]);

            cv2.Add(_boxBranches[level]);
            cv3.Add(_classBranches[level]);
        }

        // reg_max = 1 时官方用 nn.Identity，state_dict 里没有 dfl.*，这里也不建
        if (regressionMax > 1)
        {
            dfl = new DflModule(regressionMax, device);
        }

        ApplyBiasPrior(imageSize);

        register_module("cv2", cv2);
        register_module("cv3", cv3);

        if (dfl is not null)
        {
            register_module("dfl", dfl);
        }

        if (endToEnd)
        {
            one2one_cv2 = new ModuleList<Module<Tensor, Tensor>>();
            one2one_cv3 = new ModuleList<Module<Tensor, Tensor>>();
            _oneToOneBoxBranches = new YoloSequential[inputChannels.Count];
            _oneToOneClassBranches = new YoloSequential[inputChannels.Count];

            for (int level = 0; level < inputChannels.Count; level++)
            {
                _oneToOneBoxBranches[level] = BuildBox(inputChannels[level]);
                _oneToOneClassBranches[level] = BuildClass(inputChannels[level]);

                one2one_cv2.Add(_oneToOneBoxBranches[level]);
                one2one_cv3.Add(_oneToOneClassBranches[level]);
            }

            register_module("one2one_cv2", one2one_cv2);
            register_module("one2one_cv3", one2one_cv3);

            // 官方 __init__ 里是 copy.deepcopy(cv2/cv3)：两套分支初始权重完全相同，
            // 且都已带上分类 bias 先验。这里在 bias 先验之后整体拷一次，效果等价
            one2one_cv2.load_state_dict(cv2.state_dict());
            one2one_cv3.load_state_dict(cv3.state_dict());
        }
    }

    /// <summary>
    /// 按 ultralytics 的 <c>Detect.bias_init()</c> 设两支末层 bias 的先验：
    /// 框分支全设 <c>2.0</c>（初始距离 2 个 stride），分类分支设
    /// <c>bias = log(5 / nc / (imgsz / stride)²)</c>，即初始正样本概率约 <c>5/nc</c> 个格子。
    /// <para>
    /// 不做这一步时初始 sigmoid 全是 0.5，8400 个锚点的 BCE 汇总后 cls loss 会到上千，
    /// 与 box / dfl 相差几个数量级；在只有几百步的小数据集训练里，前几十步都在纠正这个偏置。
    /// </para>
    /// <para>
    /// 创建模型时按 640 先设一次；实际输入尺寸不同（例如 320）时，训练前再按真实 <paramref name="imageSize"/> 覆盖一次。
    /// 双头模型的 one-to-one 分支是在这之后 deepcopy 的，所以先验自动跟着过去。
    /// </para>
    /// </summary>
    public void ApplyBiasPrior(int imageSize)
    {
        ApplyBoxBiasPrior();
        ApplyClassBiasPrior(imageSize);
    }

    /// <summary>框分支末层 bias 全设成 <paramref name="value"/>（官方 2.0，即初始 2 个 stride 的距离）。</summary>
    public void ApplyBoxBiasPrior(float value = 2f)
    {
        using (torch.no_grad())
        {
            for (int level = 0; level < _levels; level++)
            {
                if (_boxBranches[level].Items[^1] is Conv2d box && box.bias is not null)
                    box.bias.fill_(value);
            }
        }
    }

    /// <summary>
    /// 只重设分类分支的 bias 先验。用在「载入官方权重但类别数不同」的场合：
    /// 框分支已经训练好了，不能跟着一起重置。
    /// </summary>
    public void ApplyClassBiasPrior(int imageSize)
    {
        if (imageSize <= 0)
            return;

        using (torch.no_grad())
        {
            for (int level = 0; level < _levels && level < DefaultStrides.Length; level++)
            {
                if (_classBranches[level].Items[^1] is not Conv2d conv || conv.bias is null)
                    continue;

                float cells = imageSize / (float)DefaultStrides[level];
                float priorProbability = 5f / ClassCount / (cells * cells);
                conv.bias.fill_(MathF.Log(priorProbability));
            }
        }
    }

    /// <summary>类别数（nc）。</summary>
    public int ClassCount { get; }

    /// <summary>DFL 区间数（reg_max）。</summary>
    public int RegressionMax { get; }

    /// <summary>分类分支是否用深度可分离卷积（YOLOv11 / v26）。</summary>
    public bool UsesDepthwiseClassBranch { get; }

    /// <summary>是否带 one-to-one 分支（YOLO26 的无 NMS 推理）。</summary>
    public bool UsesEndToEnd { get; }

    /// <summary>每个锚点的输出宽度：<c>4 * reg_max + nc</c>。</summary>
    public int OutputChannels => 4 * RegressionMax + ClassCount;

    /// <summary>尺度数量（P3 / P4 / P5 为 3）。</summary>
    public int LevelCount => _levels;

    /// <summary>DFL 积分层（导出时需要它的权重）；<c>reg_max = 1</c> 时为 <c>null</c>。</summary>
    public DflModule? Dfl => dfl;

    /// <summary>各尺度的 box 回归分支（<c>cv2[level]</c>，导出时逐层展开）。</summary>
    public IReadOnlyList<YoloSequential> BoxBranches => _boxBranches;

    /// <summary>各尺度的分类分支（<c>cv3[level]</c>，导出时逐层展开）。</summary>
    public IReadOnlyList<YoloSequential> ClassBranches => _classBranches;

    /// <summary>one-to-one 的 box 分支；未开 <see cref="UsesEndToEnd"/> 时为空。</summary>
    public IReadOnlyList<YoloSequential>? OneToOneBoxBranches => _oneToOneBoxBranches;

    /// <summary>one-to-one 的分类分支；未开 <see cref="UsesEndToEnd"/> 时为空。</summary>
    public IReadOnlyList<YoloSequential>? OneToOneClassBranches => _oneToOneClassBranches;

    /// <inheritdoc />
    public override Tensor forward(Tensor[] features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Length != _levels)
        {
            throw new ArgumentException(
                $"检测头需要 {_levels} 个尺度输入，实际收到 {features.Length} 个。",
                nameof(features));
        }

        var outputs = new Tensor[_levels];
        for (int level = 0; level < _levels; level++)
        {
            Tensor box = cv2[level].forward(features[level]);
            Tensor classes = cv3[level].forward(features[level]);
            outputs[level] = torch.cat(new[] { box, classes }, 1).flatten(2);
        }

        return torch.cat(outputs, 2);
    }

    /// <summary>
    /// one-to-one 分支的前向，输出布局与 <see cref="forward"/> 相同（<c>[B, 4*reg_max+nc, N]</c>）。
    /// <para>
    /// <paramref name="detachFeatures"/> = <c>true</c> 时先 detach 特征图：官方在<b>训练</b>时
    /// 用 <c>x_detach</c> 接 one-to-one 分支，避免这条「只为去 NMS」的支路把梯度顶回主干。
    /// </para>
    /// </summary>
    public Tensor ForwardOneToOne(Tensor[] features, bool detachFeatures)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (one2one_cv2 is null || one2one_cv3 is null)
        {
            throw new InvalidOperationException(
                "该检测头没有 one-to-one 分支（创建时 endToEnd=false）。");
        }

        if (features.Length != _levels)
        {
            throw new ArgumentException(
                $"检测头需要 {_levels} 个尺度输入，实际收到 {features.Length} 个。",
                nameof(features));
        }

        var outputs = new Tensor[_levels];
        for (int level = 0; level < _levels; level++)
        {
            Tensor input = detachFeatures ? features[level].detach() : features[level];
            Tensor box = one2one_cv2[level].forward(input);
            Tensor classes = one2one_cv3[level].forward(input);
            outputs[level] = torch.cat(new[] { box, classes }, 1).flatten(2);
        }

        return torch.cat(outputs, 2);
    }
}
