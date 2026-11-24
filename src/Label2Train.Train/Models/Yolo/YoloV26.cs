using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Models.Yolo;

/// <summary>
/// YOLOv26 检测模型工厂：按 ultralytics <c>yolo26.yaml</c> 的 24 层结构组装主干、PAN 颈部与检测头。
/// <para>
/// 与 <see cref="YoloV11"/> 的差异（官方 yaml 只差两行）：
/// <list type="bullet">
/// <item><c>model.22</c> 的 <c>C3k2</c> 带 <c>attn=True</c>：内层块是
/// <c>Sequential(Bottleneck, PSABlock)</c>，而不是单个 Bottleneck。</item>
/// <item><c>model.23</c> 的 <c>Detect</c> 带 <c>end2end=True</c>、<c>reg_max=1</c>：
/// 没有 DFL，多一套 <c>one2one_cv2</c> / <c>one2one_cv3</c>，推理直接 top-k 输出、不需要 NMS。</item>
/// </list>
/// 主干 / 颈部 / scales 与 <c>yolo11.yaml</c> 逐行相同。
/// </para>
/// </summary>
public static class YoloV26
{
    /// <summary>输入尺寸必须是该值的倍数（P3 / P4 / P5 的最大下采样倍数）。</summary>
    public const int InputStride = YoloDetectionModel.InputStride;

    /// <summary>创建 YOLOv26 检测模型。</summary>
    public static YoloV26DetectionModel Create(
        Device device,
        YoloScale scale = YoloScale.N,
        int classCount = 80)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "类别数必须大于 0。");

        YoloScaleConfig config = YoloScales.Config(YoloModelFamily.V26, scale);

        var from = new List<int[]>();
        var modules = new List<Module>();
        var channels = new List<int>();

        int Ch(int index) => channels[index];

        void Add(int[] source, Module module, int outputChannels)
        {
            from.Add(source);
            modules.Add(module);
            channels.Add(outputChannels);
        }

        int Width(int channelsBeforeScale) => YoloScales.ScaledChannels(channelsBeforeScale, config);
        int Repeats(int repeats) => YoloScales.ScaledRepeats(repeats, config);

        int[] Self() => new[] { -1 };

        // backbone 0-10（与 v11 相同）
        Add(Self(), new ConvModule(3, Width(64), 3, 2, device), Width(64));
        Add(Self(), new ConvModule(Ch(0), Width(128), 3, 2, device), Width(128));
        Add(Self(), new C3k2Module(Ch(1), Width(256), Repeats(2), false, 0.25, true, device), Width(256));
        Add(Self(), new ConvModule(Ch(2), Width(256), 3, 2, device), Width(256));
        Add(Self(), new C3k2Module(Ch(3), Width(512), Repeats(2), false, 0.25, true, device), Width(512));
        Add(Self(), new ConvModule(Ch(4), Width(512), 3, 2, device), Width(512));
        Add(Self(), new C3k2Module(Ch(5), Width(512), Repeats(2), true, 0.5, true, device), Width(512));
        Add(Self(), new ConvModule(Ch(6), Width(1024), 3, 2, device), Width(1024));
        Add(Self(), new C3k2Module(Ch(7), Width(1024), Repeats(2), true, 0.5, true, device), Width(1024));
        // yolo26.yaml 层 9 是 [1024, 5, 3, True]：多一个 shortcut（c1 == c2 时输出再与输入相加）；
        // 官方新版 SPPF 的 cv1 是 act=False，与 yolov8/yolo11 的老检查点不同
        Add(
            Self(),
            new SppfModule(Ch(8), Width(1024), 5, device, shortcut: true, cv1Activation: false),
            Width(1024));
        Add(Self(), new C2PsaModule(Ch(9), Width(1024), Repeats(2), 0.5, device), Width(1024));

        // neck 11-22（上采样 + 跨层拼接）
        Add(Self(), nn.Upsample(null, new double[] { 2.0, 2.0 }, UpsampleMode.Nearest), Ch(10));
        Add(new[] { 11, 6 }, new ConcatModule(), Ch(11) + Ch(6));

        // yolo26 的颈部 C3k2 用 c3k=True（v11 是 False），内层块是 C3k 而非单个 Bottleneck
        Add(Self(), new C3k2Module(Ch(12), Width(512), Repeats(2), true, 0.5, true, device), Width(512));
        Add(Self(), nn.Upsample(null, new double[] { 2.0, 2.0 }, UpsampleMode.Nearest), Ch(13));
        Add(new[] { 14, 4 }, new ConcatModule(), Ch(14) + Ch(4));
        Add(Self(), new C3k2Module(Ch(15), Width(256), Repeats(2), true, 0.5, true, device), Width(256));
        Add(Self(), new ConvModule(Ch(16), Width(256), 3, 2, device), Width(256));
        Add(new[] { 17, 13 }, new ConcatModule(), Ch(17) + Ch(13));
        Add(Self(), new C3k2Module(Ch(18), Width(512), Repeats(2), true, 0.5, true, device), Width(512));
        Add(Self(), new ConvModule(Ch(19), Width(512), 3, 2, device), Width(512));
        Add(new[] { 20, 10 }, new ConcatModule(), Ch(20) + Ch(10));

        // yolo26.yaml 层 22 是 [-1, 1, C3k2, [1024, True, 0.5, True]]：
        // repeats 为 1（v11 是 2），内层块多了 PSA 注意力（attn=True）
        Add(
            Self(),
            new C3k2Module(Ch(21), Width(1024), Repeats(1), true, 0.5, true, device, attention: true),
            Width(1024));

        // head 23：reg_max = 1（无 DFL）+ one-to-one 分支（YOLO26 的去 NMS 推理）
        var head = new DetectHeadModule(
            new[] { Ch(16), Ch(19), Ch(22) },
            classCount,
            DetectHeadModule.EndToEndRegressionMax,
            device,
            depthwiseClassBranch: true,
            endToEnd: true);
        Add(new[] { 16, 19, 22 }, head, head.OutputChannels);

        var chain = new YoloModuleChain("model", from, modules);
        return new YoloV26DetectionModel("DetectionModel", chain, head, scale, classCount);
    }
}
