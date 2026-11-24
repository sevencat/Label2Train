using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Models.Yolo;

/// <summary>
/// ultralytics 的 <c>SPPF</c>：串联三个同核最大池化再拼接，用两个 1×1 卷积压回目标通道。
/// <para>子模块名 <c>cv1</c> / <c>cv2</c> / <c>m</c> 与 ultralytics 一致。</para>
/// <para>
/// <paramref name="shortcut"/> 对应官方 <c>SPPF(shortcut=True)</c>（yolo26.yaml 层 9 就带它）：
/// 输出再与输入相加，且只在 <c>c1 == c2</c> 时生效（官方 <c>self.add = shortcut and c1 == c2</c>）。
/// </para>
/// <para>
/// <paramref name="cv1Activation"/> 对应官方 <c>cv1</c> 是否接 SiLU。ultralytics 更新后把
/// <c>cv1</c> 改成了 <c>Conv(c1, c_, 1, 1, act=False)</c>：
/// 老检查点（yolov8n / yolo11n 等）里仍是 SiLU，新结构（yolo26）是 Identity，
/// 加载旧权重时必须保持 <c>true</c>，否则数值会整体偏移。
/// </para>
/// </summary>
public sealed class SppfModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> cv1;
    private readonly Module<Tensor, Tensor> cv2;
    private readonly Module<Tensor, Tensor> m;
    private readonly bool _addShortcut;

    /// <summary>创建 SPPF；<paramref name="kernelSize"/> 默认 5（与 yolov8.yaml 一致）。</summary>
    public SppfModule(
        int inputChannels,
        int outputChannels,
        int kernelSize = 5,
        Device? device = null,
        bool shortcut = false,
        bool cv1Activation = true)
        : base("SPPF")
    {
        if (inputChannels <= 0 || outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), inputChannels, "通道数必须大于 0。");
        if (kernelSize <= 0 || kernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), kernelSize, "SPPF 的池化核必须是正奇数。");

        int hiddenChannels = inputChannels / 2;
        KernelSize = kernelSize;
        UsesShortcut = shortcut && inputChannels == outputChannels;
        _addShortcut = UsesShortcut;

        cv1 = new ConvModule(inputChannels, hiddenChannels, 1, 1, device, activation: cv1Activation);
        cv2 = new ConvModule(hiddenChannels * 4, outputChannels, 1, 1, device);
        m = nn.MaxPool2d(kernelSize, 1, kernelSize / 2);

        register_module("cv1", cv1);
        register_module("cv2", cv2);
        register_module("m", m);
    }

    /// <summary>池化核尺寸（导出时写成 MaxPool 的 <c>kernel_shape</c>）。</summary>
    public int KernelSize { get; }

    /// <summary>输出是否与输入相加（官方 <c>shortcut and c1 == c2</c>）。</summary>
    public bool UsesShortcut { get; }

    /// <summary>输入卷积（导出 ONNX 时按层展开）。</summary>
    public ConvModule Cv1 => (ConvModule)cv1;

    /// <summary>输出卷积（导出 ONNX 时按层展开）。</summary>
    public ConvModule Cv2 => (ConvModule)cv2;

    /// <inheritdoc />
    public override Tensor forward(Tensor input)
    {
        Tensor first = cv1.forward(input);
        Tensor second = m.forward(first);
        Tensor third = m.forward(second);
        Tensor fourth = m.forward(third);

        Tensor pooled = cv2.forward(torch.cat(new[] { first, second, third, fourth }, 1));

        return _addShortcut ? pooled + input : pooled;
    }
}
