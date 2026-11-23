using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Core;

/// <summary>可选计算设备。</summary>
public enum ComputeDevice
{
    /// <summary>CPU（任何机器可用）。</summary>
    Cpu,

    /// <summary>CUDA（需要 libtorch-cuda 原生包 + NVIDIA 驱动）。</summary>
    Cuda,
}

/// <summary>
/// TorchSharp 设备选择与可用性检查。
/// <para>
/// TorchSharp / LibTorch 只有 CPU 与 CUDA 两种设备；本项目<b>不提供 Vulkan</b>。
/// 缺省设备在进程里只探测一次：CUDA 可用就用 CUDA，否则用 CPU（见 <see cref="Default"/>）。
/// 显式指定 CUDA 但不可用时直接抛错并给出安装指引，不做静默回退。
/// </para>
/// </summary>
public static class DeviceFactory
{
    private static readonly Lazy<ComputeDevice> LazyDefault = new(ProbeDefault, isThreadSafe: true);

    /// <summary>
    /// 进程缺省设备：CUDA 可用就是 CUDA，否则 CPU。第一次读取时探测并缓存，
    /// 之后所有训练都读同一个结果——没有环境变量、没有配置项、也没有全局默认设备。
    /// </summary>
    public static ComputeDevice Default => LazyDefault.Value;

    /// <summary>
    /// 让探测在启动时就发生（早于窗口与训练线程），避免第一轮训练的日志里还得再等一次初始化。
    /// 多次调用无副作用。
    /// </summary>
    public static void InitializeDefault()
    {
        if (LazyDefault.Value == ComputeDevice.Cuda)
            ConfigureCuda();
    }

    /// <summary>解析设备名（cpu / cuda）；空值表示「用缺省设备」。</summary>
    public static ComputeDevice Parse(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return Default;

        return value.Trim().ToLowerInvariant() switch
        {
            "cpu" => ComputeDevice.Cpu,
            "cuda" or "gpu" => ComputeDevice.Cuda,
            _ => throw new ArgumentException($"未知设备 '{value}'；可选：cpu / cuda。", nameof(value)),
        };
    }

    /// <summary>创建 TorchSharp 设备对象；CUDA 不可用时抛错。</summary>
    public static Device Create(ComputeDevice device)
    {
        if (device == ComputeDevice.Cpu)
            return torch.CPU;

        if (!torch.cuda.is_available())
        {
            throw new InvalidOperationException(
                "CUDA 不可用：当前进程加载的是 CPU 版 LibTorch，或机器没有可用的 NVIDIA 驱动。"
                + "请把 libtorch-cpu 换成对应版本的 libtorch-cuda-*-win-x64，"
                + "并确认 nvidia-smi 可用；详见 docs/backends.md。");
        }

        ConfigureCuda();
        return torch.CUDA;
    }

    /// <summary>
    /// CUDA 上对训练安全、几乎无损的加速开关（只设一次）。
    /// <para>
    /// <c>allow_tf32</c>：Ampere 及以后的显卡用 TF32 做矩阵乘与卷积，吞吐通常提升 1.5~2 倍，
    /// 对检测训练精度的影响在千分位以内。输入尺寸固定（训练/验证同一个 imgsz）本来还能再开
    /// cuDNN 的 <c>benchmark</c>，但 TorchSharp 0.107 没有暴露该属性，这里只能留默认。
    /// </para>
    /// <para>
    /// 包了 try/catch：CPU 版 LibTorch 或较老的版本没有这些开关，缺了也不能让训练起不来。
    /// </para>
    /// </summary>
    private static void ConfigureCuda()
    {
        if (Interlocked.Exchange(ref _cudaConfigured, 1) != 0)
            return;

        try
        {
            torch.backends.cudnn.allow_tf32 = true;
            torch.backends.cuda.matmul.allow_tf32 = true;
        }
        catch (Exception)
        {
            // 没有对应开关就维持默认（FP32）；不影响正确性
        }
    }

    private static int _cudaConfigured;


    /// <summary>人类可读描述（写进日志与检查点元数据）。</summary>
    public static string Describe(ComputeDevice device) => device switch
    {
        ComputeDevice.Cpu => "cpu",
        ComputeDevice.Cuda => "cuda",
        _ => device.ToString().ToLowerInvariant(),
    };

    /// <summary>按名字描述；空值描述的是缺省设备。</summary>
    public static string Describe(string? value) => Describe(Parse(value));

    /// <summary>设备详细信息（CUDA 时带设备名）。</summary>
    public static string DescribeDetailed(ComputeDevice device)
    {
        if (device == ComputeDevice.Cpu)
            return $"cpu (torch {torch.__version__})";

        if (!torch.cuda.is_available())
            return "cuda (unavailable)";

        return $"cuda:0 (torch {torch.__version__})";
    }

    /// <summary>按名字给出详细信息；空值描述的是缺省设备。</summary>
    public static string DescribeDetailed(string? value) => DescribeDetailed(Parse(value));

    /// <summary>
    /// 探测缺省设备：原生库加载失败、驱动异常都当成「这台机器没有 CUDA」，退回 CPU 而不是让程序起不来。
    /// </summary>
    private static ComputeDevice ProbeDefault()
    {
        try
        {
            return torch.cuda.is_available() ? ComputeDevice.Cuda : ComputeDevice.Cpu;
        }
        catch (Exception)
        {
            return ComputeDevice.Cpu;
        }
    }
}
