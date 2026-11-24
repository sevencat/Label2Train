using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Training;

/// <summary>
/// 一个训练批次（TorchSharp 张量，形状由模型/损失决定）。
/// <para>
/// 一批 640×640×3×8 的输入就是几十 MB 显存，生产批次的代码（<c>BatchesForEpoch</c> 之类）
/// 在下一次 <c>yield</c> 前会 <see cref="Dispose"/> 上一批；不释放的话显存会随步数线性增长。
/// </para>
/// </summary>
public sealed record Batch(Tensor Input, Tensor Target) : IDisposable
{
    /// <inheritdoc />
    public void Dispose()
    {
        Input.Dispose();
        Target.Dispose();
    }
}

/// <summary>损失函数：给定预测与目标，返回可反向的标量张量。</summary>
public interface ILoss
{
    /// <summary>损失名（写进日志与检查点元数据）。</summary>
    string Name { get; }

    /// <summary>计算损失。</summary>
    Tensor Compute(Tensor prediction, Tensor target);

    /// <summary>
    /// 带"额外张量"的损失计算（例如两阶段提议的深监督）。
    /// <para>
    /// 默认实现**直接忽略 extras**，因此所有既有实现者无需改动；需要用到附加信息的损失
    /// （如 <c>DetrLoss</c> 的 <c>proposal_boxes</c> / <c>proposal_logits</c>）再显式覆写。
    /// 这样接口扩展是"零破坏"的：训练器可以统一走这个重载，而普通损失行为不变。
    /// </para>
    /// </summary>
    Tensor Compute(Tensor prediction, Tensor target, IReadOnlyDictionary<string, Tensor> extras) =>
        Compute(prediction, target);
}

/// <summary>
/// 需要在 epoch 边界调整自身状态的损失（例如 YOLO26 双头的 one-to-many / one-to-one 权重衰减）。
/// <para>
/// 训练器在每个 epoch 的所有批次跑完后调用一次，与日志节流无关；实现方自己决定如何用
/// <paramref name="epoch"/> 与 <paramref name="totalEpochs"/> 换算进度。
/// </para>
/// </summary>
public interface IEpochAwareLoss
{
    /// <summary>一个 epoch 结束时回调；<paramref name="epoch"/> 从 0 开始，<paramref name="totalEpochs"/> 是本次训练的总轮数。</summary>
    void OnEpochEnd(int epoch, int totalEpochs);
}

/// <summary>均方误差（回归 / 冒烟测试）。</summary>
public sealed class MeanSquaredError : ILoss
{
    public string Name => "mse";

    public Tensor Compute(Tensor prediction, Tensor target) =>
        torch.nn.functional.mse_loss(prediction, target);
}
