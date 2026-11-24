using Label2Train.Train.Training;
using TorchSharp;
using TorchSharp.Modules;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.App.Tests;

/// <summary>
/// 官方 <c>build_optimizer</c> 的衰减分组：conv/linear 权重吃 weight_decay，BN/LN 权重与 bias 不吃。
/// TorchSharp 的 <c>ParamGroup</c> 不支持逐组 weight_decay，所以走
/// <see cref="TrainerOptions.DecayOnlyParameters"/> 在每一步后补解耦衰减。
/// 这里把「只有指定参数被乘 1-lr·wd、其余一动不动」钉住，口径错了训练会悄悄跑偏。
/// </summary>
[Collection("torch")]
public sealed class TrainerDecoupledWeightDecayTests
{
    private const float LearningRate = 0.1f;
    private const float Decay = 0.5f;

    [Fact]
    public void Only_Configured_Parameters_Are_Decayed()
    {
        using var model = Linear(2, 2);

        Parameter weight = model.weight;
        Parameter bias = model.bias!;

        using (torch.no_grad())
        {
            weight.fill_(1.0);
            bias.fill_(1.0);
        }

        using var optimizer = torch.optim.AdamW(model.parameters(), lr: LearningRate, weight_decay: 0.0);
        using var batch = new Batch(torch.ones(2, 2), torch.zeros(1));

        var trainer = new Trainer(
            model,
            optimizer,
            new ZeroLoss(),
            new TrainerOptions
            {
                Epochs = 1,
                GradientClipNorm = 0f,
                Shuffle = false,
                DecayOnlyParameters = [weight],
                DecayOnlyWeightDecay = Decay,
            });

        trainer.Fit([batch]);

        // 梯度全 0 时 AdamW 本体不动参数，只剩解耦衰减：1 × (1 - 0.1 × 0.5) = 0.95
        Assert.Equal(0.95f, weight.mean().item<float>(), 4);
        Assert.Equal(1.0f, bias.mean().item<float>(), 4);
    }

    [Fact]
    public void Without_Configuration_Nothing_Is_Decayed()
    {
        using var model = Linear(2, 2);
        Parameter weight = model.weight;

        using (torch.no_grad())
            weight.fill_(1.0);

        using var optimizer = torch.optim.AdamW(model.parameters(), lr: LearningRate, weight_decay: 0.0);
        using var batch = new Batch(torch.ones(2, 2), torch.zeros(1));

        var trainer = new Trainer(
            model,
            optimizer,
            new ZeroLoss(),
            new TrainerOptions { Epochs = 1, GradientClipNorm = 0f, Shuffle = false });

        trainer.Fit([batch]);

        // 没配置就是老路径：优化器 weight_decay 为 0，谁都不衰减
        Assert.Equal(1.0f, weight.mean().item<float>(), 4);
    }

    /// <summary>损失恒等于 0：反向得到全 0 梯度，只留下权重衰减的影响。</summary>
    private sealed class ZeroLoss : ILoss
    {
        public string Name => "zero";

        public Tensor Compute(Tensor prediction, Tensor target) => prediction.sum() * 0f;
    }
}
