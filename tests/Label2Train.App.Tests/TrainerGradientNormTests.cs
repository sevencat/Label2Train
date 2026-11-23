using Label2Train.Train.Training;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.App.Tests;

/// <summary>
/// <see cref="Trainer.LastGradientNorm"/> 要能回答「模型还在不在学」：
/// 参数被推到 sigmoid 饱和端之后梯度是「极小」而不是「精确为零」，而冻结时 loss 仍会抖百分之几，
/// 所以判据只能是梯度范数。这里用「损失乘 0」造出真正没有梯度的情形，
/// 顺带盯住「关掉裁剪时也要统计范数」这一点（否则诊断彻底失效）。
/// </summary>
[Collection("torch")]
public sealed class TrainerGradientNormTests
{
    [Theory]
    [InlineData(1f, true)]
    [InlineData(0f, false)]
    public void LastGradientNorm_Tells_Learning_From_Frozen(float lossScale, bool parametersMove)
    {
        using var model = Linear(2, 1);
        using var optimizer = torch.optim.SGD(model.parameters(), learningRate: 0.1);
        using var batch = new Batch(torch.ones(2, 2), torch.zeros(1));

        // GradientClipNorm = 0 表示关掉裁剪：范数仍然要算出来，这正是诊断冻结的用法
        var trainer = new Trainer(
            model,
            optimizer,
            new ScaledLoss(lossScale),
            new TrainerOptions { Epochs = 1, GradientClipNorm = 0f, Shuffle = false });

        trainer.Fit([batch]);

        Assert.False(double.IsNaN(trainer.LastGradientNorm));
        Assert.Equal(parametersMove, trainer.LastGradientNorm > 0);
    }

    /// <summary>损失 = 预测求和 × 倍率；倍率 0 时反向得到全 0 梯度，等价于「一步都没学」。</summary>
    private sealed class ScaledLoss(float scale) : ILoss
    {
        public string Name => "scaled";

        public Tensor Compute(Tensor prediction, Tensor target) => prediction.sum() * scale;
    }
}
