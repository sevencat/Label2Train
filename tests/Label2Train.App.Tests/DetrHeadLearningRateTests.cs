using Label2Train.Train.Sessions;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 随机初始化的头（分类头 / query / 参考点）的梯度放大规则。
/// 之前是写死的 ×10：基础 lr 取 1e-3（从零训的档位）时随机头就是 1e-2，
/// Adam 一步把 query_embed 与参考点推到 sigmoid 饱和端，之后梯度精确为 0、loss 冻住。
/// </summary>
public sealed class DetrHeadLearningRateTests
{
    [Theory]
    [InlineData(1e-4f, 10f)]
    [InlineData(2e-4f, 5f)]
    [InlineData(5e-4f, 2f)]
    [InlineData(1e-3f, 1f)]
    [InlineData(5e-3f, 1f)]
    [InlineData(0f, 10f)]
    public void Scale_Keeps_Head_Learning_Rate_At_Cap(float learningRate, float expected)
    {
        Assert.Equal(expected, DetrTrainingSession.HeadGradientScale(learningRate), 3);
    }

    [Fact]
    public void Scale_Never_Pushes_Head_Above_Cap()
    {
        foreach (float learningRate in new[] { 1e-5f, 1e-4f, 3e-4f, 1e-3f, 1e-2f })
        {
            float effective = learningRate * DetrTrainingSession.HeadGradientScale(learningRate);

            // 只放大、不缩小；且基础 lr 超过上限时不再往上顶（原来的 ×10 会到 1e-2）
            Assert.True(effective >= learningRate);
            Assert.True(effective <= Math.Max(learningRate, 1e-3f));
        }
    }
}
