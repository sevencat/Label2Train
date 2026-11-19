using Label2Train.Train.Training.Detr;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// loss 分项累积：日志里的 cls / l1 / giou 必须能和 epoch 平均 loss 对上，
/// 否则拿它判断「卡在分类项还是框回归」就会看错。
/// </summary>
public sealed class DetrLossPartsTests
{
    [Fact]
    public void AverageParts_Sums_To_Mean_Total_Loss()
    {
        const int classes = 2;
        const int queries = 4;

        var loss = new DetrLoss(classes, 32, 32);
        var random = new Random(20261119);

        using Tensor prediction = torch.tensor(
            RandomValues(random, queries * (4 + classes)), [1, queries, 4 + classes]);

        // [1,2,6]：第一个 GT 有效（像素 xyxy），第二个是填充
        using Tensor target = torch.tensor(
            new float[] { 1f, 0f, 2f, 3f, 6f, 8f, 0f, 1f, 0f, 0f, 0f, 0f },
            [1, 2, 6]);

        using Tensor first = loss.Compute(prediction, target);
        using Tensor second = loss.Compute(prediction, target);

        double expected = (first.item<float>() + second.item<float>()) / 2;
        (float classPart, float l1Part, float giouPart) = loss.AverageParts;

        Assert.Equal(expected, classPart + l1Part + giouPart, 4);

        loss.ResetParts();

        Assert.Equal((0f, 0f, 0f), loss.AverageParts);
    }

    private static float[] RandomValues(Random random, int count)
    {
        var values = new float[count];

        for (int index = 0; index < count; index++)
        {
            values[index] = (float)random.NextDouble();
        }

        return values;
    }
}
