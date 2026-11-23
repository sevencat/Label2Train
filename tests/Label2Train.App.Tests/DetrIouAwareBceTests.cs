using Label2Train.Train.Training.Detr;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Label2Train.App.Tests;

/// <summary>
/// IA-BCE 与官方 rfdetr 逐位对照。
/// <para>
/// 期望值由官方实现算出（<c>rfdetr.models.criterion._batched_detection_loss_tensors</c>，
/// torch 2.15 + seed 20261123，L=B=1、Q=4、C=2、匹配 query1→gt0(cls1)、query3→gt1(cls0)、
/// num_boxes=2、focal_alpha=0.25），因此这组数字代表「官方口径」而不是本项目的自洽结果。
/// </para>
/// </summary>
public sealed class DetrIouAwareBceTests
{
    private static readonly float[] Logits =
    [
        -0.6651385426521301f, -0.27876031398773193f, 1.077854871749878f, -2.1869657039642334f,
        -5.380314826965332f, 2.029109001159668f, 3.035881519317627f, 1.2394661903381348f,
    ];

    private static readonly float[] Boxes =
    [
        0.684701681137085f, 0.6959888339042664f, 0.4854077100753784f, 0.3729640245437622f,
        0.7992106080055237f, 0.288455605506897f, 0.28392094373703003f, 0.5868996977806091f,
        0.2510398030281067f, 0.7962103486061096f, 0.7034198641777039f, 0.4873809218406677f,
        0.30655044317245483f, 0.6425231099128723f, 0.21272575855255127f, 0.2382851392030716f,
    ];

    private static readonly float[] TargetBoxes =
    [
        0.7421932220458984f, 0.3276681900024414f, 0.39260441064834595f, 0.7269731760025024f,
        0.4562264084815979f, 0.27155157923698425f, 0.7338467836380005f, 0.7838349342346191f,
    ];

    /// <summary>官方 <c>_batched_detection_loss_tensors</c> 返回的 loss_ce。</summary>
    private const float OfficialLossCe = 3.590634822845459f;

    [Fact]
    public void MatchesOfficialIouAwareBce()
    {
        using Tensor logits = torch.tensor(Logits, new long[] { 1, 1, 4, 2 });
        using Tensor unusedTargets = torch.zeros(1, 1, 4, 2);

        // 匹配对：(query 1 → gt 0, class 1)、(query 3 → gt 1, class 0)
        long[] indices =
        [
            (((0 * 4) + 1) * 2) + 1,
            (((0 * 4) + 3) * 2) + 0,
        ];

        // IoU 由同一份 cxcywh 框算出（官方在 criterion 内部用 box_ops.elementwise_box_iou）
        float[] ious = [Iou(Boxes, 4, TargetBoxes, 0), Iou(Boxes, 12, TargetBoxes, 4)];

        var loss = new DetrLoss(2, 100, 100, new DetrLossOptions { IouAwareBce = true });

        using Tensor actual = loss.ClassificationLoss(logits, unusedTargets, indices, ious, 2);

        float value = actual.item<float>();

        Assert.True(
            MathF.Abs(value - OfficialLossCe) <= 1e-4f,
            $"actual {value:F6}  official {OfficialLossCe:F6}  iou {ious[0]:F6}/{ious[1]:F6}");
    }

    /// <summary>
    /// 旧实现（把标签换成 IoU² 的软标签 BCE）与官方不是一个东西：
    /// 同一组输入上数值不同，且早期 IoU 小的时候目标分会塌到 0.0x。
    /// </summary>
    [Fact]
    public void SoftTargetVariant_IsNotOfficial()
    {
        using Tensor logits = torch.tensor(Logits, new long[] { 1, 1, 4, 2 });

        float iou1 = Iou(Boxes, 4, TargetBoxes, 0);
        float iou2 = Iou(Boxes, 12, TargetBoxes, 4);

        var soft = new float[8];
        soft[(((0 * 4) + 1) * 2) + 1] = iou1 * iou1;
        soft[(((0 * 4) + 3) * 2) + 0] = iou2 * iou2;

        using Tensor targets = torch.tensor(soft, new long[] { 1, 1, 4, 2 });
        using Tensor elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction: torch.nn.Reduction.None);

        float softLoss = elementwise.sum().item<float>() / 2f;

        Assert.True(
            MathF.Abs(softLoss - OfficialLossCe) > 1e-3f,
            $"软标签 BCE 与官方 IA-BCE 不应同值，实际 {softLoss:F4} vs {OfficialLossCe:F4}");
    }

    private static float Iou(float[] boxes, int boxOffset, float[] targets, int targetOffset)
    {
        float ax1 = boxes[boxOffset] - (boxes[boxOffset + 2] / 2f);
        float ay1 = boxes[boxOffset + 1] - (boxes[boxOffset + 3] / 2f);
        float ax2 = boxes[boxOffset] + (boxes[boxOffset + 2] / 2f);
        float ay2 = boxes[boxOffset + 1] + (boxes[boxOffset + 3] / 2f);

        float bx1 = targets[targetOffset] - (targets[targetOffset + 2] / 2f);
        float by1 = targets[targetOffset + 1] - (targets[targetOffset + 3] / 2f);
        float bx2 = targets[targetOffset] + (targets[targetOffset + 2] / 2f);
        float by2 = targets[targetOffset + 1] + (targets[targetOffset + 3] / 2f);

        float width = MathF.Max(0f, MathF.Min(ax2, bx2) - MathF.Max(ax1, bx1));
        float height = MathF.Max(0f, MathF.Min(ay2, by2) - MathF.Max(ay1, by1));
        float intersection = width * height;
        float union = (MathF.Max(0f, ax2 - ax1) * MathF.Max(0f, ay2 - ay1))
            + (MathF.Max(0f, bx2 - bx1) * MathF.Max(0f, by2 - by1))
            - intersection;

        return union <= 0f ? 0f : intersection / union;
    }
}
