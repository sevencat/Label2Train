using Label2Train.Train.Core;
using Label2Train.Train.Data.Geometry;
using Label2Train.Train.Data.Images;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training;
using Label2Train.Train.Training.Yolo;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Label2Train.App.Tests;

/// <summary>
/// 收敛性冒烟：一张图能不能被背下来。
/// <para>
/// 这是「加新模型族」之前的前置验证——如果 v8 在一张图上都 overfit 不了，
/// 问题在 loss / 匹配 / 数据处理，换 backbone 不会有任何帮助。
/// </para>
/// </summary>
public sealed class OverfitSmokeTests(ITestOutputHelper output)
{
    private const int Size = 128;
    private const int Classes = 2;
    private const int Steps = 600;

    private static readonly YoloTargetBox[] Targets =
    [
        new(0, 16f, 16f, 48f, 48f),
        new(1, 72f, 72f, 112f, 112f),
    ];

    [Fact]
    [Trait("Category", "Slow")]
    public void YoloV8_Overfits_Single_Image()
    {
        RunOverfit(YoloModelFamily.V8);
    }

    /// <summary>
    /// YOLO26 的双头链路：one-to-many + one-to-one 一起算损失，reg_max = 1 没有 DFL，
    /// 框分支直接回归距离。能背下这张图就说明新结构与新损失接对了。
    /// </summary>
    [Fact]
    [Trait("Category", "Slow")]
    public void YoloV26_Overfits_Single_Image()
    {
        RunOverfit(YoloModelFamily.V26);
    }

    private void RunOverfit(YoloModelFamily family)
    {
        // 权重初始化必须固定：否则全量跑时其他测试动过 RNG，验收阈值会时过时不过
        torch.random.manual_seed(20261120);

        Device device = torch.CPU;

        using YoloDetectionModel model = YoloArchitectures.Create(device, family, YoloScale.N, Classes);
        model.Head.ApplyBiasPrior(Size);

        using Tensor input = BuildInput(Size, Targets).to(device);
        using Tensor target = BuildTarget(Targets, device);

        ILoss loss = YoloEndToEndDetectionLoss.CreateFor(
            model, Size, Size, Classes, trackComponents: true);

        using var optimizer = torch.optim.AdamW(model.parameters(), lr: 0.003, weight_decay: 0.0);

        float first = 0f;
        float last = 0f;

        for (int step = 0; step < Steps; step++)
        {
            using var scope = torch.NewDisposeScope();
            model.train();
            optimizer.zero_grad();

            using Tensor prediction = model.forward(input);
            using Tensor value = ComputeLoss(loss, model, prediction, target);
            value.backward();
            optimizer.step();

            last = TorchInterop.ToScalar(value);
            if (step == 0)
            {
                first = last;
            }

            if ((step + 1) % 100 == 0)
            {
                YoloLossParts parts = LastParts(loss);
                output.WriteLine(
                    $"step {step + 1,3}  loss {last:F3}  box {parts.Box:F3}  cls {parts.Class:F3}  dfl {parts.Dfl:F3}");
            }
        }

        model.eval();
        IReadOnlyList<YoloDetection> detections;

        using (torch.no_grad())
        using (var scope = torch.NewDisposeScope())
        {
            using Tensor prediction = model.forward(input);
            detections = YoloPredictionDecoder.Decode(
                prediction, Size, Size, Classes, model.Head.RegressionMax, confidenceThreshold: 0.3f);

            foreach (YoloDetection detection in detections)
            {
                output.WriteLine(
                    $"det cls {detection.ClassId} score {detection.Score:F3} "
                    + $"({detection.X1:F1},{detection.Y1:F1})-({detection.X2:F1},{detection.Y2:F1})");
            }
        }

        YoloLossParts finalParts = LastParts(loss);
        output.WriteLine($"box/cls/dfl: {finalParts.Box:F3} / {finalParts.Class:F3} / {finalParts.Dfl:F3}");

        Assert.True(last < first * 0.4f, $"200 步后 loss 只从 {first:F3} 降到 {last:F3}，链路可能有问题");

        float best = 0f;
        foreach (YoloTargetBox box in Targets)
        {
            float iou = 0f;
            foreach (YoloDetection detection in detections)
            {
                iou = Math.Max(iou, IntersectionOverUnion(detection, box));
            }

            best = Math.Max(best, iou);
        }

        Assert.True(best > 0.5f, $"最高 IoU 只有 {best:F3}：loss 降了但框没贴住目标");
    }

    /// <summary>双头模型要额外把 one-to-one 分支的输出塞进损失，其它家族走单头。</summary>
    private static Tensor ComputeLoss(ILoss loss, YoloDetectionModel model, Tensor prediction, Tensor target)
    {
        if (loss is YoloEndToEndDetectionLoss dual
            && model is ILossExtrasProvider provider
            && provider.GetLossExtras() is { } extras)
        {
            return dual.Compute(prediction, target, extras);
        }

        return loss.Compute(prediction, target);
    }

    private static YoloLossParts LastParts(ILoss loss) => loss switch
    {
        YoloEndToEndDetectionLoss dual => dual.LastParts,
        YoloDetectionLoss single => single.LastParts,
        _ => new YoloLossParts(0f, 0f, 0f, 0f),
    };

    /// <summary>合成一张带纹理的 128×128 图：框内画高对比度色块。</summary>
    private static Tensor BuildInput(int size, IReadOnlyList<YoloTargetBox> boxes)
    {
        var rgb = new byte[size * size * 3];

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int offset = (y * size + x) * 3;
                bool inside = false;

                foreach (YoloTargetBox box in boxes)
                {
                    if (x >= box.X1 && x < box.X2 && y >= box.Y1 && y < box.Y2)
                    {
                        inside = true;
                        break;
                    }
                }

                byte value = inside ? (byte)230 : (byte)((x * 3 + y * 5) % 40);
                rgb[offset] = value;
                rgb[offset + 1] = inside ? (byte)40 : (byte)value;
                rgb[offset + 2] = inside ? (byte)40 : (byte)value;
            }
        }

        var sample = new YoloImageSample(
            "synthetic:overfit",
            size,
            size,
            size,
            size,
            rgb,
            LetterboxTransform.Compute(size, size, size, size),
            boxes,
            flipped: false);

        return TorchInterop.FromArray(sample.ToChw(), [1, 3, size, size], torch.CPU);
    }

    private static Tensor BuildTarget(IReadOnlyList<YoloTargetBox> boxes, Device device)
    {
        var values = new float[boxes.Count * YoloBatchBuilder.TargetColumns];

        for (int index = 0; index < boxes.Count; index++)
        {
            YoloTargetBox box = boxes[index];
            int offset = index * YoloBatchBuilder.TargetColumns;
            values[offset] = 1f;
            values[offset + 1] = box.ClassId;
            values[offset + 2] = box.X1;
            values[offset + 3] = box.Y1;
            values[offset + 4] = box.X2;
            values[offset + 5] = box.Y2;
        }

        return TorchInterop.FromArray(values, [1, boxes.Count, YoloBatchBuilder.TargetColumns], device);
    }

    private static float IntersectionOverUnion(YoloDetection detection, YoloTargetBox target)
    {
        float x1 = Math.Max(detection.X1, target.X1);
        float y1 = Math.Max(detection.Y1, target.Y1);
        float x2 = Math.Min(detection.X2, target.X2);
        float y2 = Math.Min(detection.Y2, target.Y2);

        float intersection = Math.Max(0f, x2 - x1) * Math.Max(0f, y2 - y1);
        float union = detection.Width * detection.Height + target.Width * target.Height - intersection;

        return union <= 0f ? 0f : intersection / union;
    }
}
