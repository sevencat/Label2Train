using Label2Train.Train.Training.Yolo;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Label2Train.App.Tests;

/// <summary>
/// TAL 分配的语义回归：冲突锚点必须被丢弃（官方 <c>count_tensor &gt; 1 → 0</c>），
/// 以及 <c>topk2</c> 二次筛选与 YOLO26 双头权重衰减的数值。
/// <para>
/// 这几条曾经被实现成「冲突锚点归给分数更高的 GT」，会让密集目标的正样本偏多、梯度偏大，
/// 而且和 ultralytics 对不上（对拍时正样本 117 vs 90、梯度差 8.5%）。
/// </para>
/// </summary>
public sealed class YoloTalAssignerTests
{
    /// <summary>两个 GT 完全相同、两个类别的最佳锚点也是同一个 → 该锚点被两个 GT 抢 → 丢弃。</summary>
    [Fact]
    public void Conflicting_Anchor_Is_Dropped_Not_Stolen()
    {
        using var scope = torch.NewDisposeScope();

        Tensor anchors = torch.tensor(new float[] { 8, 8, 24, 8, 40, 8, 56, 8 }).reshape(4, 2);

        // 两个类别都在锚点 2 上最高（0.9），其余 0.1 → 两个 GT 的 top-1 都落在锚点 2
        Tensor scores = torch.tensor(new float[]
        {
            0.1f, 0.1f,
            0.1f, 0.1f,
            0.9f, 0.9f,
            0.1f, 0.1f,
        }).reshape(1, 4, 2);

        // 每个锚点的预测框都是同尺寸，让选择完全由分类分数决定
        Tensor boxes = torch.tensor(new float[]
        {
            0, 0, 16, 16,
            16, 0, 32, 16,
            32, 0, 48, 16,
            48, 0, 64, 16,
        }).reshape(1, 4, 4);

        Tensor gtBoxes = torch.tensor(new float[] { 0, 0, 64, 16, 0, 0, 64, 16 }).reshape(1, 2, 4);
        Tensor gtClasses = torch.tensor(new long[] { 0, 1 }).reshape(1, 2);
        Tensor gtMask = torch.ones(new long[] { 1, 2 });

        var assigner = new TaskAlignedAssigner(classCount: 2, topK: 1);
        TalResult result = assigner.Assign(scores, boxes, anchors, gtClasses, gtBoxes, gtMask);

        Assert.Equal(0, (int)result.Foreground.sum().item<long>());
    }

    /// <summary>单 GT 时没有冲突；topk2 = 1 只留对齐度最高的那一个锚点。</summary>
    [Fact]
    public void TopK2_Keeps_Single_Anchor_Per_Target()
    {
        using var scope = torch.NewDisposeScope();

        Tensor anchors = torch.tensor(new float[] { 8, 8, 24, 8, 40, 8, 56, 8 }).reshape(4, 2);
        Tensor scores = torch.tensor(new float[] { 0.1f, 0.1f, 0.9f, 0.1f }).reshape(1, 4, 1);
        Tensor boxes = torch.tensor(new float[]
        {
            0, 0, 16, 16,
            16, 0, 32, 16,
            32, 0, 48, 16,
            48, 0, 64, 16,
        }).reshape(1, 4, 4);

        Tensor gtBoxes = torch.tensor(new float[] { 0, 0, 64, 16 }).reshape(1, 1, 4);
        Tensor gtClasses = torch.tensor(new long[] { 0 }).reshape(1, 1);
        Tensor gtMask = torch.ones(new long[] { 1, 1 });

        // topK = 7 会被裁到锚点数 4：先全选，再由 topk2 = 1 收敛到 1 个
        var assigner = new TaskAlignedAssigner(classCount: 1, topK: 7, topk2: 1);
        TalResult result = assigner.Assign(scores, boxes, anchors, gtClasses, gtBoxes, gtMask);

        Assert.Equal(1, (int)result.Foreground.sum().item<long>());
        Assert.Equal(2, (int)result.Foreground.reshape(-1).to(ScalarType.Int64).argmax().item<long>());
    }

    /// <summary>topk2 不指定时等于 topK，不做二次筛选（v8 / v11 的默认行为）。</summary>
    [Fact]
    public void TopK2_Defaults_To_TopK()
    {
        using var scope = torch.NewDisposeScope();

        Tensor anchors = torch.tensor(new float[] { 8, 8, 24, 8, 40, 8, 56, 8 }).reshape(4, 2);
        Tensor scores = torch.tensor(new float[] { 0.1f, 0.2f, 0.9f, 0.3f }).reshape(1, 4, 1);
        Tensor boxes = torch.tensor(new float[]
        {
            0, 0, 16, 16,
            16, 0, 32, 16,
            32, 0, 48, 16,
            48, 0, 64, 16,
        }).reshape(1, 4, 4);

        Tensor gtBoxes = torch.tensor(new float[] { 0, 0, 64, 16 }).reshape(1, 1, 4);
        Tensor gtClasses = torch.tensor(new long[] { 0 }).reshape(1, 1);
        Tensor gtMask = torch.ones(new long[] { 1, 1 });

        var assigner = new TaskAlignedAssigner(classCount: 1, topK: 2);
        TalResult result = assigner.Assign(scores, boxes, anchors, gtClasses, gtBoxes, gtMask);

        Assert.Equal(2, (int)result.Foreground.sum().item<long>());
    }

    /// <summary>YOLO26 双头权重：0.8 起步，按 epoch 线性衰减到 0.1，one-to-one 取补数。</summary>
    [Fact]
    public void EndToEnd_Weights_Decay_To_Official_Schedule()
    {
        var loss = new YoloEndToEndDetectionLoss(320, 320, 3);

        Assert.Equal(0.8f, loss.OneToManyWeight, 5);
        Assert.Equal(0.2f, loss.OneToOneWeight, 5);

        // 官方 decay(updates) = max(1 - updates/(epochs-1), 0) * (0.8 - 0.1) + 0.1
        Assert.Equal(0.7631579f, YoloEndToEndDetectionLoss.Decay(1, 20), 5);
        Assert.Equal(0.4315789f, YoloEndToEndDetectionLoss.Decay(10, 20), 5);
        Assert.Equal(0.1f, YoloEndToEndDetectionLoss.Decay(19, 20), 5);
        Assert.Equal(0.1f, YoloEndToEndDetectionLoss.Decay(50, 20), 5);
        Assert.Equal(0.1f, YoloEndToEndDetectionLoss.Decay(1, 1), 5);

        loss.OnEpochEnd(0, 20);
        Assert.Equal(0.7631579f, loss.OneToManyWeight, 5);
        Assert.Equal(1 - loss.OneToManyWeight, loss.OneToOneWeight, 5);

        loss.OnEpochEnd(18, 20);
        Assert.Equal(0.1f, loss.OneToManyWeight, 5);
        Assert.Equal(0.9f, loss.OneToOneWeight, 5);
    }
}
