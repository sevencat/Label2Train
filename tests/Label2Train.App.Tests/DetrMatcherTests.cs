using Label2Train.Train.Training.Detr;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 匈牙利匹配的正确性：新实现把「GT 当行、query 当列」以缩小成本矩阵，
/// 必须仍然给出**全局最小成本**的单射（穷举作参考解）。
/// </summary>
public sealed class DetrMatcherTests
{
    [Theory]
    [InlineData(1, 3)]
    [InlineData(1, 5)]
    [InlineData(2, 3)]
    [InlineData(4, 2)]
    public void Match_Returns_Minimum_Cost_Assignment(int groups, int targetsPerImage)
    {
        const int queries = 20;
        const int classes = 3;
        const int batch = 2;
        const int maxTargets = 5;

        var random = new Random(20261119);
        var selection = new DetrLossOptions();

        float[] boxValues = RandomBoxes(random, batch * queries);
        float[] logitValues = RandomLogits(random, batch * queries * classes);
        float[] targetValues = RandomBoxes(random, batch * maxTargets);
        var classValues = new long[batch * maxTargets];
        var maskValues = new long[batch * maxTargets];

        for (int image = 0; image < batch; image++)
        {
            for (int target = 0; target < maxTargets; target++)
            {
                classValues[image * maxTargets + target] = random.Next(classes);
                maskValues[image * maxTargets + target] = target < targetsPerImage ? 1 : 0;
            }
        }

        using Tensor boxes = torch.tensor(boxValues, [batch, queries, 4]);
        using Tensor logits = torch.tensor(logitValues, [batch, queries, classes]);
        using Tensor gtBoxes = torch.tensor(targetValues, [batch, maxTargets, 4]);
        using Tensor gtClasses = torch.tensor(classValues, [batch, maxTargets]);
        using Tensor gtMask = torch.tensor(maskValues, [batch, maxTargets]);

        IReadOnlyList<SimpleDetrImageMatch> matches = DetrMatcher.Match(
            boxes, logits, gtBoxes, gtClasses, gtMask, selection, groups);

        int groupSize = queries / groups;
        float[] scores = ToFloat(logits.sigmoid());
        float[] boxData = ToFloat(boxes);
        float[] targetData = ToFloat(gtBoxes);
        long[] classData = gtClasses.data<long>().ToArray();

        for (int image = 0; image < batch; image++)
        {
            var byGroup = new Dictionary<int, List<(int Query, int Target)>>();

            foreach ((int query, int target) in matches[image].Pairs)
            {
                int group = query / groupSize;

                if (!byGroup.TryGetValue(group, out List<(int, int)>? list))
                {
                    list = [];
                    byGroup[group] = list;
                }

                list.Add((query, target));
            }

            Assert.Equal(groups * targetsPerImage, matches[image].Pairs.Count);

            for (int group = 0; group < groups; group++)
            {
                List<(int Query, int Target)> pairs = byGroup[group];

                // 单射：同一张图里一个 query 只配一个 GT
                Assert.Equal(pairs.Count, pairs.Select(pair => pair.Query).Distinct().Count());

                double actual = 0;

                foreach ((int query, int target) in pairs)
                {
                    Assert.InRange(query, group * groupSize, (group * groupSize) + groupSize - 1);
                    actual += Cost(image, query, target, boxData, targetData, classData, scores, classes, maxTargets,
                        queries, selection);
                }

                double optimal = BestAssignmentCost(
                    group, groupSize, targetsPerImage, image, boxData, targetData, classData, scores, classes,
                    maxTargets, queries, selection);

                Assert.True(
                    Math.Abs(actual - optimal) < 1e-4,
                    $"第 {image} 张图第 {group} 组：实际成本 {actual}，最优 {optimal}");
            }
        }
    }

    /// <summary>穷举组内「选 targetsPerImage 个不同 query」的所有排列，取最小总成本。</summary>
    private static double BestAssignmentCost(
        int group,
        int groupSize,
        int targetsPerImage,
        int image,
        float[] boxValues,
        float[] targetValues,
        long[] classValues,
        float[] scores,
        int classes,
        int maxTargets,
        int queries,
        DetrLossOptions selection)
    {
        double best = double.PositiveInfinity;

        void Walk(int position, int usedMask, double total)
        {
            if (position == targetsPerImage)
            {
                best = Math.Min(best, total);
                return;
            }

            for (int local = 0; local < groupSize; local++)
            {
                if ((usedMask & (1 << local)) != 0)
                    continue;

                int query = (group * groupSize) + local;
                double next = total + Cost(
                    image, query, position, boxValues, targetValues, classValues, scores, classes, maxTargets,
                    queries, selection);

                Walk(position + 1, usedMask | (1 << local), next);
            }
        }

        Walk(0, 0, 0.0);
        return best;
    }

    private static double Cost(
        int image,
        int query,
        int target,
        float[] boxValues,
        float[] targetValues,
        long[] classValues,
        float[] scores,
        int classes,
        int maxTargets,
        int queries,
        DetrLossOptions selection)
    {
        long classId = classValues[(image * maxTargets) + target];

        (double px, double py, double pw, double ph) = Box(boxValues, (image * queries + query) * 4);
        (double tx, double ty, double tw, double th) = Box(targetValues, (image * maxTargets + target) * 4);

        double score = scores[((image * queries) + query) * classes + classId];
        double distance = Math.Abs(px - tx) + Math.Abs(py - ty) + Math.Abs(pw - tw) + Math.Abs(ph - th);

        return (-selection.ClassWeight * score)
            + (selection.L1Weight * distance)
            + (selection.GiouWeight * (1.0 - Giou(px, py, pw, ph, tx, ty, tw, th)));
    }

    private static (double X, double Y, double W, double H) Box(float[] values, int offset)
        => (values[offset], values[offset + 1], values[offset + 2], values[offset + 3]);

    private static double Giou(
        double ax, double ay, double aw, double ah,
        double bx, double by, double bw, double bh)
    {
        double ax1 = ax - (aw / 2), ay1 = ay - (ah / 2), ax2 = ax + (aw / 2), ay2 = ay + (ah / 2);
        double bx1 = bx - (bw / 2), by1 = by - (bh / 2), bx2 = bx + (bw / 2), by2 = by + (bh / 2);

        double interWidth = Math.Max(0, Math.Min(ax2, bx2) - Math.Max(ax1, bx1));
        double interHeight = Math.Max(0, Math.Min(ay2, by2) - Math.Max(ay1, by1));
        double intersection = interWidth * interHeight;

        double areaA = Math.Max(0, ax2 - ax1) * Math.Max(0, ay2 - ay1);
        double areaB = Math.Max(0, bx2 - bx1) * Math.Max(0, by2 - by1);
        double union = (areaA + areaB - intersection) + 1e-12;

        double enclose = (Math.Max(ax2, bx2) - Math.Min(ax1, bx1))
            * (Math.Max(ay2, by2) - Math.Min(ay1, by1)) + 1e-12;

        return (intersection / union) - ((enclose - union) / enclose);
    }

    private static float[] RandomBoxes(Random random, int count)
    {
        var values = new float[count * 4];

        for (int index = 0; index < count; index++)
        {
            int offset = index * 4;
            values[offset] = 0.2f + ((float)random.NextDouble() * 0.6f);
            values[offset + 1] = 0.2f + ((float)random.NextDouble() * 0.6f);
            values[offset + 2] = 0.05f + ((float)random.NextDouble() * 0.3f);
            values[offset + 3] = 0.05f + ((float)random.NextDouble() * 0.3f);
        }

        return values;
    }

    private static float[] RandomLogits(Random random, int count)
    {
        var values = new float[count];

        for (int index = 0; index < count; index++)
            values[index] = (float)((random.NextDouble() * 6) - 3);

        return values;
    }

    private static float[] ToFloat(Tensor tensor) => tensor.data<float>().ToArray();
}
