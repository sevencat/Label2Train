using Label2Train.Train.Core;
using Label2Train.Train.IO.SafeTensors;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Models.RfDetr;

/// <summary>
/// 把官方 RF-DETR 权重（<c>model.safetensors</c>）装载进本项目模型。
/// <para>
/// 只做形状匹配：对不上就跳过并在结果里报出来，不抛异常——比如分类头的类别数
/// （官方 COCO 91）和用户数据集的类别数不一样，那一层就该保持随机初始化。
/// Group-DETR 的 <c>query_feat</c> / <c>refpoint_embed</c> 存了全部组，组 0 排在最前，
/// 这里按行切片只取组 0。
/// </para>
/// </summary>
public static class RfDetrPretrained
{
    /// <summary>载入权重，返回覆盖情况。</summary>
    public static RfDetrPretrainedReport Apply(
        Module<Tensor, Tensor> detector,
        string path,
        Device device,
        RfDetrWeightsInfo? info = null)
    {
        ArgumentNullException.ThrowIfNull(detector);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        info ??= RfDetrWeightsProbe.Inspect(path)
            ?? throw new InvalidDataException(
                $"'{path}' 不是官方 RF-DETR 权重（没有 {RfDetrWeightsProbe.RootPrefix} 前缀的键）。");

        Dictionary<string, Tensor> state = detector.state_dict();
        var mapped = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var qkv = new Dictionary<string, float[]>(StringComparer.Ordinal);
        var qkvParts = new Dictionary<string, int>(StringComparer.Ordinal);
        int skipped = 0;

        using SafeTensorFile file = SafeTensorFile.Open(path);

        foreach (SafeTensorEntry entry in file.Tensors)
        {
            if (!RfDetrPretrainedMapper.TryMap(entry.Name, out string target, out int slot))
            {
                // 检查点里有、本项目结构没有的（mask_token / enc_output / 分组辅助头等）
                skipped++;

                continue;
            }

            if (!state.TryGetValue(target, out Tensor? current))
            {
                // 本项目没建这层（例如编码器 0 层时的 encoder_norm）
                skipped++;

                continue;
            }

            long[] want = current.shape;
            float[] data = file.ReadFloat32(entry.Name);

            if (slot >= 0)
            {
                // 官方分开存 q/k/v，本项目是融合的 [3E, E]（权重）或 [3E]（偏置）
                if (want.Length != entry.Shape.Length || want[0] != entry.Shape[0] * 3)
                {
                    skipped++;

                    continue;
                }

                if (!qkv.TryGetValue(target, out float[]? buffer))
                {
                    buffer = new float[data.Length * 3];
                    qkv[target] = buffer;
                }

                Array.Copy(data, 0, buffer, (long)slot * data.Length, data.Length);
                qkvParts[target] = qkvParts.GetValueOrDefault(target) + 1;

                continue;
            }

            // 位置编码：官方权重按自己的输入边长（576 → 36×36 个 patch）存，用户换成别的
            // imgsz 时 token 数对不上。这里按官方 DINOv2 的做法做双三次插值；不能像其他层那样
            // 直接跳过——跳过会让 pos_embed 停在构造时的全 0，模型彻底失去位置信息，
            // 表现为 loss 缓慢下降但 mAP 几十轮都是 0。
            if (target == "backbone.pos_embed" && !entry.Shape.SequenceEqual(want))
            {
                Tensor? resized = ResizePositionEmbedding(current, data, entry.Shape, device);

                if (resized is null)
                {
                    skipped++;

                    continue;
                }

                mapped[target] = resized;

                continue;
            }

            (float[] Values, long[] Shape)? prepared = Prepare(target, data, entry.Shape, want);

            if (prepared is null)
            {
                skipped++;

                continue;
            }

            mapped[target] = TorchInterop.FromArray(prepared.Value.Values, prepared.Value.Shape, device);
        }

        foreach ((string target, float[] buffer) in qkv)
        {
            if (qkvParts.GetValueOrDefault(target) < 3)
            {
                skipped++;
                qkvParts.Remove(target);

                continue;
            }

            mapped[target] = TorchInterop.FromArray(buffer, state[target].shape, device);
        }

        if (mapped.Count == 0)
        {
            throw new InvalidDataException($"'{path}' 里没有一个张量能对上当前模型结构。");
        }

        detector.load_state_dict(mapped, strict: false);

        string[] untouched =
        [
            .. state.Keys
                .Where(key => !mapped.ContainsKey(key))
                .OrderBy(key => key, StringComparer.Ordinal),
        ];

        return new RfDetrPretrainedReport(mapped.Count, skipped, untouched);
    }

    /// <summary>
    /// 把官方的位置编码（<c>[1, 1 + h·w, D]</c>，第一个 token 是 cls）按双三次插值到目标尺寸。
    /// 两边都必须是正方形 patch 网格；否则返回 <c>null</c>，交给调用方按「跳过」处理。
    /// </summary>
    private static Tensor? ResizePositionEmbedding(Tensor current, float[] data, long[] source, Device device)
    {
        long[] want = current.shape;

        if (source.Length != 3 || want.Length != 3 || source[2] != want[2])
            return null;

        long sourcePatches = source[1] - 1;
        long targetPatches = want[1] - 1;

        if (sourcePatches <= 0 || targetPatches <= 0)
            return null;

        int sourceSide = (int)Math.Round(Math.Sqrt(sourcePatches));
        int targetSide = (int)Math.Round(Math.Sqrt(targetPatches));

        if ((long)sourceSide * sourceSide != sourcePatches
            || (long)targetSide * targetSide != targetPatches)
        {
            return null;
        }

        using Tensor embeddings = TorchInterop.FromArray(data, source, device);

        Tensor patches = embeddings
            .narrow(1, 1, sourcePatches)
            .view(1, sourceSide, sourceSide, want[2])
            .permute(0, 3, 1, 2);

        Tensor resized = torch.nn.functional.interpolate(
            patches,
            new long[] { targetSide, targetSide },
            mode: InterpolationMode.Bicubic,
            align_corners: false);

        Tensor flat = resized.permute(0, 2, 3, 1).contiguous().view(1, targetPatches, want[2]);

        return torch.cat(new[] { embeddings.narrow(1, 0, 1), flat }, 1);
    }

    /// <summary>
    /// 决定这条张量怎么用：形状一致就直接用；<c>query_embed</c> / <c>refpoint_embed</c>
    /// 这类「官方存了全部 Group-DETR 组、且 query 数与用户设置无关」的按行切前 <c>target[0]</c> 行。
    /// 其他形状不一致的一律放弃。
    /// </summary>
    /// <param name="target">模型侧参数名（只对 query / refpoint 允许按行切片）。</param>
    /// <param name="data">检查点里的原始数据。</param>
    /// <param name="source">检查点里的形状。</param>
    /// <param name="target">模型侧形状。</param>
    private static (float[] Values, long[] Shape)? Prepare(
        string name, float[] data, long[] source, long[] target)
    {
        if (source.SequenceEqual(target))
        {
            return (data, target);
        }

        // 只对 query / refpoint 做按行切片：官方 query_feat 存的是 Group-DETR 的 n 组 × 300 行，
        // 组 0 排在最前；用户把 Queries 调小（官方 300 → 我们 20/50/100）时同样取前若干行即可。
        // 其他权重（尤其 class_embed 的 91 类）绝对不能按行截——类别语义会对不上。
        bool sliceable = name is "query_embed" or "refpoint_embed";

        if (sliceable && source.Length == 2 && target.Length == 2 && source[1] == target[1]
            && target[0] > 0 && source[0] > target[0])
        {
            long count = target[0] * target[1];

            if (count > data.Length)
            {
                return null;
            }

            var sliced = new float[count];
            Array.Copy(data, sliced, count);

            return (sliced, target);
        }

        return null;
    }
}
