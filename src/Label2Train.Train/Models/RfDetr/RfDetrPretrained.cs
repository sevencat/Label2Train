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

            (float[] Values, long[] Shape)? prepared = Prepare(data, entry.Shape, want);

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
    /// 决定这条张量怎么用：形状一致就直接用；<c>query_embed</c> / <c>refpoint_embed</c>
    /// 这类「官方存了全部 Group-DETR 组」的按行切前 <c>target[0]</c> 行。
    /// 其他形状不一致的一律放弃。
    /// </summary>
    private static (float[] Values, long[] Shape)? Prepare(float[] data, long[] source, long[] target)
    {
        if (source.SequenceEqual(target))
        {
            return (data, target);
        }

        // Group-DETR：组 0 排在最前，正好是前 target[0] 行
        if (source.Length == 2 && target.Length == 2 && source[1] == target[1]
            && target[0] > 0 && source[0] > target[0] && source[0] % target[0] == 0)
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
