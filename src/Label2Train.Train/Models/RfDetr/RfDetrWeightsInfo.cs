using Label2Train.Train.IO.SafeTensors;

namespace Label2Train.Train.Models.RfDetr;

/// <summary>
/// 官方 RF-DETR 检查点（HuggingFace 格式，键名 <c>backbone.0.*</c> / <c>transformer.*</c>）
/// 的结构参数，全部从张量形状反推，不读 config.json。
/// </summary>
public sealed record RfDetrWeightsInfo(
    int PatchSize,
    int EmbeddingSize,
    int Depth,
    int Heads,
    int PosTokens,
    int DecoderLayers,
    int DecoderHeads,
    int DeformLevels,
    int DeformPoints,
    int Queries,
    int GroupDetr,
    int ClassCount,
    bool HasC2fProjector)
{
    /// <summary>pos_embed 对应的正方形输入边长（36 个 patch × 16 = 576）。</summary>
    public int ImageSize
    {
        get
        {
            int side = (int)Math.Round(Math.Sqrt(Math.Max(1, PosTokens - 1)));
            return Math.Max(PatchSize, side * PatchSize);
        }
    }
}

/// <summary>权重载入结果：载入多少、跳过多少、哪些参数仍然是随机初始化的。</summary>
public sealed record RfDetrPretrainedReport(int Loaded, int Skipped, IReadOnlyList<string> Untouched)
{
    /// <summary>给训练日志用的一句话摘要。</summary>
    public string Summary => Untouched.Count == 0
        ? $"载入 {Loaded} 个张量（检查点里另有 {Skipped} 个用不上）"
        : $"载入 {Loaded} 个张量（跳过 {Skipped} 个，{Untouched.Count} 个参数保持随机初始化："
            + $"{string.Join("、", Untouched.Take(3))}{(Untouched.Count > 3 ? " 等" : string.Empty)}）";
}

/// <summary>官方 RF-DETR 检查点的结构探测。</summary>
public static class RfDetrWeightsProbe
{
    /// <summary>官方检查点的键名前缀；有这个前缀才按「整网权重」处理。</summary>
    public const string RootPrefix = "backbone.0.";

    internal const string DinoPrefix = "backbone.0.encoder.encoder.";

    internal const string BlockPrefix = "backbone.0.encoder.encoder.encoder.layer.";

    internal const string DecoderPrefix = "transformer.decoder.layers.";

    /// <summary>检查点里是不是官方 RF-DETR 整网权重（而不是纯 DINOv2 主干）。</summary>
    public static bool IsOfficialRfDetr(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using SafeTensorFile file = SafeTensorFile.Open(path);

        return file.Tensors.Any(entry => entry.Name.StartsWith(RootPrefix, StringComparison.Ordinal));
    }

    /// <summary>从张量形状反推结构参数；不是官方整网权重时返回 <c>null</c>。</summary>
    public static RfDetrWeightsInfo? Inspect(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using SafeTensorFile file = SafeTensorFile.Open(path);

        Dictionary<string, SafeTensorEntry> byName = file.Tensors
            .ToDictionary(entry => entry.Name, StringComparer.Ordinal);

        string patchWeight = DinoPrefix + "embeddings.patch_embeddings.projection.weight";
        string position = DinoPrefix + "embeddings.position_embeddings";

        if (!byName.TryGetValue(patchWeight, out SafeTensorEntry? patchShapeEntry)
            || patchShapeEntry.Shape.Length != 4
            || !byName.TryGetValue(position, out SafeTensorEntry? posEntry))
        {
            return null;
        }

        int embedding = (int)patchShapeEntry.Shape[0];
        int patchSize = (int)patchShapeEntry.Shape[2];
        int posTokens = posEntry.Shape.Length >= 2 ? (int)posEntry.Shape[1] : 0;

        int depth = CountIndexed(byName, BlockPrefix, ".attention.attention.query.weight");
        int decoderLayers = CountIndexed(byName, DecoderPrefix, ".self_attn.in_proj_weight");

        if (embedding <= 0 || patchSize <= 0 || posTokens <= 1 || depth == 0 || decoderLayers == 0)
        {
            return null;
        }

        bool hasC2f = byName.Keys.Any(name => name.StartsWith("backbone.0.projector.", StringComparison.Ordinal));

        // C2f 投影器只产生 1 档特征；老的多档投影器按 4 档算
        int levels = hasC2f ? 1 : RfDetrDetector.MaxLevels;
        int points = 2;
        int deformHeads = 0;

        if (byName.TryGetValue($"{DecoderPrefix}0.cross_attn.sampling_offsets.weight", out SafeTensorEntry? offsets))
        {
            // 行数 = 头数 × 档数 × 采样点 × 2（dy/dx）
            deformHeads = Math.Max(1, (int)offsets.Shape[0] / (levels * points * 2));
        }

        const int officialQueries = 300;
        int queries = officialQueries;
        int groupDetr = 1;

        if (byName.TryGetValue("query_feat.weight", out SafeTensorEntry? queryFeat))
        {
            int rows = (int)queryFeat.Shape[0];

            // Group-DETR 把每组 query 连续排：组 0 排在最前，正好取前 queries 行
            if (rows % officialQueries == 0)
            {
                groupDetr = Math.Max(1, rows / officialQueries);
            }
            else
            {
                queries = rows;
            }
        }

        int classCount = byName.TryGetValue("class_embed.weight", out SafeTensorEntry? classEmbed)
            ? (int)classEmbed.Shape[0]
            : 0;

        return new RfDetrWeightsInfo(
            patchSize,
            embedding,
            depth,
            Math.Max(1, embedding / 64),
            posTokens,
            decoderLayers,
            deformHeads,
            levels,
            points,
            queries,
            groupDetr,
            classCount,
            hasC2f);
    }

    private static int CountIndexed(
        IReadOnlyDictionary<string, SafeTensorEntry> byName,
        string prefix,
        string suffix)
    {
        int count = 0;

        while (byName.ContainsKey($"{prefix}{count}{suffix}"))
        {
            count++;
        }

        return count;
    }
}
