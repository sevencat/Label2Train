namespace Label2Train.Train.Models.RfDetr;

/// <summary>
/// 官方键名 → 本项目键名。两者的命名本来就对齐过，差异只有三处：
/// 主干多一层 <c>backbone.0.encoder.encoder.</c> 前缀、FFN 叫 <c>linear1/linear2</c>（本项目 <c>ffn0/ffn1</c>）、
/// 可变形注意力叫 <c>cross_attn</c>（本项目 <c>deform_attn</c>）。
/// 另外官方把 q/k/v 分开存，本项目是一个融合的 <c>qkv</c>，需要按 q→k→v 顺序拼接。
/// </summary>
internal static class RfDetrPretrainedMapper
{
    /// <summary>映射一条参数名；<paramref name="slot"/> 大于等于 0 表示这是 q/k/v 的第几段。</summary>
    internal static bool TryMap(string source, out string target, out int slot)
    {
        target = string.Empty;
        slot = -1;

        if (source.StartsWith("backbone.0.projector.", StringComparison.Ordinal))
        {
            target = "projector." + source["backbone.0.projector.".Length..];

            return true;
        }

        if (source.StartsWith(RfDetrWeightsProbe.DinoPrefix, StringComparison.Ordinal))
        {
            return TryMapDino(source, out target, out slot);
        }

        if (TrySplitIndex(source, RfDetrWeightsProbe.DecoderPrefix, out string index, out string tail))
        {
            return TryMapDecoder(index, tail, out target);
        }

        if (source.StartsWith("transformer.decoder.norm.", StringComparison.Ordinal))
        {
            target = "decoder_norm." + source["transformer.decoder.norm.".Length..];

            return true;
        }

        // 参考点编码头（4 维 cxcywh → sin/cos → 两层 MLP），键名只差 transformer.decoder. 前缀
        if (source.StartsWith("transformer.decoder.ref_point_head.", StringComparison.Ordinal))
        {
            target = source["transformer.decoder.".Length..];

            return true;
        }

        switch (source)
        {
            case "query_feat.weight":
                target = "query_embed";

                return true;

            case "refpoint_embed.weight":
                target = "refpoint_embed";

                return true;
        }

        // bbox_embed.layers.0.* 与 class_embed.* 的名字本来就一致
        if (source.StartsWith("class_embed.", StringComparison.Ordinal)
            || source.StartsWith("bbox_embed.", StringComparison.Ordinal))
        {
            target = source;

            return true;
        }

        return false;
    }

    private static bool TryMapDino(string source, out string target, out int slot)
    {
        target = string.Empty;
        slot = -1;

        string dino = RfDetrWeightsProbe.DinoPrefix;
        string rest = source[dino.Length..];

        if (rest == "embeddings.cls_token")
        {
            target = "backbone.cls_token";

            return true;
        }

        if (rest == "embeddings.position_embeddings")
        {
            target = "backbone.pos_embed";

            return true;
        }

        const string patch = "embeddings.patch_embeddings.projection.";
        if (rest.StartsWith(patch, StringComparison.Ordinal))
        {
            target = "backbone.patch_embed.proj." + rest[patch.Length..];

            return true;
        }

        const string norm = "layernorm.";
        if (rest.StartsWith(norm, StringComparison.Ordinal))
        {
            target = "backbone.norm." + rest[norm.Length..];

            return true;
        }

        if (!TrySplitIndex(source, RfDetrWeightsProbe.BlockPrefix, out string index, out string tail))
        {
            // embeddings.mask_token 之类本项目没有的
            return false;
        }

        string prefix = $"backbone.blocks.{index}.";

        const string attention = "attention.attention.";
        if (tail.StartsWith(attention, StringComparison.Ordinal))
        {
            // query / key / value 三段拼成融合的 qkv
            string name = tail[attention.Length..];

            slot = name switch
            {
                var value when value.StartsWith("query.", StringComparison.Ordinal) => 0,
                var value when value.StartsWith("key.", StringComparison.Ordinal) => 1,
                var value when value.StartsWith("value.", StringComparison.Ordinal) => 2,
                _ => -1,
            };

            if (slot < 0)
            {
                return false;
            }

            target = prefix + "attn.qkv." + name[(name.IndexOf('.') + 1)..];

            return true;
        }

        const string output = "attention.output.dense.";
        if (tail.StartsWith(output, StringComparison.Ordinal))
        {
            target = prefix + "attn.proj." + tail[output.Length..];

            return true;
        }

        // LayerScale：官方 layer_scale1.lambda1 ↔ 本项目 ls1.gamma
        if (tail.StartsWith("layer_scale1.lambda1", StringComparison.Ordinal))
        {
            target = prefix + "ls1.gamma";

            return true;
        }

        if (tail.StartsWith("layer_scale2.lambda1", StringComparison.Ordinal))
        {
            target = prefix + "ls2.gamma";

            return true;
        }

        // mlp.fc1 / mlp.fc2 / norm1 / norm2 名字一致
        target = prefix + tail;

        return true;
    }

    private static bool TryMapDecoder(string index, string tail, out string target)
    {
        string prefix = $"decoder_layers.{index}.";

        const string selfAttention = "self_attn.";
        if (tail.StartsWith(selfAttention, StringComparison.Ordinal))
        {
            target = prefix + tail;

            return true;
        }

        // 官方 FFN 叫 linear1/linear2，本项目是 ffn0/ffn1
        const string linear1 = "linear1.";
        if (tail.StartsWith(linear1, StringComparison.Ordinal))
        {
            target = prefix + "ffn0." + tail[linear1.Length..];

            return true;
        }

        const string linear2 = "linear2.";
        if (tail.StartsWith(linear2, StringComparison.Ordinal))
        {
            target = prefix + "ffn1." + tail[linear2.Length..];

            return true;
        }

        // 官方 cross_attn ↔ 本项目 deform_attn
        const string crossAttention = "cross_attn.";
        if (tail.StartsWith(crossAttention, StringComparison.Ordinal))
        {
            target = prefix + "deform_attn." + tail[crossAttention.Length..];

            return true;
        }

        // norm1 / norm2 / norm3 名字一致
        target = prefix + tail;

        return true;
    }

    /// <summary>把 <c>prefix + 序号 + '.' + 尾部</c> 拆开。</summary>
    private static bool TrySplitIndex(string source, string prefix, out string index, out string tail)
    {
        index = string.Empty;
        tail = string.Empty;

        if (!source.StartsWith(prefix, StringComparison.Ordinal))
        {
            return false;
        }

        string rest = source[prefix.Length..];
        int dot = rest.IndexOf('.');

        if (dot <= 0 || !rest[..dot].All(char.IsAsciiDigit))
        {
            return false;
        }

        index = rest[..dot];
        tail = rest[(dot + 1)..];

        return true;
    }
}
