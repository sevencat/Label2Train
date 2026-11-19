using System.Globalization;
using Label2Train.Train.Core;
using Label2Train.Train.IO.Onnx;
using Label2Train.Train.Models.RfDetr;
using Label2Train.Train.Training;
using TorchSharp;
using static TorchSharp.torch;

namespace Label2Train.Train.Export;

/// <summary>
/// 把 <see cref="RfDetrDetector"/> 组装成一张 ONNX 图：DINOv2（复用
/// <see cref="DinoV2OnnxExporter.AppendCore"/>）+ 4 档投影 + 多层编码/解码（复用
/// <see cref="DeformAttnOnnxExporter.Append"/> 处理可变形注意力）+ 双 MLP 头，
/// 输出 <c>boxes</c> 与 <c>logits</c>。
/// <para>固定 batch = 1、固定输入尺寸（RF-DETR 逐图推理）。</para>
/// </summary>
public static class RfDetrOnnxExporter
{
    /// <summary>输入名。</summary>
    public const string InputName = "image";

    /// <summary>框输出名（sigmoid 后的归一化 cxcywh）。</summary>
    public const string BoxesOutputName = "boxes";

    /// <summary>分类输出名（原始 logits）。</summary>
    public const string LogitsOutputName = "logits";

    /// <summary>各档相对输入的下采样倍数。</summary>
    private static readonly int[] Strides = { 4, 8, 16, 32 };

    /// <summary>按检测器结构构图。</summary>
    public static OnnxModel Build(RfDetrDetector detector, int imageHeight, int imageWidth)
    {
        ArgumentNullException.ThrowIfNull(detector);

        DinoV2Config config = detector.BackboneConfig;

        // 两个约束叠在一起：主干按 patch 14 切图，检测头按 16 下采样，
        // 所以边长得是 lcm(14, 16) = 112 的倍数（560 就是官方那个尺寸）。
        if (!TrainingPresets.IsValidDetrImageSize(imageHeight) || !TrainingPresets.IsValidDetrImageSize(imageWidth))
        {
            throw new ArgumentException(
                $"RF-DETR 的输入边长必须是 {TrainingPresets.DetrImageSizeStride} 的倍数"
                + "（DINOv2 的 patch 14 与检测头的下采样 16 都要整除），例如 336 / 448 / 560 / 672；"
                + $"实际 {imageHeight}x{imageWidth}。",
                nameof(imageHeight));
        }
        if (config.WindowSize > 0)
        {
            throw new NotSupportedException(
                "窗口注意力主干（WindowSize > 0）暂不支持 ONNX 导出：窗口划分与移位需要额外算子，见 plan.md P2-3-4。");
        }

        if (detector.UsesTwoStage)
        {
            throw new NotSupportedException(
                "两阶段提议（enc_output + top-k）暂不支持 ONNX 导出，见 plan.md P2-3-5。");
        }

        if (detector.GroupDetr > 1)
        {
            throw new NotSupportedException(
                "Group-DETR（groupDetr > 1）暂不支持 ONNX 导出：导出器目前只处理单组 query，见 plan.md P2-3-5。");
        }

        IReadOnlyDictionary<string, Tensor> state = detector.state_dict();
        var builder = new OnnxGraphBuilder($"rfdetr-{detector.ClassCount}");
        builder.Input(InputName, OnnxDataType.Float, 1L, 3L, imageHeight, imageWidth);

        var context = new Context(builder, state);

        int levels = detector.LevelCount;
        var levelNames = new string[levels];
        var levelShapes = new (int Height, int Width)[levels];

        if (detector.UsesC2fProjector)
        {
            // 官方路径：主干 4 个深度层的中间输出 → 沿通道拼接 → C2f 融合 → 单档 256 维特征图
            int side = imageHeight / config.PatchSize;

            DinoV2OnnxExporter.AppendCoreTapped(
                builder,
                config,
                Scoped(state, "backbone."),
                InputName,
                imageHeight,
                imageWidth,
                detector.C2fTaps,
                out IReadOnlyList<string> tapMaps);

            string fused = tapMaps.Count == 1
                ? tapMaps[0]
                : context.AppendConcat(tapMaps, axis: 1, "rf.c2f.taps");

            levelNames[0] = context.AppendC2fProjector(
                "projector.stages.0", fused, config.EmbeddingSize * tapMaps.Count, side, side, "rf.c2f");
            levelShapes[0] = (side, side);
        }
        else
        {
            // 1) DINOv2 → tokens [1, 1+N, D]
            int tokens = DinoV2OnnxExporter.AppendCore(
                builder, config, Scoped(state, "backbone."), InputName, imageHeight, imageWidth, "rf.tokens");

            int side = imageHeight / config.PatchSize;

            // 2) 去掉 cls → [1,D,hp,wp]，再 1×1 投影到 256
            string patchTokens = context.AppendSlice("rf.tokens", "rf.patches", axis: 1, start: 1, end: tokens);
            string channelsFirst = context.AppendTranspose(patchTokens, "rf.patches.transposed", 0, 2, 1);
            string map = context.AppendReshape(
                channelsFirst, "rf.map", new long[] { 1, config.EmbeddingSize, side, side });

            string projected = context.AppendConv("input_proj", map, pads: 0, stride: 1);

            // 3) 多档投影：Conv1×1 + GroupNorm(32)，最细一档再接 Conv3×3(stride 2)
            for (int level = 0; level < levels; level++)
            {
                int stride = level == 0 ? Strides[level] / 2 : Strides[level];
                int height = Math.Max(1, imageHeight / stride);
                int width = Math.Max(1, imageWidth / stride);

                string resized = context.AppendResize(projected, $"rf.level{level}.resized", 256, height, width);
                string conv = context.AppendConv($"projector.{level}.conv", resized, pads: 0, stride: 1);
                string norm = context.AppendGroupNorm(
                    $"projector.{level}.norm", conv, RfDetrProjectionLevel.Groups, 256, height, width,
                    $"rf.level{level}.norm");

                if (level == 0)
                {
                    norm = context.AppendConv("projector.0.down", norm, pads: 1, stride: 2);
                    height = Math.Max(1, imageHeight / Strides[level]);
                    width = Math.Max(1, imageWidth / Strides[level]);
                }

                levelNames[level] = norm;
                levelShapes[level] = (height, width);
            }
        }

        int count = levelShapes[0].Height * levelShapes[0].Width;

        string memory = context.AppendReshape(
            context.AppendTranspose(levelNames[0], "rf.memory.transposed", 0, 2, 3, 1),
            "rf.memory", new long[] { 1, count, RfDetrDetector.EmbeddingSize });

        string reference = context.ConstantFloat(
            "rf.reference", new long[] { 1, count, levels, 2 },
            ReferenceGrid(levelShapes[0].Height, levelShapes[0].Width, levels));

        // 4) 编码器：每层「自注意力 → 可变形注意力 → FFN」，三处都有残差并把 norm 放在前面
        int heads = detector.EncoderLayerCount > 0
            ? Heads(state, "encoder_layers.0.deform_attn.", levels, detector.DeformPoints)
            : detector.DecoderHeads;

        for (int index = 0; index < detector.EncoderLayerCount; index++)
        {
            string prefix = $"encoder_layers.{index}.";

            string self = context.AppendSelfAttention(
                $"{prefix}self_attn",
                context.AppendLayerNorm($"{prefix}norm1", memory, $"rf.enc{index}.norm1"),
                count,
                detector.AttentionHeads,
                $"rf.enc{index}.self");

            memory = context.AppendAdd(memory, self, $"rf.enc{index}.sum1");

            string deformed = DeformAttnOnnxExporter.Append(
                builder,
                Scoped(state, $"{prefix}deform_attn."),
                context.AppendLayerNorm($"{prefix}norm2", memory, $"rf.enc{index}.norm2"),
                reference,
                levelNames,
                count,
                heads,
                detector.DeformPoints,
                levelShapes,
                RfDetrDetector.EmbeddingSize,
                $"rf.enc{index}.deform",
                scope: $"enc{index}.");

            memory = context.AppendAdd(memory, deformed, $"rf.enc{index}.sum2");

            string ffn = context.AppendLinear(
                $"{prefix}ffn0",
                context.AppendLayerNorm($"{prefix}norm3", memory, $"rf.enc{index}.norm3"),
                $"rf.enc{index}.ffn0");

            ffn = context.AppendRelu(ffn, $"rf.enc{index}.relu");
            ffn = context.AppendLinear($"{prefix}ffn1", ffn, $"rf.enc{index}.ffn1");
            memory = context.AppendAdd(memory, ffn, $"rf.enc{index}.sum3");
        }

        // 官方权重没有编码器层（encoderLayers = 0），这时也没有 encoder_norm
        if (detector.EncoderLayerCount > 0)
            memory = context.AppendLayerNorm("encoder_norm", memory, "rf.enc.out");

        // 5) 解码器：可学习 query + 逐层参考点细化（每层用输出预测参考点增量）
        string queries = context.ConstantFloat(
            "rf.queries",
            new long[] { 1, detector.Queries, RfDetrDetector.EmbeddingSize },
            TorchInterop.ToArray(context.State("query_embed")));

        // 官方参考点是 4 维 cxcywh（后两维是归一化宽高），2 维 xy 是早期的简化版
        int referenceDims = detector.UsesRefPointHead ? 4 : 2;
        float[] referenceValues = TorchInterop.ToArray(context.State("refpoint_embed"));
        var centerValues = new float[detector.Queries * 2];

        for (int query = 0; query < detector.Queries; query++)
        {
            centerValues[query * 2] = referenceValues[query * referenceDims];
            centerValues[query * 2 + 1] = referenceValues[query * referenceDims + 1];
        }

        string centers = context.ConstantFloat(
            "rf.centers", new long[] { 1, detector.Queries, 2 }, centerValues);

        // 4 维时每层都要先用 ref_point_head 把完整参考点编码回 query 空间
        string? references = referenceDims == 4
            ? context.ConstantFloat("rf.references", new long[] { 1, detector.Queries, 4 }, referenceValues)
            : null;

        int decoderHeads = Heads(state, "decoder_layers.0.deform_attn.", levels, detector.DeformPoints);

        for (int index = 0; index < detector.DecoderLayerCount; index++)
        {
            string prefix = $"decoder_layers.{index}.";

            string layerInput = queries;

            if (references is not null)
            {
                string encoded = context.AppendSineEmbed(
                    references,
                    referenceDims,
                    RfDetrDetector.EmbeddingSize,
                    detector.Queries,
                    $"rf.dec{index}.refpoint");

                string head = context.AppendMlp(
                    "ref_point_head.layers", encoded, $"rf.dec{index}.refpoint.mlp", layerCount: 2);

                layerInput = context.AppendAdd(queries, head, $"rf.dec{index}.refpoint.sum");
            }

            string points = context.AppendExpand(
                context.AppendUnsqueeze(centers, $"rf.dec{index}.unsqueeze", axis: 2),
                new long[] { 1, detector.Queries, levels, 2 },
                $"rf.dec{index}.points");

            string self = context.AppendSelfAttention(
                $"{prefix}self_attn",
                context.AppendLayerNorm($"{prefix}norm1", layerInput, $"rf.dec{index}.norm1"),
                detector.Queries,
                detector.AttentionHeads,
                $"rf.dec{index}.self");

            // 残差基必须是 layerInput（= queries + ref_point_head 编码），而不是上一层的 queries
            queries = context.AppendAdd(layerInput, self, $"rf.dec{index}.sum1");

            string deformed = DeformAttnOnnxExporter.Append(
                builder,
                Scoped(state, $"{prefix}deform_attn."),
                context.AppendLayerNorm($"{prefix}norm2", queries, $"rf.dec{index}.norm2"),
                points,
                levelNames,
                detector.Queries,
                decoderHeads,
                detector.DeformPoints,
                levelShapes,
                RfDetrDetector.EmbeddingSize,
                $"rf.dec{index}.deform",
                scope: $"dec{index}.");

            queries = context.AppendAdd(queries, deformed, $"rf.dec{index}.sum2");

            string ffn = context.AppendLinear(
                $"{prefix}ffn0",
                context.AppendLayerNorm($"{prefix}norm3", queries, $"rf.dec{index}.norm3"),
                $"rf.dec{index}.ffn0");

            ffn = context.AppendRelu(ffn, $"rf.dec{index}.relu");
            ffn = context.AppendLinear($"{prefix}ffn1", ffn, $"rf.dec{index}.ffn1");
            queries = context.AppendAdd(queries, ffn, $"rf.dec{index}.sum3");

            // 官方 lite_refpoint_refine：复用 bbox_embed 的前两维作为中心点增量（无额外参数）
            string boxOutput = context.AppendMlp("bbox_embed.layers", queries, $"rf.dec{index}.box", layerCount: 3);
            string delta = context.AppendSlice(boxOutput, $"rf.dec{index}.refine", axis: 2, start: 0, end: 2);
            centers = context.AppendSigmoid(
                context.AppendAdd(centers, delta, $"rf.dec{index}.center.sum"),
                $"rf.dec{index}.center");

            // 4 维参考点的后两维（宽高）不细化，沿用常量，只把中心替换成刚算出来的
            if (references is not null)
            {
                references = context.AppendConcat(
                    new[]
                    {
                        centers,
                        context.AppendSlice(references, $"rf.dec{index}.size", axis: 2, start: 2, end: 4),
                    },
                    axis: 2,
                    $"rf.dec{index}.references");
            }
        }

        queries = context.AppendLayerNorm("decoder_norm", queries, "rf.dec.out");

        // 6) 检测头：bbox_embed 三层 MLP + class_embed 单层 Linear（与官方键名一致）
        string boxes = context.AppendSigmoid(
            context.AppendMlp("bbox_embed.layers", queries, "rf.box", layerCount: 3), "rf.boxes");

        string logits = context.AppendLinear("class_embed", queries, "rf.logits");

        builder.Output(BoxesOutputName, OnnxDataType.Float, 1L, detector.Queries, 4);
        builder.Output(LogitsOutputName, OnnxDataType.Float, 1L, detector.Queries, detector.ClassCount);

        builder.Node("Identity", new[] { boxes }, new[] { BoxesOutputName });
        builder.Node("Identity", new[] { logits }, new[] { LogitsOutputName });

        return new OnnxModel(builder.Build());
    }

    /// <summary>构图并写出文件。</summary>
    public static void Save(RfDetrDetector detector, string path, int imageHeight, int imageWidth)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        string? directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        Build(detector, imageHeight, imageWidth).Save(path);
    }

    /// <summary>取某个子模块的 state_dict 子集（去掉模块前缀）。</summary>
    private static Dictionary<string, Tensor> Scoped(IReadOnlyDictionary<string, Tensor> state, string prefix)
    {
        var scoped = new Dictionary<string, Tensor>(StringComparer.Ordinal);

        foreach ((string key, Tensor tensor) in state)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                scoped[key[prefix.Length..]] = tensor;
        }

        return scoped;
    }

    /// <summary>从 <c>attention_weights.weight</c> 的行数（头数 × 档数 × 采样点）反推头数。</summary>
    private static int Heads(IReadOnlyDictionary<string, Tensor> state, string prefix, int levels, int points)
    {
        long rows = state[prefix + "attention_weights.weight"].shape[0];
        return (int)(rows / (levels * points));
    }

    /// <summary>归一化参考点网格（与 <c>RfDetrDetector.ReferenceGrid</c> 完全一致）。</summary>
    private static float[] ReferenceGrid(int height, int width, int levels)
    {
        int count = height * width;
        var values = new float[count * levels * 2];

        for (int index = 0; index < count; index++)
        {
            int row = index / width;
            int column = index % width;
            float x = (column + 0.5f) / width;
            float y = (row + 0.5f) / height;

            for (int level = 0; level < levels; level++)
            {
                values[(index * levels + level) * 2] = x;
                values[(index * levels + level) * 2 + 1] = y;
            }
        }

        return values;
    }

    /// <summary>粘合层的导出上下文（卷积 / Resize / Linear / LayerNorm / GroupNorm / 自注意力等）。</summary>
    private sealed class Context
    {
        private readonly Dictionary<string, Tensor> _state;
        private readonly HashSet<string> _names = new(StringComparer.Ordinal);

        public Context(OnnxGraphBuilder builder, IReadOnlyDictionary<string, Tensor> state)
        {
            Builder = builder;
            _state = new Dictionary<string, Tensor>(state.Count, StringComparer.Ordinal);

            foreach ((string key, Tensor tensor) in state)
                _state[key] = tensor;
        }

        public OnnxGraphBuilder Builder { get; }

        public Tensor State(string key) =>
            _state.TryGetValue(key, out Tensor? tensor)
                ? tensor
                : throw new InvalidDataException($"state_dict 缺少 '{key}'。");

        public string Next(string hint)
        {
            string candidate = hint;
            int suffix = 0;

            while (!_names.Add(candidate))
                candidate = hint + "." + (++suffix).ToString(CultureInfo.InvariantCulture);

            return candidate;
        }

        public string ConstantFloat(string hint, long[] dims, float[] values)
        {
            string name = Next(hint);
            Builder.Initializer(OnnxTensor.FromFloat32(name, dims, values));
            return name;
        }

        public string ConstantInt64(string hint, params long[] values)
        {
            string name = Next(hint);
            Builder.Initializer(OnnxTensor.FromInt64(name, new long[] { values.Length }, values));
            return name;
        }

        public string AppendTensor(string key) =>
            ConstantFloat(key, State(key).shape, TorchInterop.ToArray(State(key)));

        /// <summary>按指定形状写 initializer（用于广播，例如把 <c>[C]</c> 写成 <c>[C,1,1]</c>）。</summary>
        public string AppendTensorReshaped(string key, long[] dims) =>
            ConstantFloat(key, dims, TorchInterop.ToArray(State(key)));

        public string AppendSlice(string input, string hint, int axis, long start, long end)
        {
            string output = Next(hint);
            Builder.Node(
                "Slice",
                new[]
                {
                    input,
                    ConstantInt64(hint + ".starts", start),
                    ConstantInt64(hint + ".ends", end),
                    ConstantInt64(hint + ".axes", axis),
                },
                new[] { output });

            return output;
        }

        public string AppendTranspose(string input, string hint, params int[] perm)
        {
            string output = Next(hint);
            Builder.Node(
                "Transpose",
                new[] { input },
                new[] { output },
                OnnxAttribute.Ints("perm", perm.Select(value => (long)value).ToArray()));

            return output;
        }

        public string AppendReshape(string input, string hint, long[] shape)
        {
            string output = Next(hint);
            Builder.Node("Reshape", new[] { input, ConstantInt64(hint + ".shape", shape) }, new[] { output });
            return output;
        }

        public string AppendUnsqueeze(string input, string hint, int axis)
        {
            string output = Next(hint);
            Builder.Node("Unsqueeze", new[] { input, ConstantInt64(hint + ".axes", axis) }, new[] { output });
            return output;
        }

        public string AppendExpand(string input, long[] shape, string hint)
        {
            string output = Next(hint);
            Builder.Node("Expand", new[] { input, ConstantInt64(hint + ".shape", shape) }, new[] { output });
            return output;
        }

        /// <summary>把 2D 权重转置后写 initializer（MatMul 需要 <c>[in,out]</c>）。</summary>
        public string AppendTensorTransposed(string key, long[] dims)
        {
            Tensor transposed = State(key).transpose(0, 1).contiguous();
            return ConstantFloat(key + ".transposed", dims, TorchInterop.ToArray(transposed));
        }

        /// <summary>带偏置的 Conv（权重直接来自 state_dict，形状 <c>[out,in,k,k]</c>）。</summary>
        public string AppendConv(string prefix, string input, int pads, int stride)
        {
            string output = Next("rf.conv." + prefix);
            Builder.Node(
                "Conv",
                new[] { input, AppendTensor(prefix + ".weight"), AppendTensor(prefix + ".bias") },
                new[] { output },
                OnnxAttribute.Ints("pads", pads, pads, pads, pads),
                OnnxAttribute.Ints("strides", stride, stride),
                OnnxAttribute.Ints("dilations", 1, 1),
                OnnxAttribute.Int("group", 1));

            return output;
        }

        /// <summary>双线性 Resize（<c>align_corners=False</c> ⇔ <c>half_pixel</c>）。</summary>
        public string AppendResize(string input, string hint, int channels, int height, int width)
        {
            string sizes = ConstantInt64(hint + ".sizes", 1, channels, height, width);

            string output = Next(hint);
            Builder.Node(
                "Resize",
                new[] { input, string.Empty, string.Empty, sizes },
                new[] { output },
                OnnxAttribute.String("mode", "linear"),
                OnnxAttribute.String("coordinate_transformation_mode", "half_pixel"));

            return output;
        }

        public string AppendLinear(string prefix, string input, string hint)
        {
            Tensor weight = State(prefix + ".weight");

            string weights = ConstantFloat(
                prefix + ".weight",
                new long[] { weight.shape[1], weight.shape[0] },
                TorchInterop.ToArray(weight.transpose(0, 1).contiguous()));

            string product = Next(hint + ".matmul");
            Builder.Node("MatMul", new[] { input, weights }, new[] { product });

            string output = Next(hint);
            Builder.Node("Add", new[] { product, AppendTensor(prefix + ".bias") }, new[] { output });
            return output;
        }

        public string AppendMatMul(string left, string right, string hint)
        {
            string output = Next(hint);
            Builder.Node("MatMul", new[] { left, right }, new[] { output });
            return output;
        }

        public string AppendSoftmax(string input, string hint, int axis)
        {
            string output = Next(hint);
            Builder.Node("Softmax", new[] { input }, new[] { output }, OnnxAttribute.Int("axis", axis));
            return output;
        }

        public string AppendLayerNorm(string prefix, string input, string hint)
        {
            string output = Next(hint);
            Builder.Node(
                "LayerNormalization",
                new[] { input, AppendTensor(prefix + ".weight"), AppendTensor(prefix + ".bias") },
                new[] { output },
                OnnxAttribute.Int("axis", -1),
                OnnxAttribute.Float("epsilon", 1e-5f));

            return output;
        }

        /// <summary>单轴 ReduceMean，保留维度（<c>axes</c> 是属性，不是输入）。</summary>
        public string AppendReduceMean(string input, string hint, int axis)
        {
            string output = Next(hint);
            Builder.Node(
                "ReduceMean",
                new[] { input },
                new[] { output },
                OnnxAttribute.Ints("axes", axis),
                OnnxAttribute.Int("keepdims", 1));

            return output;
        }

        public string AppendRelu(string input, string hint)
        {
            string output = Next(hint);
            Builder.Node("Relu", new[] { input }, new[] { output });
            return output;
        }

        public string AppendSigmoid(string input, string hint)
        {
            string output = Next(hint);
            Builder.Node("Sigmoid", new[] { input }, new[] { output });
            return output;
        }

        public string AppendSqrt(string input, string hint)
        {
            string output = Next(hint);
            Builder.Node("Sqrt", new[] { input }, new[] { output });
            return output;
        }

        public string AppendAdd(string left, string right, string hint)
        {
            string output = Next(hint);
            Builder.Node("Add", new[] { left, right }, new[] { output });
            return output;
        }

        public string AppendSub(string left, string right, string hint)
        {
            string output = Next(hint);
            Builder.Node("Sub", new[] { left, right }, new[] { output });
            return output;
        }

        public string AppendMul(string left, string right, string hint)
        {
            string output = Next(hint);
            Builder.Node("Mul", new[] { left, right }, new[] { output });
            return output;
        }

        public string AppendDiv(string left, string right, string hint)
        {
            string output = Next(hint);
            Builder.Node("Div", new[] { left, right }, new[] { output });
            return output;
        }

        /// <summary>
        /// 官方 MLP 头：<c>layers.0..layers.(n-1)</c> 全是 Linear，ReLU 在层间函数式串联
        /// （与 <c>RfDetrMlp</c> 的键名/顺序一致）。
        /// </summary>
        public string AppendMlp(string prefix, string input, string hint, int layerCount = 3)
        {
            string hidden = input;

            for (int index = 0; index < layerCount; index++)
            {
                hidden = AppendLinear($"{prefix}.{index}", hidden, $"{hint}.{index}");

                if (index < layerCount - 1)
                    hidden = AppendRelu(hidden, $"{hint}.relu{index}");
            }

            return hidden;
        }

        /// <summary>双 MLP 头：Linear → ReLU → Linear → ReLU → Linear（键名 <c>.0/.2/.4</c>）。</summary>
        public string AppendHead(string prefix, string input, string hint)
        {
            string hidden = AppendLinear(prefix + ".0", input, hint + ".0");
            hidden = AppendRelu(hidden, hint + ".relu0");
            hidden = AppendLinear(prefix + ".2", hidden, hint + ".2");
            hidden = AppendRelu(hidden, hint + ".relu2");
            return AppendLinear(prefix + ".4", hidden, hint + ".4");
        }

        /// <summary>
        /// GroupNorm：<c>[1,C,H,W]</c> → <c>[1,G,P]</c> 上求均值/方差，再还原形状并做仿射。
        /// ONNX 没有原生 GroupNorm，这里按定义展开（<c>eps</c> 与 TorchSharp 默认一致）。
        /// </summary>
        public string AppendGroupNorm(
            string prefix,
            string input,
            int groups,
            int channels,
            int height,
            int width,
            string hint)
        {
            long perGroup = (long)channels / groups * height * width;

            string flat = AppendReshape(input, hint + ".flat", new long[] { 1, groups, perGroup });
            string mean = AppendReduceMean(flat, hint + ".mean", axis: 2);
            string centered = AppendSub(flat, mean, hint + ".centered");
            string squared = AppendMul(centered, centered, hint + ".squared");
            string variance = AppendReduceMean(squared, hint + ".variance", axis: 2);
            string eps = ConstantFloat(hint + ".eps", new long[] { 1 }, new[] { 1e-5f });
            string standard = AppendSqrt(AppendAdd(variance, eps, hint + ".variance.eps"), hint + ".std");
            string normalised = AppendDiv(centered, standard, hint + ".normalised");
            string restored = AppendReshape(
                normalised, hint + ".restored", new long[] { 1, channels, height, width });

            string scaled = AppendMul(
                restored,
                AppendTensorReshaped(prefix + ".weight", new long[] { channels, 1, 1 }),
                hint + ".scaled");

            return AppendAdd(
                scaled,
                AppendTensorReshaped(prefix + ".bias", new long[] { channels, 1, 1 }),
                hint + ".shifted");
        }

        /// <summary>无偏置 Conv（<c>RfDetrConvBn</c> 里的 conv 是 <c>bias:false</c>）。</summary>
        private string AppendConvNoBias(string prefix, string input, int pads, int stride)
        {
            string output = Next("rf.conv." + prefix);
            Builder.Node(
                "Conv",
                new[] { input, AppendTensor(prefix + ".weight") },
                new[] { output },
                OnnxAttribute.Ints("pads", pads, pads, pads, pads),
                OnnxAttribute.Ints("strides", stride, stride),
                OnnxAttribute.Ints("dilations", 1, 1),
                OnnxAttribute.Int("group", 1));

            return output;
        }

        /// <summary>
        /// 用**批统计量**的 BatchNorm（<c>track_running_stats=false</c>，checkpoint 里也没有
        /// <c>running_mean/var</c>）。ONNX 的 BatchNormalization 只认 running stats，所以按定义
        /// 展开：固定 batch=1 时就是在 H·W 上求均值/方差。
        /// </summary>
        private string AppendBatchNorm(string prefix, string input, int channels, int height, int width, string hint)
        {
            long spatial = (long)height * width;

            string flat = AppendReshape(input, hint + ".flat", new long[] { channels, spatial });
            string mean = AppendReduceMean(flat, hint + ".mean", axis: 1);
            string centered = AppendSub(flat, mean, hint + ".centered");
            string squared = AppendMul(centered, centered, hint + ".squared");
            string variance = AppendReduceMean(squared, hint + ".variance", axis: 1);
            string eps = ConstantFloat(hint + ".eps", new long[] { 1 }, new[] { 1e-5f });
            string standard = AppendSqrt(AppendAdd(variance, eps, hint + ".variance.eps"), hint + ".std");
            string normalised = AppendDiv(centered, standard, hint + ".normalised");
            string restored = AppendReshape(
                normalised, hint + ".restored", new long[] { 1, channels, height, width });

            string scaled = AppendMul(
                restored,
                AppendTensorReshaped(prefix + ".weight", new long[] { channels, 1, 1 }),
                hint + ".scaled");

            return AppendAdd(
                scaled,
                AppendTensorReshaped(prefix + ".bias", new long[] { channels, 1, 1 }),
                hint + ".shifted");
        }

        /// <summary>Conv+BN（<c>RfDetrConvBn</c>）：1×1/3×3 卷积后接批统计量 BN。</summary>
        private string AppendConvBn(
            string prefix, string input, int inChannels, int outChannels, int pads, int height, int width, string hint)
        {
            _ = inChannels;
            string conv = AppendConvNoBias(prefix + ".conv", input, pads, stride: 1);

            return AppendBatchNorm(prefix + ".bn", conv, outChannels, height, width, hint + ".bn");
        }

        /// <summary>通道 LayerNorm（<c>RfDetrChannelNorm</c>）：NCHW → NHWC → LayerNorm(axis=-1) → NCHW。</summary>
        private string AppendChannelNorm(string prefix, string input, int channels, string hint)
        {
            string channelsLast = AppendTranspose(input, hint + ".channels.last", 0, 2, 3, 1);

            string normalised = Next(hint + ".norm");
            Builder.Node(
                "LayerNormalization",
                new[]
                {
                    channelsLast,
                    AppendTensorReshaped(prefix + ".weight", new long[] { channels }),
                    AppendTensor(prefix + ".bias"),
                },
                new[] { normalised },
                OnnxAttribute.Int("axis", -1),
                OnnxAttribute.Float("epsilon", 1e-6f));

            return AppendTranspose(normalised, hint, 0, 3, 1, 2);
        }

        /// <summary>
        /// C2f 投影器（官方 <c>backbone.0.projector</c>）：主干多层特征图拼成 <c>[1,4D,h,w]</c> 后
        /// <c>cv1 → 对半切 → 若干 Bottleneck → 拼接 → cv2</c>，最后一次通道 LayerNorm。
        /// </summary>
        public string AppendC2fProjector(
            string prefix, string input, int inChannels, int height, int width, string hint)
        {
            string block = prefix + ".0";
            int outChannels = (int)State(block + ".cv1.conv.weight").shape[0];
            int hidden = outChannels / 2;

            string cv1 = AppendConvBn(
                block + ".cv1", input, inChannels, outChannels, pads: 0, height, width, hint + ".cv1");

            var parts = new List<string>
            {
                AppendSlice(cv1, hint + ".split0", axis: 1, start: 0, end: hidden),
            };

            string current = AppendSlice(cv1, hint + ".split1", axis: 1, start: hidden, end: outChannels);
            parts.Add(current);

            int bottlenecks = 0;

            while (_state.ContainsKey($"{block}.m.{bottlenecks}.cv1.conv.weight"))
                bottlenecks++;

            for (int index = 0; index < bottlenecks; index++)
            {
                string bottleneck = $"{block}.m.{index}";

                string middle = AppendConvBn(
                    bottleneck + ".cv1", current, hidden, hidden, pads: 1, height, width, hint + $".m{index}.cv1");

                string output = AppendConvBn(
                    bottleneck + ".cv2", middle, hidden, hidden, pads: 1, height, width, hint + $".m{index}.cv2");

                current = AppendAdd(current, output, hint + $".m{index}.residual");
                parts.Add(current);
            }

            string merged = AppendConcat(parts, axis: 1, hint + ".concat");

            string cv2 = AppendConvBn(
                block + ".cv2",
                merged,
                outChannels + hidden * bottlenecks,
                outChannels,
                pads: 0,
                height,
                width,
                hint + ".cv2");

            return AppendChannelNorm(prefix + ".1", cv2, outChannels, hint + ".norm");
        }

        public string AppendConcat(IReadOnlyList<string> inputs, int axis, string hint)
        {
            string output = Next(hint);
            Builder.Node("Concat", inputs.ToArray(), new[] { output }, OnnxAttribute.Int("axis", axis));
            return output;
        }

        /// <summary>
        /// 参考点的正弦编码：<c>[1,Q,n]</c> → <c>[1,Q,2·n·feats]</c>，逐维
        /// <c>sin/cos(x·2π / 10000^(i/feats))</c>，<c>feats = D/n</c>
        /// （与 <c>RfDetrDetector.SineEmbed</c> 完全一致）。
        /// </summary>
        public string AppendSineEmbed(string input, int dimensions, int embedding, int queries, string hint)
        {
            // feats 固定按 D/4 算（与 RfDetrDetector.SineEmbed 一致，官方即 4 个坐标共 2D 维）
            int feats = embedding / 4;

            // 频率表用 torch 算出来写成常量，避免 MathF.Pow 与 torch.pow 在 float32 上的细微分歧
            using Tensor exponent = torch.arange(feats, dtype: ScalarType.Float32) / feats;
            using Tensor scaleTensor = torch.pow(torch.tensor(10000.0, dtype: ScalarType.Float32), exponent);
            float[] scales = scaleTensor.data<float>().ToArray();

            string expanded = AppendUnsqueeze(input, hint + ".unsqueeze", axis: 3);
            string doubled = AppendMul(
                expanded,
                ConstantFloat(hint + ".two.pi", new long[] { 1 }, new[] { (float)(2.0 * Math.PI) }),
                hint + ".doubled");

            string angles = AppendDiv(
                doubled, ConstantFloat(hint + ".scales", new long[] { 1, 1, 1, feats }, scales), hint + ".angles");

            string sine = Next(hint + ".sin");
            Builder.Node("Sin", new[] { angles }, new[] { sine });

            string cosine = Next(hint + ".cos");
            Builder.Node("Cos", new[] { angles }, new[] { cosine });

            string concatenated = AppendConcat(new[] { sine, cosine }, axis: 3, hint + ".encoded");

            return AppendReshape(
                concatenated, hint, new long[] { 1, queries, dimensions * 2 * feats });
        }

        /// <summary>把 <c>[1,L,D]</c> 拆成每个头 <c>[1,heads,L,hd]</c>。</summary>
        private string AppendHeads(string input, string hint, int tokens, int heads, int headDim)
        {
            string reshaped = AppendReshape(
                input, hint + ".reshape", new long[] { 1, tokens, heads, headDim });

            return AppendTranspose(reshaped, hint, 0, 2, 1, 3);
        }

        /// <summary>
        /// 多头自注意力，键名与 <c>nn.MultiheadAttention</c> 一致
        /// （<c>in_proj_weight [3D,D]</c> / <c>in_proj_bias [3D]</c> / <c>out_proj</c>）。
        /// </summary>
        public string AppendSelfAttention(string prefix, string input, int tokens, int heads, string hint)
        {
            int embedding = (int)(State(prefix + ".in_proj_weight").shape[0] / 3);
            int headDim = embedding / heads;

            string packedWeights = AppendTensorTransposed(
                prefix + ".in_proj_weight", new long[] { embedding, embedding * 3L });
            string packed = AppendAdd(
                AppendMatMul(input, packedWeights, hint + ".qkv.matmul"),
                AppendTensor(prefix + ".in_proj_bias"),
                hint + ".qkv");

            string q = AppendHeads(
                AppendSlice(packed, hint + ".q", axis: 2, start: 0, end: embedding),
                hint + ".q.heads", tokens, heads, headDim);

            string k = AppendHeads(
                AppendSlice(packed, hint + ".k", axis: 2, start: embedding, end: 2L * embedding),
                hint + ".k.heads", tokens, heads, headDim);

            string v = AppendHeads(
                AppendSlice(packed, hint + ".v", axis: 2, start: 2L * embedding, end: 3L * embedding),
                hint + ".v.heads", tokens, heads, headDim);

            string scores = AppendMatMul(
                q,
                AppendTranspose(k, hint + ".k.transposed", 0, 1, 3, 2),
                hint + ".scores");

            string scale = ConstantFloat(
                hint + ".scale", new long[] { 1 }, new[] { (float)(1.0 / Math.Sqrt(headDim)) });

            string weights = AppendSoftmax(AppendMul(scores, scale, hint + ".scaled"), hint + ".weights", -1);
            string context = AppendMatMul(weights, v, hint + ".context");

            string merged = AppendReshape(
                AppendTranspose(context, hint + ".context.transposed", 0, 2, 1, 3),
                hint + ".merged",
                new long[] { 1, tokens, embedding });

            string outWeights = AppendTensorTransposed(
                prefix + ".out_proj.weight", new long[] { embedding, embedding });

            string projected = AppendMatMul(merged, outWeights, hint + ".out.matmul");
            return AppendAdd(projected, AppendTensor(prefix + ".out_proj.bias"), hint + ".out");
        }
    }
}

