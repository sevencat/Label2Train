using Label2Train.Train.Export;
using Label2Train.Train.Models.RfDetr;
using OnnxSharp.Core;
using TorchSharp;
using TorchSharp.Modules;
using Xunit;
using static TorchSharp.torch;
// OnnxSharp 也有 <c>RfDetrDetector</c>，这里只借用它的解码器，避免与训练侧的检测器重名
using RfDetrDecoder = OnnxSharp.Vision.RfDetrDecoder;
using RfDetrOptions = OnnxSharp.Vision.RfDetrOptions;
using Detection = OnnxSharp.Vision.Detection;
using ChannelNormalization = OnnxSharp.Vision.ChannelNormalization;

namespace Label2Train.App.Tests;

/// <summary>
/// 导出的 ONNX 必须与训练时的 torch 前向数值一致：结构对不上（Reshape 顺序、BN 的
/// running/batch 统计量、pos_embed 插值）时文件照样能生成，只是框全乱——这里逐元素对拍。
/// </summary>
public sealed class RfDetrOnnxParityTests
{
    /// <summary>小模型（多档投影器 + 无 ref_point_head），跑得快，覆盖主干 / 投影 / 解码器。</summary>
    [Fact]
    public void Exported_RfDetr_Onnx_Matches_Torch_Forward()
    {
        const int classCount = 2;
        const int queries = 20;
        const int size = 336;   // 同时是 14（patch）与 16（位置网格）的倍数
        string path = Path.Combine(Path.GetTempPath(), $"label2train-rfdetr-parity-{Guid.NewGuid():N}.onnx");

        try
        {
            float[] pixels;
            float[] expected;

            using (var detector = new RfDetrDetector(
                classCount,
                queries,
                new DinoV2Config(ImageSize: 518, EmbeddingSize: 64, Depth: 2, Heads: 4),
                heads: 8,
                device: torch.CPU))
            {
                // 权重不用随机初始化：见 FillDeterministically 的说明
                FillDeterministically(detector);

                using Tensor input = torch.rand(new long[] { 1, 3, size, size });
                pixels = input.flatten().data<float>().ToArray();

                using Tensor output = detector.forward(input);
                expected = output.flatten().data<float>().ToArray();

                RfDetrOnnxExporter.Save(detector, path, size, size);
            }

            Compare(path, pixels, size, queries, classCount, expected);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    /// <summary>
    /// 官方结构（C2f 投影 + 4 维参考点 + ref_point_head + 单档 1296 个 token）。
    /// 权重文件不在本机时跳过。
    /// </summary>
    [Fact]
    public void Exported_Official_RfDetr_Onnx_Matches_Torch_Forward()
    {
        const int classCount = 3;
        const int size = 224;

        string weights = Path.Combine("F:\\ai\\rfdetr", "model.safetensors");

        if (!File.Exists(weights))
        {
            return;
        }

        RfDetrWeightsInfo? info = RfDetrWeightsProbe.Inspect(weights);

        Assert.NotNull(info);

        string path = Path.Combine(Path.GetTempPath(), $"label2train-rfdetr-official-{Guid.NewGuid():N}.onnx");

        try
        {
            float[] pixels;
            float[] expected;

            using (var detector = new RfDetrDetector(
                classCount,
                info.Queries,
                new DinoV2Config(
                    ImageSize: info.ImageSize,
                    PatchSize: info.PatchSize,
                    EmbeddingSize: info.EmbeddingSize,
                    Depth: info.Depth,
                    Heads: info.Heads),
                heads: 8,
                device: torch.CPU,
                encoderLayers: 0,
                decoderLayers: info.DecoderLayers,
                ffnSize: 2048,
                points: info.DeformPoints,
                levels: info.DeformLevels,
                decoderHeads: info.DecoderHeads,
                useRefPointHead: true,
                useC2fProjector: info.HasC2fProjector))
            {
                RfDetrPretrained.Apply(detector, weights, torch.CPU, info);

                using Tensor input = torch.rand(new long[] { 1, 3, size, size });
                pixels = input.flatten().data<float>().ToArray();

                using Tensor output = detector.forward(input);
                expected = output.flatten().data<float>().ToArray();

                RfDetrOnnxExporter.Save(detector, path, size, size);
            }

            Compare(path, pixels, size, info.Queries, classCount, expected);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    /// <summary>
    /// 端到端检测：同一张输入分别走 torch 前向与 OnnxSharp 推理，再把两侧的
    /// <c>boxes</c> / <c>logits</c> 都交给 OnnxSharp 的 RF-DETR 检测接口（<see cref="RfDetrDecoder"/>）
    /// 解码，比较最终检测列表。权重文件不在本机时跳过。
    /// </summary>
    [Fact]
    public void Exported_Official_RfDetr_Detections_Match_Torch_Decoding()
    {
        const int classCount = 3;
        const int size = 224;

        string weights = Path.Combine("F:\\ai\\rfdetr", "model.safetensors");

        if (!File.Exists(weights))
        {
            return;
        }

        RfDetrWeightsInfo? info = RfDetrWeightsProbe.Inspect(weights);

        Assert.NotNull(info);

        string path = Path.Combine(Path.GetTempPath(), $"label2train-rfdetr-detect-{Guid.NewGuid():N}.onnx");

        try
        {
            float[] pixels;
            float[] expectedBoxes;
            float[] expectedLogits;

            using (var detector = new RfDetrDetector(
                classCount,
                info.Queries,
                new DinoV2Config(
                    ImageSize: info.ImageSize,
                    PatchSize: info.PatchSize,
                    EmbeddingSize: info.EmbeddingSize,
                    Depth: info.Depth,
                    Heads: info.Heads),
                heads: 8,
                device: torch.CPU,
                encoderLayers: 0,
                decoderLayers: info.DecoderLayers,
                ffnSize: 2048,
                points: info.DeformPoints,
                levels: info.DeformLevels,
                decoderHeads: info.DecoderHeads,
                useRefPointHead: true,
                useC2fProjector: info.HasC2fProjector))
            {
                RfDetrPretrained.Apply(detector, weights, torch.CPU, info);

                using Tensor input = torch.rand(new long[] { 1, 3, size, size });
                pixels = input.flatten().data<float>().ToArray();

                using Tensor output = detector.forward(input);
                using Tensor boxes = output.narrow(2, 0, 4).contiguous();
                using Tensor logits = output.narrow(2, 4, classCount).contiguous();

                expectedBoxes = boxes.flatten().data<float>().ToArray();
                expectedLogits = logits.flatten().data<float>().ToArray();

                RfDetrOnnxExporter.Save(detector, path, size, size);
            }

            Net net = Net.ReadModel(path);
            net.SetInput(
                RfDetrOnnxExporter.InputName,
                OnnxSharp.Tensors.Tensor.FromArray(
                    pixels, new OnnxSharp.Tensors.TensorShape(1, 3, size, size)));

            List<OnnxSharp.Tensors.Tensor> outputs = net.ForwardAll();
            IReadOnlyList<string> names = net.OutputNames;

            float[] boxesArray = Take(outputs, names, RfDetrOnnxExporter.BoxesOutputName);
            float[] logitsArray = Take(outputs, names, RfDetrOnnxExporter.LogitsOutputName);

            var decoderOptions = new RfDetrOptions
            {
                // 不设阈值：所有 query 都参与解码，随机权重下也一定有东西可比
                Confidence = 0.0001f,
                MaxDetections = 300,
                BackgroundClassId = null,
                Normalization = ChannelNormalization.Divide255,
            };

            IReadOnlyList<Detection> expected = RfDetrDecoder.Decode(
                DecoderInput(expectedBoxes, info.Queries, 4),
                DecoderInput(expectedLogits, info.Queries, classCount),
                size,
                size,
                decoderOptions);

            IReadOnlyList<Detection> actual = RfDetrDecoder.Decode(
                DecoderInput(boxesArray, info.Queries, 4),
                DecoderInput(logitsArray, info.Queries, classCount),
                size,
                size,
                decoderOptions);

            Assert.Equal(expected.Count, actual.Count);

            double worst = 0;

            for (int index = 0; index < expected.Count; index++)
            {
                Detection left = expected[index];
                Detection right = actual[index];

                Assert.Equal(left.ClassId, right.ClassId);

                worst = Math.Max(worst, Math.Abs(left.X - right.X));
                worst = Math.Max(worst, Math.Abs(left.Y - right.Y));
                worst = Math.Max(worst, Math.Abs(left.Right - right.Right));
                worst = Math.Max(worst, Math.Abs(left.Bottom - right.Bottom));
                worst = Math.Max(worst, Math.Abs(left.Score - right.Score) * size);
            }

            // 框坐标按像素比：1e-3 相对量级在 224 输入上约 0.22px
            Assert.True(
                worst < 0.5,
                $"ONNX 与 torch 的检测结果不一致：最大偏差 {worst:F3}（{expected.Count} 个检测）");
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    /// <summary>
    /// 把所有权重写成固定种子的小随机数。
    /// <para>
    /// 不能靠 <c>torch.manual_seed</c>：实测同一个 seed 连续建模型，得到的权重并不一样
    /// （libtorch 的全局 RNG 在进程内没被真正复位），于是随机初始化的 RF-DETR 偶尔初始化出
    /// 一组让前向全 NaN 的权重（120/120 NaN），对拍就变成「跑五次红一次」。
    /// 权重由 .NET 的 <see cref="Random"/> 决定，才是真可复现，也顺带避开极端初始化的数值爆点。
    /// </para>
    /// </summary>
    private static void FillDeterministically(RfDetrDetector module)
    {
        var rng = new Random(20240118);

        foreach ((string _, Parameter parameter) in module.named_parameters())
        {
            var values = new float[parameter.numel()];

            for (int index = 0; index < values.Length; index++)
                values[index] = (float)((rng.NextDouble() - 0.5) * 0.1);

            using Tensor source = torch.tensor(values).reshape(parameter.shape);
            using (torch.no_grad())
            {
                parameter.copy_(source);
            }
        }
    }

    /// <summary>把 <c>[1,Q,C]</c> 的扁平数组包成 OnnxSharp 张量，供解码器使用。</summary>
    private static OnnxSharp.Tensors.Tensor DecoderInput(float[] values, int queries, int channels) =>
        OnnxSharp.Tensors.Tensor.FromArray(
            values, new OnnxSharp.Tensors.TensorShape(1, queries, channels));

    /// <summary>跑 ONNX 并与 torch 的 <c>[1,Q,4+C]</c> 输出逐元素比较。</summary>
    private static void Compare(
        string path, float[] pixels, int size, int queries, int classCount, float[] expected)
    {
        Net net = Net.ReadModel(path);

        net.SetInput(
            RfDetrOnnxExporter.InputName,
            OnnxSharp.Tensors.Tensor.FromArray(
                pixels, new OnnxSharp.Tensors.TensorShape(1, 3, size, size)));

        List<OnnxSharp.Tensors.Tensor> outputs = net.ForwardAll();
        IReadOnlyList<string> names = net.OutputNames;

        float[] boxes = Take(outputs, names, RfDetrOnnxExporter.BoxesOutputName);
        float[] logits = Take(outputs, names, RfDetrOnnxExporter.LogitsOutputName);

        Assert.Equal(queries * 4, boxes.Length);
        Assert.Equal(queries * classCount, logits.Length);

        int stride = 4 + classCount;
        double boxWorst = 0;
        double logitWorst = 0;

        for (int query = 0; query < queries; query++)
        {
            for (int index = 0; index < 4; index++)
                boxWorst = Math.Max(boxWorst, Math.Abs(expected[query * stride + index] - boxes[query * 4 + index]));

            for (int index = 0; index < classCount; index++)
            {
                logitWorst = Math.Max(
                    logitWorst,
                    Math.Abs(expected[query * stride + 4 + index] - logits[query * classCount + index]));
            }
        }

        Assert.True(
            boxWorst < 1e-3 && logitWorst < 1e-3,
            $"ONNX 与 torch 的输出现不一致：box 误差 {boxWorst:E3}，logit 误差 {logitWorst:E3}；"
            + $"torch NaN {expected.Count(float.IsNaN)}/{expected.Length}，"
            + $"ONNX NaN {boxes.Count(float.IsNaN) + logits.Count(float.IsNaN)}/{boxes.Length + logits.Length}");
    }

    private static float[] Take(
        List<OnnxSharp.Tensors.Tensor> outputs, IReadOnlyList<string> names, string name)
    {
        for (int index = 0; index < names.Count && index < outputs.Count; index++)
        {
            if (string.Equals(names[index], name, StringComparison.Ordinal))
            {
                return outputs[index].AsSpan<float>().ToArray();
            }
        }

        throw new InvalidOperationException($"ONNX 输出里没有「{name}」：{string.Join(", ", names)}");
    }
}
