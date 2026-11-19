using Label2Train.Train.Export;
using Label2Train.Train.Models.RfDetr;
using OnnxSharp.Core;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

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
        double worst = 0;

        for (int query = 0; query < queries; query++)
        {
            for (int index = 0; index < 4; index++)
                worst = Math.Max(worst, Math.Abs(expected[query * stride + index] - boxes[query * 4 + index]));

            for (int index = 0; index < classCount; index++)
            {
                worst = Math.Max(
                    worst,
                    Math.Abs(expected[query * stride + 4 + index] - logits[query * classCount + index]));
            }
        }

        // bicubic 插值（pos_embed）与批统计量 BN 都有浮点差异，1e-3 足够严到能发现结构性错误
        Assert.True(worst < 1e-3, $"ONNX 与 torch 的输出现不一致：最大误差 {worst:E3}");
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
