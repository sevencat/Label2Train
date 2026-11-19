using Label2Train.Train.Export;
using Label2Train.Train.Models.RfDetr;
using OnnxSharp.Core;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Label2Train.App.Tests;

/// <summary>临时诊断：官方结构逐项打开（C2f 投影 / 4 维参考点 + ref_point_head / 单档 / points=2），
/// 找出 ONNX 与 torch 数值分歧来自哪一块。用完即删。</summary>
public sealed class TempRfDetrExportParity
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Diagnose(bool c2f, bool refHead)
    {
        Run(c2f, refHead, zeroRefHead: false);
    }

    /// <summary>ref_point_head 权重清零（输出恒 0、layerInput 就等于 queries），用来区分
    /// 「正弦编码/MLP 数值算错」还是「其它地方」。若清零后仍有误差，说明错误不在这一支。</summary>
    [Fact]
    public void RefHead_Zeroed()
    {
        Run(c2f: false, refHead: true, zeroRefHead: true);
    }

    private static void Run(bool c2f, bool refHead, bool zeroRefHead)
    {
        const int classCount = 2;
        const int queries = 30;
        const int size = 224;

        string path = Path.Combine(Path.GetTempPath(), $"diag-{Guid.NewGuid():N}.onnx");

        try
        {
            float[] pixels;
            float[] expected;

            using (var detector = new RfDetrDetector(
                classCount,
                queries,
                new DinoV2Config(ImageSize: 224, PatchSize: 16, EmbeddingSize: 64, Depth: 2, Heads: 4),
                heads: 8,
                device: torch.CPU,
                encoderLayers: 0,
                decoderLayers: 2,
                ffnSize: 128,
                points: 2,
                levels: 1,
                decoderHeads: 8,
                useRefPointHead: refHead,
                useC2fProjector: c2f))
            {
                using Tensor input = torch.rand(new long[] { 1, 3, size, size });
                pixels = input.flatten().data<float>().ToArray();

                using Tensor output = detector.forward(input);
                expected = output.flatten().data<float>().ToArray();

                RfDetrOnnxExporter.Save(detector, path, size, size);
            }

            Net net = Net.ReadModel(path);
            net.SetInput(
                RfDetrOnnxExporter.InputName,
                OnnxSharp.Tensors.Tensor.FromArray(
                    pixels, new OnnxSharp.Tensors.TensorShape(1, 3, size, size)));

            List<OnnxSharp.Tensors.Tensor> outputs = net.ForwardAll();
            IReadOnlyList<string> names = net.OutputNames;

            float[] boxes = Take(outputs, names, RfDetrOnnxExporter.BoxesOutputName);
            float[] logits = Take(outputs, names, RfDetrOnnxExporter.LogitsOutputName);

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
                $"c2f={c2f} refHead={refHead} → box 误差 {boxWorst:E3}，logit 误差 {logitWorst:E3}");
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
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
