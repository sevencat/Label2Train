using Label2Train.Train.Export;
using Label2Train.Train.IO.Onnx;
using Label2Train.Train.Models.RfDetr;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Label2Train.App.Tests;

/// <summary>
/// RF-DETR 的默认输入边长是 560（40×14），比 DINOv2 预训练的 518（37×14，1370 个 token）大。
/// 位置编码按官方做法做双三次插值时本来就没有上限；主干与 ONNX 导出曾各留一道
/// 「超过位置编码上限」的硬校验，导致默认设置一训练就抛「输入 560x560 需要 1601 个 token」。
/// </summary>
public class RfDetrImageSizeTests
{
    /// <summary>560 = 40×14：刚好越过预训练的 37×14 网格，也是 RF-DETR 的默认边长。</summary>
    private const int Size = 560;

    [Fact]
    public void Backbone_Forward_Works_Above_The_Pretrained_Grid()
    {
        using var backbone = new DinoV2Backbone(
            new DinoV2Config(ImageSize: 518, EmbeddingSize: 32, Depth: 1, Heads: 2),
            torch.CPU);

        using Tensor input = torch.rand(new long[] { 1, 3, Size, Size });
        using Tensor tokens = backbone.forward(input);

        Assert.Equal(new long[] { 1, 1601, 32 }, tokens.shape);
    }

    [Fact]
    public void Onnx_Export_Works_Above_The_Pretrained_Grid()
    {
        using var backbone = new DinoV2Backbone(
            new DinoV2Config(ImageSize: 518, EmbeddingSize: 32, Depth: 1, Heads: 2),
            torch.CPU);

        string path = Path.Combine(Path.GetTempPath(), $"label2train-dinov2-560-{Guid.NewGuid():N}.onnx");

        try
        {
            DinoV2OnnxExporter.Save(backbone, path, Size, Size);

            OnnxModelSummary summary = OnnxModelReader.Read(path);

            Assert.Equal(new long[] { 1, 1601, 32 }, summary.Outputs[0].Dimensions.Select(d => d.Value ?? 0).ToArray());
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }
}
