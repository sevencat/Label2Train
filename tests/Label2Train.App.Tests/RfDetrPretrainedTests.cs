using Label2Train.Train.Models.RfDetr;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 官方 RF-DETR 权重（<c>backbone.0.*</c> 键名）能不能装进本项目模型。
/// 只有本机存在权重文件时才真的跑；没有就跳过（不把 134MB 塞进仓库）。
/// </summary>
public sealed class RfDetrPretrainedTests
{
    /// <summary>本机官方权重：环境变量优先，其次 F:\ai\rfdetr\model.safetensors。</summary>
    private static string? WeightsPath
    {
        get
        {
            string? configured = Environment.GetEnvironmentVariable(RfDetrWeightSource.Variable);

            if (!string.IsNullOrWhiteSpace(configured) && File.Exists(configured))
            {
                return configured;
            }

            const string fallback = @"F:\ai\rfdetr\model.safetensors";

            return File.Exists(fallback) ? fallback : null;
        }
    }

    [Fact]
    public void Official_Checkpoint_Reports_Its_Architecture()
    {
        string? path = WeightsPath;

        if (path is null)
        {
            return;
        }

        Assert.True(RfDetrWeightsProbe.IsOfficialRfDetr(path));

        RfDetrWeightsInfo? info = RfDetrWeightsProbe.Inspect(path);

        Assert.NotNull(info);
        Assert.Equal(16, info.PatchSize);
        Assert.Equal(384, info.EmbeddingSize);
        Assert.Equal(12, info.Depth);
        Assert.Equal(576, info.ImageSize);
        Assert.Equal(4, info.DecoderLayers);
        Assert.Equal(16, info.DecoderHeads);
        Assert.Equal(1, info.DeformLevels);
        Assert.Equal(13, info.GroupDetr);
        Assert.Equal(300, info.Queries);
        Assert.True(info.HasC2fProjector);
    }

    [Fact]
    public void Official_Weights_Load_Into_Detector_And_Forward_Runs()
    {
        string? path = WeightsPath;

        if (path is null)
        {
            return;
        }

        RfDetrWeightsInfo info = RfDetrWeightsProbe.Inspect(path)!;
        Device device = torch.CPU;

        // 与 DetrTrainingSession.BuildModel 在「有官方权重」时的装配保持一致
        var dinoConfig = new DinoV2Config(
            ImageSize: info.ImageSize,
            PatchSize: info.PatchSize,
            EmbeddingSize: info.EmbeddingSize,
            Depth: info.Depth,
            Heads: info.Heads);

        const int classCount = 2;

        using var detector = new RfDetrDetector(
            classCount,
            info.Queries,
            dinoConfig,
            8,
            device,
            encoderLayers: 0,
            decoderLayers: info.DecoderLayers,
            ffnSize: 2048,
            points: info.DeformPoints,
            levels: info.DeformLevels,
            decoderHeads: info.DecoderHeads,
            useRefPointHead: true,
            useC2fProjector: true);

        RfDetrPretrainedReport report = RfDetrPretrained.Apply(detector, path, device, info);

        // 整网 509 个张量里，本项目没有的（分组辅助头、mask token 等）会跳过，
        // 剩下能对上的主干 + 投影器 + 4 层解码器 + query/refpoint 大约 298 个
        Assert.True(report.Loaded >= 290, $"只载入了 {report.Loaded} 个张量：{report.Summary}");

        // 主干、投影器、解码器都该覆盖到；分类头类别数不同（COCO 91 vs 2），本来就该保持随机
        Assert.DoesNotContain("backbone.patch_embed.proj.weight", report.Untouched);
        Assert.DoesNotContain("backbone.blocks.0.attn.qkv.weight", report.Untouched);
        Assert.DoesNotContain("projector.stages.0.0.cv1.conv.weight", report.Untouched);
        Assert.DoesNotContain("decoder_layers.3.self_attn.in_proj_weight", report.Untouched);
        Assert.DoesNotContain("decoder_layers.0.deform_attn.sampling_offsets", report.Untouched);
        Assert.Contains("class_embed.weight", report.Untouched);

        detector.eval();

        using Tensor input = torch.zeros(1, 3, 224, 224, device: device);
        using Tensor output = detector.forward(input);

        Assert.True(output.isfinite().all().item<bool>(), "前向输出里有 NaN / Inf");
    }
}
