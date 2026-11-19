using Label2Train.Train.Models.RfDetr;
using Label2Train.Train.Sessions;
using Label2Train.Train.Training;
using Label2Train.Train.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>临时冒烟：用官方权重重跑一轮真实训练，验证会话链路（跑完删掉）。</summary>
public sealed class TempDetrPretrainedSmoke
{
    [Fact]
    public void Run_One_Epoch_With_Pretrained_Weights()
    {
        string dataYaml = @"F:\code\Label2Train\src\Label2Train.App\bin\Debug\net10.0\data\projects\3\models\rfdetr\dataset\data.yaml";

        if (!File.Exists(dataYaml))
        {
            return;
        }

        Label2Train.Core.Imaging.ImageRuntime.Use(new ImageFacade.SkiaSharp.SkiaBackend());

        string output_ = Path.Combine(Path.GetTempPath(), "detr-smoke-" + Guid.NewGuid().ToString("N"));

        var options = new DetrTrainingOptions
        {
            DataYaml = dataYaml,
            Epochs = 1,
            BatchSize = 1,
            ImageSize = 224,
            LearningRate = 2e-4f,
            OutputDirectory = output_,
            Device = DeviceFactory.Parse("cpu"),
            UseRfDetr = true,
            DinoWeights = @"F:\ai\rfdetr\model.safetensors",
        };

        using var session = new DetrTrainingSession();
        IReadOnlyList<EpochMetrics> history = session.StartAsync(
            options,
            log: message => File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "detr-smoke-log.txt"),
                message + Environment.NewLine)).GetAwaiter().GetResult();

        Assert.NotEmpty(history);

        // 导出会按元数据重建模型：dino_patch / dino_image / c2f 这些没存对，ONNX 就是半个随机网
        var exporter = new Label2Train.Train.Export.TrainingExportService();
        Label2Train.Train.Export.TrainingExportResult result = exporter.Export(
            output_, Path.Combine(output_, "export"), exportOnnx: true);

        Assert.True(File.Exists(result.OnnxPath), "没导出 ONNX");
        Assert.True(new FileInfo(result.OnnxPath!).Length > 10_000_000, "ONNX 太小，可能是空图");
    }
}
