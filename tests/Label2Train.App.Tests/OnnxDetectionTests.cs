using Label2Train.Core.AutoLabel;
using Label2Train.Train.Export;
using Label2Train.Train.IO.Onnx;
using Label2Train.Train.Models.Yolo;
using TorchSharp;
using Xunit;

namespace Label2Train.App.Tests;

/// <summary>
/// 「检测」按钮靠 OnnxSharp 加载导出模型：输入里带符号维（dim_param，如 <c>batch</c>）时
/// OnnxSharp 会把 shape 读成空列表并拒绝加载（must be rank-4 NCHW, got []），
/// 于是点检测看起来毫无反应。这里把「导出后必须能被检测端加载」固定成回归用例。
/// </summary>
public class OnnxDetectionTests
{
    [Fact]
    public void Exported_Yolo_Onnx_Can_Be_Loaded_By_Detector()
    {
        string path = Path.Combine(Path.GetTempPath(), $"label2train-onnx-{Guid.NewGuid():N}.onnx");

        try
        {
            using (YoloDetectionModel model = YoloArchitectures.CreateFromArchitecture(torch.CPU, "yolov8n-detect-2"))
            {
                YoloOnnxExporter.Save(model, path, 640, 640);
            }

            OnnxModelSummary summary = OnnxModelReader.Read(path);

            Assert.All(summary.Inputs[0].Dimensions, dimension => Assert.NotNull(dimension.Value));
            Assert.All(summary.Outputs[0].Dimensions, dimension => Assert.NotNull(dimension.Value));

            using var runner = new YoloOnnxRunner(path);

            byte[] bgra = new byte[640 * 640 * 4];
            IReadOnlyList<AutoLabelCandidate> candidates =
                runner.Predict(bgra, 640, 640, new AutoLabelOptions(path, 0.25, ClassNames: null));

            Assert.NotNull(candidates);
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
