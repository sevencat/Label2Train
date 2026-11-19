using System.Linq;
using Label2Train.Core.AutoLabel;
using Label2Train.Train.Export;
using Label2Train.Train.IO.Onnx;
using Label2Train.Train.Models.RfDetr;
using Label2Train.Train.Models.Yolo;
using TorchSharp;
using Xunit;
using TrainRfDetrDetector = Label2Train.Train.Models.RfDetr.RfDetrDetector;

namespace Label2Train.App.Tests;

/// <summary>
/// 「检测」按钮靠 OnnxSharp 加载导出模型：输入里带符号维（dim_param，如 <c>batch</c>）时
/// OnnxSharp 会把 shape 读成空列表并拒绝加载（must be rank-4 NCHW, got []），
/// 于是点检测看起来毫无反应。这里把「导出后必须能被检测端加载」固定成回归用例。
/// </summary>
public class OnnxDetectionTests
{
    /// <summary>
    /// OnnxSharp 的 Preprocess 只吃紧凑的 3 字节像素；传 BGRA 会整幅图错位，
    /// 表现为「模型明明 mAP50=1，检测框却完全乱」。这里锁住打包的正确性。
    /// </summary>
    [Fact]
    public void Bgra_Is_Packed_To_Compact_Bgr()
    {
        // 2×2，按行：红、绿 / 蓝、白
        byte[] bgra =
        [
            0, 0, 255, 255,   0, 255, 0, 255,
            255, 0, 0, 255,   255, 255, 255, 255,
        ];

        byte[] bgr = YoloOnnxRunner.ToPackedBgr(bgra, 2, 2);

        Assert.Equal(12, bgr.Length);
        Assert.Equal([0, 0, 255], bgr[..3]);        // 红
        Assert.Equal([0, 255, 0], bgr[3..6]);       // 绿
        Assert.Equal([255, 0, 0], bgr[6..9]);       // 蓝
        Assert.Equal([255, 255, 255], bgr[9..12]);  // 白
    }

    /// <summary>
    /// RF-DETR 走的是另一条推理路径（boxes + logits 双输出、letterbox 反变换）。
    /// 这里锁住「导出的 rfdetr.onnx 能被 <see cref="RfDetrOnnxRunner"/> 加载并跑完整条链路」。
    /// </summary>
    [Fact]
    public void Exported_RfDetr_Onnx_Can_Be_Loaded_By_Runner()
    {
        string path = Path.Combine(Path.GetTempPath(), $"label2train-rfdetr-{Guid.NewGuid():N}.onnx");

        try
        {
            using (TrainRfDetrDetector detector = new(
                classCount: 2,
                queries: 20,
                new DinoV2Config(ImageSize: 518, EmbeddingSize: 64, Depth: 2, Heads: 4),
                heads: 8,
                device: torch.CPU))
            {
                // DINOv2 的 patch 是 14，输入边长必须同时是 14 与 16 的倍数
                RfDetrOnnxExporter.Save(detector, path, 336, 336);
            }

            OnnxModelSummary summary = OnnxModelReader.Read(path);

            Assert.Equal([RfDetrOnnxExporter.InputName], summary.Inputs.Select(item => item.Name));
            Assert.Equal(
                [RfDetrOnnxExporter.BoxesOutputName, RfDetrOnnxExporter.LogitsOutputName],
                summary.Outputs.Select(item => item.Name));

            using var runner = new RfDetrOnnxRunner(path);

            // 非方图才能验证 letterbox 的反变换（减填边、除比例）没有把框算飞
            byte[] bgra = new byte[480 * 640 * 4];
            IReadOnlyList<AutoLabelCandidate> candidates =
                runner.Predict(bgra, 640, 480, new AutoLabelOptions(path, 0.01, ClassNames: null));

            // 未训练的权重不保证出框；这里只要求链路跑通
            Assert.All(candidates, candidate =>
            {
                Assert.InRange(candidate.Box.X, 0, 640);
                Assert.InRange(candidate.Box.Y, 0, 480);
                Assert.InRange(candidate.Box.Right, 0, 640);
                Assert.InRange(candidate.Box.Bottom, 0, 480);
            });
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

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
