using System.Linq;
using Label2Train.Core.AutoLabel;
using Label2Train.Train.Export;
using Label2Train.Train.IO.Onnx;
using Label2Train.Train.Models.RfDetr;
using Label2Train.Train.Models.Yolo;
using Label2Train.Train.Training.Yolo;
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

    /// <summary>
    /// 导出图的 cx/cy 必须带 dist2bbox 的中心偏移 <c>(rb - lt) / 2</c>。只写网格中心的话，
    /// 每个框都会被吸到最近的网格中心（最多半个 stride），原图上就是「所有框整体错位」——
    /// stride 32 那层能差 ±16px，放大到 1080 更明显。这里把 ONNX 的输出与 torch 侧
    /// 同一套解码（<see cref="YoloPredictionDecoder.DecodeOutput0"/>）逐锚点对齐。
    /// </summary>
    [Fact]
    public void Exported_Yolo_Onnx_Boxes_Match_Torch_Decoding()
    {
        const int size = 320;
        const int classCount = 2;
        string path = Path.Combine(Path.GetTempPath(), $"label2train-align-{Guid.NewGuid():N}.onnx");

        try
        {
            float[] pixels = new float[3 * size * size];
            var random = new Random(11);
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = (float)random.NextDouble();
            }

            using var model = YoloArchitectures.CreateFromArchitecture(torch.CPU, $"yolov8n-detect-{classCount}");
            model.eval();

            YoloOnnxExporter.Save(model, path, size, size);

            // torch：前向 → 与训练验证同一套解码（xyxy）
            float[] expected;
            using (var input = torch.tensor(pixels, new long[] { 1, 3, size, size }))
            using (var prediction = model.forward(input))
            using (torch.Tensor reference = YoloPredictionDecoder.DecodeOutput0(
                prediction,
                size,
                size,
                classCount))
            {
                expected = reference.data<float>().ToArray();
            }

            OnnxSharp.Core.Net net = OnnxSharp.Core.Net.ReadModel(path);
            OnnxSharp.Tensors.Tensor input2 = OnnxSharp.Tensors.Tensor.FromArray(
                pixels,
                new OnnxSharp.Tensors.TensorShape(1, 3, size, size));
            net.SetInput(YoloOnnxExporter.DefaultInputName, input2);
            float[] actual = net.Forward().AsSpan<float>().ToArray();

            int anchors = (size / 8 * (size / 8)) + (size / 16 * (size / 16)) + (size / 32 * (size / 32));
            Assert.Equal((4 + classCount) * anchors, actual.Length);

            double maxError = 0;

            for (int index = 0; index < anchors; index++)
            {
                double x1 = expected[index];
                double y1 = expected[anchors + index];
                double x2 = expected[2 * anchors + index];
                double y2 = expected[3 * anchors + index];

                double cx = actual[index];
                double cy = actual[anchors + index];
                double width = actual[2 * anchors + index];
                double height = actual[3 * anchors + index];

                maxError = Math.Max(maxError, Math.Abs(cx - (width / 2) - x1));
                maxError = Math.Max(maxError, Math.Abs(cy - (height / 2) - y1));
                maxError = Math.Max(maxError, Math.Abs(cx + (width / 2) - x2));
                maxError = Math.Max(maxError, Math.Abs(cy + (height / 2) - y2));
            }

            Assert.True(maxError < 0.5, $"ONNX 与 torch 的框不一致，最大偏差 {maxError:F2}px（中心偏移丢了？）");
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
