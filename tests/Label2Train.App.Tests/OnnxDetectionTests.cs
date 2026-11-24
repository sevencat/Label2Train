using System.Linq;
using Label2Train.Core.AutoLabel;
using Label2Train.Train.Export;
using Label2Train.Train.IO.Onnx;
using Label2Train.Train.Models.Weights;
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
    private static string? Find(string name)
    {
        foreach (string directory in PretrainedWeights.Directories)
        {
            string path = Path.Combine(directory, name);

            if (File.Exists(path))
            {
                return path;
            }
        }

        return null;
    }

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
    [Theory]
    [InlineData("yolov8n")]
    [InlineData("yolo11n")]
    [InlineData("yolo26n")]
    public void Exported_Yolo_Onnx_Boxes_Match_Torch_Decoding(string family)
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

            using var model = YoloArchitectures.CreateFromArchitecture(
                torch.CPU, $"{family}-detect-{classCount}");
            model.eval();

            YoloOnnxExporter.Save(model, path, size, size);

            // torch：前向 → 与训练验证同一套解码（xyxy）。
            // YOLO26 的 reg_max = 1，没有 DFL，解码要按 1 走（否则距离被 softmax 抹成 0）
            float[] expected;
            using (var input = torch.tensor(pixels, new long[] { 1, 3, size, size }))
            using (var prediction = model.forward(input))
            using (torch.Tensor reference = YoloPredictionDecoder.DecodeOutput0(
                prediction,
                size,
                size,
                classCount,
                model.Head.RegressionMax))
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

    [Theory]
    [InlineData("yolov8n")]
    [InlineData("yolo26n")]
    public void Exported_Yolo_Onnx_Can_Be_Loaded_By_Detector(string family)
    {
        string path = Path.Combine(Path.GetTempPath(), $"label2train-onnx-{Guid.NewGuid():N}.onnx");

        try
        {
            using (YoloDetectionModel model = YoloArchitectures.CreateFromArchitecture(torch.CPU, $"{family}-detect-2"))
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

    /// <summary>
    /// YOLO26 的 one-to-one 分支导出成无 NMS 图：输出 <c>[1, 300, 6]</c>（x1、y1、x2、y2、score、class）。
    /// 这里把 ONNX 的 top-k 后处理与 torch 的 <see cref="YoloPredictionDecoder.DecodeEndToEnd"/> 逐元素对齐。
    /// <para>
    /// 必须用官方权重：随机初始化的网络特征图在空间上几乎相同，几乎所有锚点的分类分数完全相等，
    /// top-k 会在同分锚点里各按各的平手规则挑（两边选中的框集合不同），没法逐行对拍。
    /// </para>
    /// </summary>
    [Fact]
    [Trait("Category", "Slow")]
    public void Exported_Yolo26_NmsFree_Onnx_Matches_Torch_Decoding()
    {
        const int size = 320;
        const int classCount = 80;
        string path = Path.Combine(Path.GetTempPath(), $"label2train-nmsfree-{Guid.NewGuid():N}.onnx");

        string? weights = Find("yolo26n.safetensors");
        if (weights is null)
        {
            Assert.Skip("缺 data/weights/yolo26n.safetensors；先跑 tools/convert_yolo_weights.py");
        }

        try
        {
            float[] pixels = new float[3 * size * size];
            var random = new Random(23);
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = (float)random.NextDouble();
            }

            using var model = YoloArchitectures.Create(torch.CPU, YoloModelFamily.V26, YoloScale.N, classCount);
            model.eval();
            YoloPretrained.Apply(model, weights, torch.CPU);

            YoloOnnxExporter.Save(model, path, size, size, endToEnd: true);

            float[] expected;
            float[] allScores;
            float[] allBoxes;
            float[] allClasses;
            long anchors;

            using (var input = torch.tensor(pixels, new long[] { 1, 3, size, size }))
            using (var noGrad = torch.no_grad())
            {
                // 导出的无 NMS 图走 one-to-one 分支，参考值也必须取这条分支的原始输出
                IReadOnlyList<torch.Tensor> trace = model.ForwardTrace(input);
                IReadOnlyList<int> sources = model.InputSources[^1];
                var features = new torch.Tensor[sources.Count];
                for (int index = 0; index < sources.Count; index++)
                {
                    features[index] = trace[sources[index]];
                }

                using torch.Tensor raw = model.Head.ForwardOneToOne(features, detachFeatures: false);
                using (torch.Tensor reference = YoloPredictionDecoder.DecodeEndToEnd(
                    raw,
                    size,
                    size,
                    classCount,
                    model.Head.RegressionMax))
                {
                    expected = reference.data<float>().ToArray();
                }

                // 全部锚点的解码结果：用来验证 ONNX 的每一行确实是「某个真实锚点」的框与分数
                using torch.Tensor decoded = YoloPredictionDecoder.DecodeOutput0(
                    raw,
                    size,
                    size,
                    classCount,
                    model.Head.RegressionMax);
                float[] flat = decoded.data<float>().ToArray();
                anchors = decoded.shape[2];

                allScores = new float[anchors];
                allClasses = new float[anchors];
                allBoxes = new float[4 * anchors];
                for (long anchor = 0; anchor < anchors; anchor++)
                {
                    float best = float.NegativeInfinity;
                    float bestClass = 0;
                    for (int cls = 0; cls < classCount; cls++)
                    {
                        float value = flat[(4 + cls) * anchors + anchor];
                        if (value > best)
                        {
                            best = value;
                            bestClass = cls;
                        }
                    }

                    allScores[anchor] = best;
                    allClasses[anchor] = bestClass;
                    for (int side = 0; side < 4; side++)
                    {
                        allBoxes[side * anchors + anchor] = flat[side * anchors + anchor];
                    }
                }
            }

            OnnxSharp.Core.Net net = OnnxSharp.Core.Net.ReadModel(path);
            OnnxSharp.Tensors.Tensor input2 = OnnxSharp.Tensors.Tensor.FromArray(
                pixels,
                new OnnxSharp.Tensors.TensorShape(1, 3, size, size));
            net.SetInput(YoloOnnxExporter.DefaultInputName, input2);
            float[] actual = net.Forward().AsSpan<float>().ToArray();

            int keep = YoloOnnxExporter.DefaultEndToEndDetections;
            Assert.Equal(keep * 6, actual.Length);
            Assert.Equal(expected.Length, actual.Length);

            // 分数几乎相等的锚点谁进 top-k 由各自的平手规则决定（torch 的 CPU topk 不保证按索引，
            // 这批噪声输入下分数都落在 1e-6 量级，成片的平手），所以不能按行直接比。
            // 换个等价且更强的判据：ONNX 输出的每一行都必须是「某个真实锚点」的分数/框/类别，
            // 且 300 个分数与 torch 的 top-300 分数完全一致。
            static (float Score, float X1, float Y1, float X2, float Y2, float Class)[] Rows(float[] values)
            {
                var rows = new (float, float, float, float, float, float)[values.Length / 6];
                for (int row = 0; row < rows.Length; row++)
                {
                    int offset = row * 6;
                    rows[row] = (values[offset + 4], values[offset], values[offset + 1], values[offset + 2], values[offset + 3], values[offset + 5]);
                }

                return rows;
            }

            var expectedRows = Rows(expected);
            var actualRows = Rows(actual);

            // 1) 选中的 300 个分数必须是同一批：排序后逐元素相同
            float[] expectedScores = expectedRows.Select(row => row.Score).OrderByDescending(score => score).ToArray();
            float[] actualScores = actualRows.Select(row => row.Score).OrderByDescending(score => score).ToArray();
            double maxScoreError = expectedScores.Zip(actualScores, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(maxScoreError < 1e-6, $"无 NMS 导出的 top-k 分数与 torch 不一致，最大偏差 {maxScoreError:E3}");

            // 2) 每一行的框、分数、类别都必须能在全部锚点里找到对应项
            for (int row = 0; row < actualRows.Length; row++)
            {
                var value = actualRows[row];
                bool matched = false;

                for (long anchor = 0; anchor < anchors && !matched; anchor++)
                {
                    matched = Math.Abs(allScores[anchor] - value.Score) < 1e-6
                        && Math.Abs(allClasses[anchor] - value.Class) < 0.5
                        && Math.Abs(allBoxes[anchor] - value.X1) < 0.5
                        && Math.Abs(allBoxes[anchors + anchor] - value.Y1) < 0.5
                        && Math.Abs(allBoxes[2 * anchors + anchor] - value.X2) < 0.5
                        && Math.Abs(allBoxes[3 * anchors + anchor] - value.Y2) < 0.5;
                }

                Assert.True(matched, $"ONNX 第 {row} 行 {value} 在 torch 的全部锚点里找不到对应项");
            }
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    /// <summary>无 NMS 导出必须能被检测端识别成 EndToEnd 布局（否则 6 字段会被当成通道优先解码）。</summary>
    [Fact]
    public void Exported_Yolo26_NmsFree_Onnx_Can_Be_Loaded_By_Detector()
    {
        string path = Path.Combine(Path.GetTempPath(), $"label2train-nmsfree-load-{Guid.NewGuid():N}.onnx");

        try
        {
            using (YoloDetectionModel model = YoloArchitectures.CreateFromArchitecture(torch.CPU, "yolo26n-detect-2"))
            {
                YoloOnnxExporter.Save(model, path, 640, 640, endToEnd: true);
            }

            OnnxModelSummary summary = OnnxModelReader.Read(path);

            Assert.Equal(
                [1L, YoloOnnxExporter.DefaultEndToEndDetections, 6L],
                summary.Outputs[0].Dimensions.Select(dimension => dimension.Value).ToArray());

            using var runner = new YoloOnnxRunner(path);

            byte[] bgra = new byte[640 * 640 * 4];
            IReadOnlyList<AutoLabelCandidate> candidates =
                runner.Predict(bgra, 640, 640, new AutoLabelOptions(path, 0.25, ClassNames: null));

            Assert.All(candidates, candidate =>
            {
                Assert.InRange(candidate.Box.X, 0, 640);
                Assert.InRange(candidate.Box.Y, 0, 640);
                Assert.InRange(candidate.Box.Right, 0, 640);
                Assert.InRange(candidate.Box.Bottom, 0, 640);
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

}
