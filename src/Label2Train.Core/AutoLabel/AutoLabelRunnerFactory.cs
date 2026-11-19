namespace Label2Train.Core.AutoLabel;

/// <summary>
/// 按模型文件挑推理实现。训练页的「检测」和标注页的「AI 标注」都从
/// <c>settings.AutoLabelModelPath</c> 拿模型，导出的可能是 YOLO 也可能是 RF-DETR，
/// 两处各判一次早晚会分叉，所以统一走这里。
/// </summary>
public static class AutoLabelRunnerFactory
{
    /// <summary>
    /// 创建推理器：文件名带 <c>detr</c>（rfdetr / simple-detr）的用 DETR 解码，其余按 YOLO。
    /// 命名规则与 <c>TrainingExportService.PrefixFor</c> 一致。
    /// </summary>
    public static IAutoLabelRunner Create(string modelPath) =>
        IsDetrModel(modelPath) ? new RfDetrOnnxRunner(modelPath) : new YoloOnnxRunner(modelPath);

    /// <summary>
    /// 是不是 DETR 系的导出。两者输出格式完全不同：DETR 是 boxes + logits 双输出，
    /// 拿 YOLO 解码器读会直接崩。
    /// </summary>
    public static bool IsDetrModel(string modelPath) =>
        Path.GetFileNameWithoutExtension(modelPath).Contains("detr", StringComparison.OrdinalIgnoreCase);
}
