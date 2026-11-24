using Label2Train.Train.Core;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Models.Yolo;

/// <summary>
/// YOLOv26 检测模型（detect 任务）：层级与 ultralytics <c>yolo26*.pt</c> 的
/// <c>state_dict</c> 一一对应，<c>model.23</c> 是带 one-to-one 分支的检测头。
/// <para>
/// 与 v11 的差异只有两处：<c>model.22</c> 的 C3k2 内层块多一个 PSA 注意力，
/// 检测头去掉 DFL（<c>reg_max = 1</c>）并多出 <c>one2one_cv2</c> / <c>one2one_cv3</c>。
/// </para>
/// <para>
/// <see cref="forward"/> 仍然只返回 one-to-many 的原始预测（评估 / 解码 / 导出都按它走），
/// one-to-one 分支的结果按 ultralytics 的做法通过 <see cref="GetLossExtras"/> 交给损失。
/// </para>
/// </summary>
public sealed class YoloV26DetectionModel : YoloDetectionModel, ILossExtrasProvider
{
    /// <summary>
    /// extras 里 one-to-one 分支原始输出的键名。与 <c>YoloEndToEndDetectionLoss.OneToOneKey</c> 一致
    /// （<c>Models</c> 不能引用 <c>Training</c>，故此处用字面量，由测试断言两侧一致）。
    /// </summary>
    public const string OneToOneKey = "yolo.one2one";

    private Tensor? _oneToOne;

    internal YoloV26DetectionModel(
        string name,
        YoloModuleChain model,
        DetectHeadModule head,
        YoloScale scale,
        int classCount)
        : base(name, YoloModelFamily.V26, scale, classCount, model, head)
    {
    }

    /// <inheritdoc />
    public override Tensor forward(Tensor input)
    {
        IReadOnlyList<Tensor> trace = ForwardTrace(input);
        Tensor prediction = trace[^1];
        Tensor[] features = ResolveHeadInputs(trace);

        // 与 ultralytics 一致：one-to-one 分支吃 detach 过的特征（只在训练时），
        // 梯度只从 one-to-many 那条路走。评估时算它是为了让验证损失和训练损失可比。
        _oneToOne = Head.ForwardOneToOne(features, detachFeatures: training);

        return prediction;
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, Tensor>? GetLossExtras() =>
        _oneToOne is null
            ? null
            : new Dictionary<string, Tensor>(1, StringComparer.Ordinal) { [OneToOneKey] = _oneToOne };
}
