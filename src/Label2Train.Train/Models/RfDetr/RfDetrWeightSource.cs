using Label2Train.Core.Infrastructure;
using Label2Train.Train.Models.Weights;

namespace Label2Train.Train.Models.RfDetr;

/// <summary>
/// 去哪儿找 RF-DETR 的预训练权重。按顺序：
/// <list type="number">
/// <item>环境变量 <c>LABEL2TRAIN_RFDETR_WEIGHTS</c>（<c>.env</c> 里写也行）；</item>
/// <item>候选目录下的 <c>*.safetensors</c>（文件名带 rf-detr / rfdetr 的优先），
/// 目录顺序见 <see cref="PretrainedWeights.Directories"/>：exe 同级 <c>data/weights</c>、
/// 仓库根 <c>data/weights</c>、当前工作目录 <c>data/weights</c>。</item>
/// </list>
/// 都找不到就返回 <c>null</c>，训练按随机初始化跑（老行为）。
/// </summary>
public static class RfDetrWeightSource
{
    /// <summary>环境变量名。</summary>
    public const string Variable = "LABEL2TRAIN_RFDETR_WEIGHTS";

    /// <summary>数据目录下放权重的子目录：<c>data/weights</c>。</summary>
    public const string FolderName = PretrainedWeights.FolderName;

    private static readonly string[] PreferredFragments = ["rf-detr", "rfdetr"];

    /// <summary>找一份可用的权重；没有返回 <c>null</c>。</summary>
    public static string? Resolve() => PretrainedWeights.Resolve(Variable, PreferredFragments);

    /// <summary>没找到时给用户看的提示（只在训练日志里打一次）。</summary>
    public static string Hint => PretrainedWeights.Hint(Variable, PreferredFragments);
}
