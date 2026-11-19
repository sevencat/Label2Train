using Label2Train.Core.Infrastructure;

namespace Label2Train.Train.Models.RfDetr;

/// <summary>
/// 去哪儿找 RF-DETR 的预训练权重。按顺序：
/// <list type="number">
/// <item>环境变量 <c>LABEL2TRAIN_RFDETR_WEIGHTS</c>（<c>.env</c> 里写也行）；</item>
/// <item><c>data/weights/</c> 下的 <c>*.safetensors</c>（文件名带 rf-detr / rfdetr 的优先）。</item>
/// </list>
/// 都找不到就返回 <c>null</c>，训练按随机初始化跑（老行为）。
/// </summary>
public static class RfDetrWeightSource
{
    /// <summary>环境变量名。</summary>
    public const string Variable = "LABEL2TRAIN_RFDETR_WEIGHTS";

    /// <summary>数据目录下放权重的子目录：<c>data/weights</c>。</summary>
    public const string FolderName = "weights";

    /// <summary>找一份可用的权重；没有返回 <c>null</c>。</summary>
    public static string? Resolve()
    {
        string? configured = AppEnvironment.Get(Variable);

        if (!string.IsNullOrWhiteSpace(configured))
        {
            string path = Path.GetFullPath(configured.Trim().Trim('"'));

            if (File.Exists(path))
            {
                return path;
            }
        }

        string directory = Path.Combine(AppDataPaths.DataDirectory, FolderName);

        if (!Directory.Exists(directory))
        {
            return null;
        }

        return Directory
            .EnumerateFiles(directory, "*.safetensors", SearchOption.TopDirectoryOnly)
            .OrderByDescending(path =>
                Path.GetFileName(path).Contains("rf-detr", StringComparison.OrdinalIgnoreCase)
                || Path.GetFileName(path).Contains("rfdetr", StringComparison.OrdinalIgnoreCase))
            .ThenBy(path => Path.GetFileName(path), StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
    }

    /// <summary>没找到时给用户看的提示（只在训练日志里打一次）。</summary>
    public static string Hint =>
        $"rfdetr: 未配置预训练权重（把 model.safetensors 放到 {Path.Combine(AppDataPaths.DataDirectory, FolderName)}"
        + $"，或设环境变量 {Variable}）；本次从随机初始化开始";
}
