using Label2Train.Core.Infrastructure;

namespace Label2Train.Train.Models.Weights;

/// <summary>
/// 预训练权重的文件定位（YOLO 与 RF-DETR 共用）。
/// <para>
/// 查找顺序：环境变量（完整文件路径）→ 候选目录里的 <c>*.safetensors</c>。
/// 候选目录按优先级：<c>exe 同级/data/weights</c>（<see cref="AppDataPaths.DataDirectory"/>）、
/// 仓库根的 <c>data/weights</c>（从 exe 往上找 .sln 所在目录）、当前工作目录的 <c>data/weights</c>。
/// 加仓库根是为了避免「把权重放在 <c>data/weights</c> 却因为程序读 bin 下的 data 而不生效」这个坑。
/// </para>
/// <para>
/// 文件名排序：先按 <c>preferredFragments</c>（例如 <c>yolov8</c>、<c>rf-detr</c>）打命中标记，
/// 再按文件名。这样同一个目录放多份权重时能挑到对的那份。
/// </para>
/// </summary>
public static class PretrainedWeights
{
    /// <summary>数据目录下放权重的子目录名。</summary>
    public const string FolderName = "weights";

    /// <summary>权重文件扩展名。</summary>
    public const string Extension = "*.safetensors";

    /// <summary>候选目录（按优先级，已去重、只保留存在的目录）。</summary>
    public static IReadOnlyList<string> Directories => ResolveDirectories();

    /// <summary>找一份可用的权重；没有返回 <c>null</c>。</summary>
    /// <param name="variable">环境变量名（<c>.env</c> 里写也行）。</param>
    /// <param name="preferredFragments">文件名里命中这些片段的优先（不区分大小写）。</param>
    public static string? Resolve(string variable, params string[] preferredFragments) =>
        Resolve(variable, ResolveDirectories(), preferredFragments);

    /// <summary>没找到权重时给日志看的提示。</summary>
    /// <param name="variable">环境变量名。</param>
    /// <param name="preferredFragments">文件名里命中这些片段的优先（不区分大小写）。</param>
    public static string Hint(string variable, params string[] preferredFragments)
    {
        string names = preferredFragments.Length == 0
            ? Extension
            : $"含「{string.Join("」或「", preferredFragments)}」的 {Extension}";

        string directories = string.Join("、", ResolveDirectories());

        return directories.Length == 0
            ? $"未配置预训练权重（把 {names} 放到 {Path.Combine(AppDataPaths.DataDirectory, FolderName)}，"
                + $"或设环境变量 {variable}）；本次从随机初始化开始"
            : $"未配置预训练权重（把 {names} 放到以下任一目录：{directories}，"
                + $"或设环境变量 {variable}）；本次从随机初始化开始";
    }

    /// <summary>测试用：在指定目录列表里找（不读环境变量）。</summary>
    internal static string? ResolveIn(IEnumerable<string> directories, params string[] preferredFragments)
    {
        string[] fragments = preferredFragments
            .Where(fragment => !string.IsNullOrWhiteSpace(fragment))
            .ToArray();

        return Enumerate(directories)
            .OrderByDescending(path => fragments.Any(fragment =>
                Path.GetFileName(path).Contains(fragment, StringComparison.OrdinalIgnoreCase)))
            .ThenBy(path => Path.GetFileName(path), StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
    }

    private static string? Resolve(
        string variable,
        IReadOnlyList<string> directories,
        string[] preferredFragments)
    {
        string? configured = AppEnvironment.Get(variable);

        if (!string.IsNullOrWhiteSpace(configured))
        {
            string path = Path.GetFullPath(configured.Trim().Trim('"'));

            if (File.Exists(path))
            {
                return path;
            }
        }

        return ResolveIn(directories, preferredFragments);
    }

    private static IEnumerable<string> Enumerate(IEnumerable<string> directories) =>
        directories
            .Where(Directory.Exists)
            .SelectMany(directory => Directory.EnumerateFiles(directory, Extension, SearchOption.TopDirectoryOnly))
            .Distinct(StringComparer.OrdinalIgnoreCase);

    private static string[] ResolveDirectories()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var result = new List<string>();

        void Add(string? directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
            {
                return;
            }

            string full = Path.GetFullPath(directory);

            if (seen.Add(full))
            {
                result.Add(full);
            }
        }

        // 1. 数据目录（exe 同级 data，或 --data / LABEL2TRAIN_DATA 指定的目录）
        Add(Path.Combine(AppDataPaths.DataDirectory, FolderName));

        // 2. 仓库根：从 exe 往上找一个含 .sln 的目录，取它的 data/weights
        Add(RepositoryDataWeights());

        // 3. 当前工作目录
        Add(Path.Combine(Directory.GetCurrentDirectory(), "data", FolderName));

        return [.. result];
    }

    /// <summary>从 <see cref="AppContext.BaseDirectory"/> 往上找仓库根（含 *.sln 的目录）。</summary>
    private static string? RepositoryDataWeights()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        for (int depth = 0; depth < 8 && directory is not null; depth++)
        {
            bool hasSolution = directory
                .EnumerateFiles("*.sln", SearchOption.TopDirectoryOnly)
                .Any();

            if (hasSolution)
            {
                return Path.Combine(directory.FullName, "data", FolderName);
            }

            directory = directory.Parent;
        }

        return null;
    }
}
