namespace Label2Train.Core.Infrastructure;

/// <summary>用户配置。字段只增不改语义；读旧文件时缺字段走默认值。</summary>
public sealed class AppSettings
{
    /// <summary>Light / Dark / Default。</summary>
    public string Theme { get; set; } = "Default";

    /// <summary>界面语言：<c>zh</c> / <c>en</c>；空表示跟随系统。</summary>
    public string Language { get; set; } = string.Empty;

    /// <summary>最近打开的数据集根目录。</summary>
    public List<string> RecentRoots { get; set; } = [];

    // 监听地址不在这里：它是进程级、改完要重启的设置，走环境变量 / .env（LABEL2TRAIN_LABEL_URL），
    // 界面上不写回，避免"改了一半要重启"的二义性

    /// <summary>启动时是否自动拉起内嵌 Web 服务。</summary>
    public bool StartWebOnLaunch { get; set; } = true;

    /// <summary>标注端上次打开的数据目录（含 label2train.db），启动时自动打开。</summary>
    public string LastDataDirectory { get; set; } = string.Empty;

    /// <summary>上次在标注库里选来训练的项目 Id；0 表示还没记过（库永远是标注端当前库）。</summary>
    public long LastTrainLibraryProjectId { get; set; }

    /// <summary>AI 标注用的 ONNX 模型路径（YOLO 导出）。</summary>
    public string AutoLabelModelPath { get; set; } = string.Empty;

    /// <summary>AI 标注置信度阈值。</summary>
    public double AutoLabelConfidence { get; set; } = 0.25;
}
