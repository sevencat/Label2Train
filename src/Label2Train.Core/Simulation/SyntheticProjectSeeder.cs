using System.Diagnostics;
using FreeSql;
using Microsoft.Extensions.Logging;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Localization;
using Label2Train.Core.Storage;

namespace Label2Train.Core.Simulation;

/// <summary>「生成模拟项目」的参数。</summary>
/// <param name="ProjectName">项目名（必填）。</param>
/// <param name="ImageCount">生成多少张图。</param>
/// <param name="ImageWidth">图片宽（像素）。</param>
/// <param name="ImageHeight">图片高（像素）。</param>
/// <param name="MinShapes">每张最少几个形状。</param>
/// <param name="MaxShapes">每张最多几个形状。</param>
/// <param name="ValRatio">验证集比例（0 表示不划分）。</param>
/// <param name="Seed">随机种子；固定后结果可复现。</param>
/// <param name="Description">项目备注；为空用默认文案。</param>
public sealed record SyntheticProjectOptions(
    string ProjectName,
    int ImageCount = 30,
    int ImageWidth = 640,
    int ImageHeight = 480,
    int MinShapes = 1,
    int MaxShapes = 3,
    double ValRatio = 0.2,
    int? Seed = 20261112,
    string? Description = null);

/// <summary>生成出来的项目概况。</summary>
public sealed record SyntheticProjectReport(
    long ProjectId,
    int ImageCount,
    int AnnotatedImages,
    int ShapeCount,
    IReadOnlyList<string> Splits);

/// <summary>
/// 在当前工作库里造一个"模拟项目"：建项目 → 建三个类别 → 生成 PNG 走图片服务进
/// <c>projects/{id}/images/</c> → 每张图按真值写 labelme 标注。
/// <para>
/// 与 <see cref="SimulationPipeline"/> 不同：这里不落任何外部数据集目录，也不建库，
/// 图片与标注全部按 TinyLabel 的「库 + 项目目录」约定存放，等同于用户手动导入图片并逐张标注完。
/// </para>
/// </summary>
public sealed class SyntheticProjectSeeder(
    IFreeSql orm,
    TinyLabelProjectService projects,
    TinyLabelClassService classes,
    TinyLabelImageService images,
    TinyLabelTaskService tasks,
    ILogger<SyntheticProjectSeeder>? logger = null)
{
    /// <summary>三个类别名对应的颜色，与模拟图里画的颜色一致。</summary>
    private static readonly string[] ClassColors = ["#3B82F6", "#22C55E", "#EF4444"];

    public async Task<SyntheticProjectReport> CreateAsync(
        StorageLocation root,
        SyntheticProjectOptions options,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(root);
        ArgumentNullException.ThrowIfNull(options);

        if (options.ImageCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), Text.Get("Synthetic.ImageCountInvalid"));
        }

        var started = Stopwatch.StartNew();

        var generation = new SyntheticOptions(
            options.ImageCount,
            Math.Max(16, options.ImageWidth),
            Math.Max(16, options.ImageHeight),
            Math.Max(1, Math.Min(options.MinShapes, options.MaxShapes)),
            Math.Max(1, options.MaxShapes),
            options.Seed);

        IReadOnlyList<SyntheticImage> generated = new SyntheticImageGenerator(generation).Generate();
        string staging = Path.Combine(Path.GetTempPath(), "label2train-synthetic", Guid.NewGuid().ToString("N"));

        try
        {
            // 图片服务的入口是文件路径：先落到临时目录，再由它复制进项目目录
            Directory.CreateDirectory(staging);

            foreach (SyntheticImage image in generated)
            {
                await File.WriteAllBytesAsync(Path.Combine(staging, image.FileName), image.Png, ct).ConfigureAwait(false);
            }

            long projectId = await projects
                .CreateAsync(root, options.ProjectName, options.Description ?? Text.Get("Synthetic.ProjectDescription"), ct)
                .ConfigureAwait(false);

            for (int index = 0; index < SyntheticImageGenerator.ClassNames.Count; index++)
            {
                await classes
                    .AddAsync(root, projectId, SyntheticImageGenerator.ClassNames[index], ClassColors[index], ct)
                    .ConfigureAwait(false);
            }

            // 按 split 分批导入：图片服务一次只认一个 split
            var bySplit = generated
                .Select((image, index) => (
                    Path: Path.Combine(staging, image.FileName),
                    Split: SyntheticDatasetWriter.ResolveSplit(index, generated.Count, options.ValRatio)))
                .GroupBy(item => item.Split, StringComparer.OrdinalIgnoreCase)
                .ToList();

            foreach (var group in bySplit)
            {
                await images
                    .ImportFilesAsync(root, projectId, [.. group.Select(item => item.Path)], group.Key, copy: true, ct)
                    .ConfigureAwait(false);
            }

            // 标注：导入后的 RelPath 就是文件名，按它找回 imageId 再走任务服务（状态自动置 labeled）
            int annotated = 0;
            int shapes = 0;

            foreach (SyntheticImage image in generated)
            {
                ct.ThrowIfCancellationRequested();

                ImageEntity? row = orm.Select<ImageEntity>()
                    .Where(item => item.ProjectId == projectId && item.RelPath == image.FileName)
                    .First();

                if (row is null)
                {
                    // 内容哈希去重跳过了某张（固定种子下不会发生，防御一下）
                    logger?.LogWarning("模拟图 {Image} 没有进入项目 {Project}，跳过标注", image.FileName, projectId);
                    continue;
                }

                string json = SyntheticLabelingSession.BuildLabelmeJson(image, jitterPixels: 0, Random.Shared);

                await tasks
                    .SaveAnnotationAsync(root, row.Id, row.Revision, json, userId: 0, ct)
                    .ConfigureAwait(false);

                annotated++;
                shapes += image.Shapes.Count;
            }

            IReadOnlyList<string> splits = [.. bySplit.Select(group => group.Key)];

            logger?.LogInformation(
                "模拟项目 {Project} 生成完成：{Images} 张图、{Annotated} 张已标注、{Shapes} 个框，split {Splits}，耗时 {Elapsed:0.0}s",
                projectId,
                generated.Count,
                annotated,
                shapes,
                string.Join(",", splits),
                started.Elapsed.TotalSeconds);

            return new SyntheticProjectReport(projectId, generated.Count, annotated, shapes, splits);
        }
        finally
        {
            TryDeleteStaging(staging);
        }
    }

    /// <summary>临时目录删不掉只记日志：留几个 PNG 不影响项目。</summary>
    private void TryDeleteStaging(string staging)
    {
        try
        {
            if (Directory.Exists(staging))
            {
                Directory.Delete(staging, recursive: true);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            logger?.LogWarning(error, "清理模拟图临时目录 {Path} 失败", staging);
        }
    }
}
