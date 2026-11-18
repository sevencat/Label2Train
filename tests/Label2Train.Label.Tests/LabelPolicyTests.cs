using Label2Train.Core.Dataset;
using Label2Train.Label;
using Xunit;

namespace Label2Train.Label.Tests;

public class LabelPolicyTests
{
    [Fact]
    public void Only_DatabaseForm_IsOpenable()
    {
        Assert.True(LabelDatasetPolicy.CanOpen(DatasetFormat.TinyLabelDb));
        Assert.False(LabelDatasetPolicy.CanOpen(DatasetFormat.Yolo));
        Assert.False(LabelDatasetPolicy.CanOpen(DatasetFormat.Coco));

        Assert.Equal(
            [DatasetFormat.Yolo, DatasetFormat.Coco],
            LabelDatasetPolicy.ExportTargets);
    }

    [Fact]
    public void EnsureOpenable_RejectsExportOnlyFormats()
    {
        var error = Assert.Throws<NotSupportedException>(
            () => LabelDatasetPolicy.EnsureOpenable(DatasetFormat.Coco));

        Assert.Contains("导出", error.Message, StringComparison.Ordinal);
    }
}
