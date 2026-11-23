using Avalonia.Controls;
using Avalonia.Interactivity;
using Label2Train.Core.Localization;

namespace Label2Train.App.Views;

/// <summary>参数说明表的一行：参数名 / 当前值 / 作用与推荐。</summary>
public sealed record DetrParamRow(string Name, string Value, string Impact);

/// <summary>
/// 「参数说明」：把 RF-DETR 这次训练真正影响结果的参数列出来——当前值是多少、
/// 它改变什么、推荐怎么填。只读窗口，改值仍然走「训练目标设置」。
/// <para>
/// 之所以要有这个窗口：这些数字之间的耦合（query 数 ↔ 每图目标数、batch ↔ 显存上限、
/// imgsz ↔ 112 的倍数）写在代码注释里只有开发者看得到，而用户只看到四个数字。
/// </para>
/// </summary>
public partial class DetrParamsDialog : Window
{
    private static Localizer Loc => Localizer.Instance;

    /// <summary>XAML 预览用。</summary>
    public DetrParamsDialog()
        : this("rf-detr-medium.safetensors", true, 100, 4, 560, 50, true)
    {
    }

    public DetrParamsDialog(
        string weightsName,
        bool hasWeights,
        int epochs,
        int batchSize,
        int imageSize,
        int queries,
        bool augment)
    {
        InitializeComponent();

        RowList.ItemsSource = BuildRows(weightsName, hasWeights, epochs, batchSize, imageSize, queries, augment);
    }

    private static IReadOnlyList<DetrParamRow> BuildRows(
        string weightsName,
        bool hasWeights,
        int epochs,
        int batchSize,
        int imageSize,
        int queries,
        bool augment) =>
    [
        new(
            Loc.Get("Train.Pretrained"),
            weightsName,
            Loc.Get(hasWeights ? "Train.DetrHelpWeightsImpact" : "Train.DetrHelpWeightsMissingImpact")),
        new(
            "epochs",
            epochs.ToString(System.Globalization.CultureInfo.InvariantCulture),
            Loc.Get("Train.DetrHelpEpochsImpact")),
        new(
            "batch",
            batchSize.ToString(System.Globalization.CultureInfo.InvariantCulture),
            Loc.Get("Train.DetrHelpBatchImpact")),
        new(
            "imgsz",
            imageSize.ToString(System.Globalization.CultureInfo.InvariantCulture),
            Loc.Get("Train.DetrHelpImageSizeImpact")),
        new(
            Loc.Get("Train.Augment"),
            Loc.Get(augment ? "Train.AugmentOn" : "Train.AugmentOff"),
            Loc.Get("Train.DetrHelpAugmentImpact")),
        new(
            "queries",
            queries.ToString(System.Globalization.CultureInfo.InvariantCulture),
            Loc.Get("Train.DetrHelpQueriesImpact")),
        new(
            Loc.Get("Train.DetrHelpLrName"),
            Loc.Get(hasWeights ? "Train.DetrHelpLrPretrained" : "Train.DetrHelpLrScratch"),
            Loc.Get("Train.DetrHelpLrImpact")),
    ];

    private void OnClose(object? sender, RoutedEventArgs e) => Close();
}
