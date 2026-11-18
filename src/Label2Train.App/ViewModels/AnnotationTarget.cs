using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Model;
using Label2Train.Core.Storage;

namespace Label2Train.App.ViewModels;

/// <summary>
/// 标注窗口里可以连续标注的一张图：行数据 + 图片引用 + 它所在的 blob store。
/// 「保存并继续」在同一窗口里换图，靠的就是这份列表（不用关窗再开）。
/// </summary>
public sealed record AnnotationTarget(TinyLabelImageRow Row, ImageRef Image, IBlobStore Store);
