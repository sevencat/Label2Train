namespace Label2Train.Train.Training;

/// <summary>
/// 这次训练实际用的结构。写进检查点元数据，导出 ONNX 时按同样的结构重建模型
/// ——键名对不上的话 <c>load_state_dict</c> 会静默丢参数，导出的模型就是半个随机网。
/// </summary>
/// <param name="Queries">query 数。</param>
/// <param name="EncoderLayers">编码器层数（官方 RF-DETR 为 0）。</param>
/// <param name="DecoderLayers">解码器层数。</param>
/// <param name="Heads">注意力头数。</param>
/// <param name="DinoEmbedding">主干宽度。</param>
/// <param name="DinoDepth">主干 block 数。</param>
/// <param name="DinoHeads">主干头数。</param>
/// <param name="DinoPatch">主干 patch 边长（官方 medium 16，普通 DINOv2 14）。</param>
/// <param name="DinoImage">主干 pos_embed 对应的输入边长。</param>
/// <param name="GroupDetr">Group-DETR 组数。</param>
/// <param name="UseRfDetr">是不是 RF-DETR（否则是简化 DETR）。</param>
public sealed record DetrArchitecture(
    int Queries,
    int EncoderLayers,
    int DecoderLayers,
    int Heads,
    int DinoEmbedding,
    int DinoDepth,
    int DinoHeads,
    int DinoPatch,
    int DinoImage,
    int GroupDetr,
    bool UseRfDetr,
    int FfnSize = 1024,
    int DeformHeads = 0,
    int DeformLevels = 4,
    int DeformPoints = 4,
    bool UseRefPointHead = false,
    bool UseC2fProjector = false);
