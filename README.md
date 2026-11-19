# Label2Train

标注 → 训练 → ONNX 导出的一体化桌面工具：在标注端整理数据集，在训练端跑 YOLO / RF-DETR，
导出可被推理端直接加载的 ONNX 模型。

## 目录结构

```
src/Label2Train.App     桌面应用（标注端 + 训练端）
src/Label2Train.Label   标注核心
src/Label2Train.Core    数据目录、数据库、设置、推理运行器等公共设施
src/Label2Train.Train   训练引擎：模型结构 / 数据管线 / 训练循环 / 检查点与导出
tests/                  单元与集成测试
tools/                  辅助脚本
```

- 目标框架：`net10.0`
- 构建：`dotnet build Label2Train.sln`
- 测试：`dotnet test`

## 数据目录

默认是 **exe 同级 `data/`**，可用命令行 `--data <目录>` 或环境变量 `LABEL2TRAIN_DATA`
（也可写进 `.env`）覆盖，命令行优先。整个程序是「绿色」的：拷走 exe 目录就把数据一起带走。

## RF-DETR 预训练权重

RF-DETR 从零训练时，前几轮 mAP 基本都是 0；用官方预训练权重热启动才有意义。
权重只需一个 `.safetensors` 文件（不能是 `.pth`）：

| 用途 | 路径 |
|---|---|
| 训练实际读取的位置 | `<数据目录>/weights/*.safetensors`，默认即 `<exe目录>/data/weights/` |
| 当前本机示例 | `src/Label2Train.App/bin/Debug/net10.0/data/weights/rf-detr-medium.safetensors`（约 135 MB） |
| 原始备份示例 | `F:\ai\rfdetr\model.safetensors`（同目录的 `rf-detr-medium.pth` 不可直接用） |

**通常不需要手动配置**——文件放到位就会自动生效。查找顺序见
`src/Label2Train.Train/Models/RfDetr/RfDetrWeightSource.cs`：

1. 环境变量 `LABEL2TRAIN_RFDETR_WEIGHTS`（支持带引号的绝对路径，也可写进 `.env`）
2. `<数据目录>/weights/` 下的 `*.safetensors`，文件名含 `rfdetr` / `rf-detr` 的优先
3. 都没有 → 随机初始化，训练日志会打印提示

训练开始时的日志会明确写出实际用了哪份权重：

```
rfdetr: 预训练权重 ...\data\weights\rf-detr-medium.safetensors
```

注意事项：

- `data` 是**数据目录**下的 `weights`，不是仓库根目录；换 `--data` 时要一并带上权重，或用环境变量指过去。
- 权重文件放在 `bin` 下，已被 `.gitignore` 忽略，不会误提交；换机器 / 发布时需手动拷贝。
- 只接受 safetensors 格式（代码直接按键名读取），`.pth` 需先转换。

## RF-DETR 训练超参与曲线判读

界面上只暴露 epoch / batch / imgsz / device，学习率与批次顺序由运行器决定
（`src/Label2Train.Train/Runners/DetrDatasetTrainingRunner.cs` 与 `Sessions/DetrTrainingSession.Run.cs`）：

| 项 | 取值 | 说明 |
|---|---|---|
| 学习率 | 有预训练 **1e-4**，无预训练 1e-3 | 官方 AdamW 微调量级；batch 被截到 4 时，2e-4 会让小数据集上的 loss 震荡回升 |
| batch | 上限 **4** | DETR 显存占用高，界面填更大也会被截断 |
| 调度 | 1 个 epoch 线性 warmup + 余弦退火到初始 lr 的 1% | 总步数 = `epochs × 每轮批次数` |
| 批次顺序 | **每个 epoch 重新打乱**（按 epoch 定种子，可复现） | DETR 没有数据增强，顺序就是唯一的随机性；`TrainerOptions.Shuffle` 在委托式 `Fit` 上不生效，所以打乱放在 `BuildEpochBatches` 里做 |
| 评估阈值 | score > **0.05**（一对一匹配，无 NMS） | 与 YOLO 路径的 0.001 不同，DETR 靠低阈值覆盖召回 |
| 归一化 | AdamW + 梯度全局范数裁剪 1.0 | |
| imgsz | 必须是 **112 的倍数**（224 / 336 / 448 / 560 / 672） | 112 = patch 14 与参考网格 16 的公倍数。填别的值会被就近纠正并在日志里打印 `imgsz : 352 → 336`；DETR 的预处理按 112 校验，不套用 YOLO 的 32 |

### CPU 上的实测耗时（batch 4，官方权重）

| 场景 | forward | loss | backward | 每步 |
|---|---|---|---|---|
| 448×448 | 0.92 s | 0.00 s | 1.66 s | 2.55 s |
| 336×336 | 0.54 s | 0.00 s | 0.88 s | 1.42 s |
| 448 + 冻结 DINOv2 主干 | 0.95 s | 0.00 s | 0.24 s | 1.21 s |

- 没有 NVIDIA GPU 时，瓶颈就是 ViT 主干的 CPU 前向/反向，算法侧能省的不多；**降 imgsz 与冻结主干是最有效的两个杠杆**。
- 数据加载只占 4%（每批 4 张约 106 ms），线程数从 8 调到 16 无收益。

曲线怎么读：

- **看 loss，不看前二三十轮的 mAP**。mAP 是滞后指标：一个 query 要同时 score > 0.05 且 IoU > 0.5 才算命中。
  24 张图 / batch 4 → 每 epoch 只有 6 步，100 epoch 才 600 步，mAP 往往在几十轮后才离开 0。
- **mAP 曲线上零星的小尖峰（0.002~0.008）是噪声**，不是「学了又忘」：验证集很小时，单个框的得失就能让 mAP 跳这么多。
- **mAP50-95 恒为 0 属正常**：它要求 IoU 一路算到 0.95，比 mAP50 晚得多。
- **loss 上下震荡、不降反升**才是有问题：先降 lr（1e-4 → 5e-5），再看数据量与标签质量。
- 想更早确认「确实在学」，可用低阈值导出 ONNX 看预测框有没有贴住目标，比等 mAP 更直观。
