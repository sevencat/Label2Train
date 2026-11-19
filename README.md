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
