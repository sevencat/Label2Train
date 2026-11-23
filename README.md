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

RF-DETR 从零训练时，前几轮 mAP 基本都是 0（24 张图、batch 4 时前几十轮就是 0），必须用官方
预训练权重热启动。程序只认**一个 `.safetensors` 文件**，而且必须是**官方整网权重**：

- 不能是 `.pth`：代码按 safetensors 的键名直接读，文件扩展名不匹配就不会被扫到。
- 键名必须带 `backbone.0.` 前缀（`RfDetrWeightsProbe.RootPrefix`）。`RfDetrWeightsProbe`
  靠这些键的形状反推主干深度、patch 步长、解码器层数、query 数——**权重结构决定建什么模型**，
  所以塞一份纯 DINOv2 主干（只有 `encoder.encoder.…`）是不行的。

### 1. 下载

来源是 HuggingFace 上 Roboflow 官方的分尺寸仓库，文件名统一叫 `model.safetensors`：

| 尺寸 | 仓库 | 备注 |
|---|---|---|
| nano | `Roboflow/rf-detr-nano` | 最小的一档 |
| small | `Roboflow/rf-detr-small` | |
| medium | `Roboflow/rf-detr-medium` | **本仓库调试用的就是它**，128.6 MB（实测可载入 302 个张量） |
| base | `Roboflow/rf-detr-base` | |
| large | `Roboflow/rf-detr-large` | 最吃内存/显存 |

国内直连 `huggingface.co` 常超时，换 `hf-mirror.com` 镜像即可（域名换掉，路径完全一样）。
命令行走镜像（PowerShell，在仓库根执行；`$dir` 就是第 2 节里对应的那个目录）：

```powershell
$dir = 'src\Label2Train.App\bin\Debug\net10.0\data\weights'   # 换成你的 <数据目录>\weights
New-Item -ItemType Directory -Force -Path $dir | Out-Null

curl.exe -L --fail --retry 2 -o "$dir\rf-detr-medium.safetensors" `
  https://hf-mirror.com/Roboflow/rf-detr-medium/resolve/main/model.safetensors
```

不想用命令行就把下面的链接粘进浏览器，另存为 `rf-detr-medium.safetensors`（放进 `$dir`）：

```
https://hf-mirror.com/Roboflow/rf-detr-medium/resolve/main/model.safetensors
```

（官方直连是同一路径，域名换成 `huggingface.co`。）

下完对一下体积——medium 应是 **128.6 MB**，明显偏小说明中断了：

```powershell
(Get-Item "$dir\rf-detr-medium.safetensors").Length / 1MB
```

再确认是不是官方整网权重（输出 `True` 才对；`False` 说明下成了纯主干或别的包装格式）：

```powershell
$f = "$dir\rf-detr-medium.safetensors"
$s = [IO.File]::OpenRead($f)
$n = (New-Object IO.BinaryReader($s)).ReadInt64()
$b = New-Object byte[] $n
$s.Read($b, 0, $n) | Out-Null
$s.Close()
[Text.Encoding]::UTF8.GetString($b) -match '"backbone\.0\.'
```

只有 `.pth` 的话得自己转：用 torch 读出 state_dict，再用
`safetensors.torch.save_file(sd, 'model.safetensors')` 另存，键名保持原样。

### 2. 放在哪里

落点永远是**数据目录**（默认 `<exe 目录>/data`，可用 `--data` / `LABEL2TRAIN_DATA` 改）下的
`weights` 子目录：

| 你的运行方式 | 放这里 |
|---|---|
| `dotnet run` / 调试（F5） | `src\Label2Train.App\bin\Debug\net10.0\data\weights\` |
| Release 构建 | `src\Label2Train.App\bin\Release\net10.0\data\weights\` |
| 发布/绿色版（拷走的 exe 目录） | `<exe 所在目录>\data\weights\` |
| 用了 `--data D:\l2t` 或 `LABEL2TRAIN_DATA` | `D:\l2t\weights\` |

举例，默认调试那一份的完整路径：

```
F:\code\Label2Train\src\Label2Train.App\bin\Debug\net10.0\data\weights\rf-detr-medium.safetensors
```

几个容易踩的点：

- 不是**仓库根目录**的 `data`，是 exe 旁边的 `data`；换 `--data` 时权重不会跟着走。
- `weights` 只扫**一层**（`SearchOption.TopDirectoryOnly`），放进 `weights\rfdetr\` 这种子目录不会被发现。
- 文件名随便，后缀必须是 `.safetensors`；同时放多个时，名字含 `rfdetr` / `rf-detr` 的优先，
  再按文件名升序取第一个——建议**只放一个**，避免默默用错尺寸。
- 放在 `bin` 下已被 `.gitignore` 忽略，不会误提交；换机器、发布时需手动拷贝。

### 3. 另一种方式：环境变量

文件不想放在数据目录里（比如统一放 `D:\models\`）就设 `LABEL2TRAIN_RFDETR_WEIGHTS`，写**绝对路径**：

```powershell
# 只对当前 PowerShell 会话里启动的程序生效
$env:LABEL2TRAIN_RFDETR_WEIGHTS = 'D:\models\rf-detr-medium.safetensors'
dotnet run --project src\Label2Train.App
```

```powershell
# 用户级永久生效（值写进注册表，之后新开的程序都能读到；已开的程序要重开）
[Environment]::SetEnvironmentVariable('LABEL2TRAIN_RFDETR_WEIGHTS', 'D:\models\rf-detr-medium.safetensors', 'User')
```

也可以写进 **exe 同级的 `.env`**（纯文本，一行一个 `KEY=VALUE`，程序启动时读进环境变量）：

```
LABEL2TRAIN_RFDETR_WEIGHTS=D:\models\rf-detr-medium.safetensors
```

- 值两边的引号会被自动去掉，路径有空格也能直接写；`#` 开头的行是注释。
- `.env` **不会覆盖**已存在的同名环境变量；改完要重启程序。
- 指的文件不存在时不会报错，自动回退到第 2 节的目录扫描。

完整查找顺序（`src/Label2Train.Train/Models/RfDetr/RfDetrWeightSource.cs`）：

1. 环境变量 `LABEL2TRAIN_RFDETR_WEIGHTS`（文件确实存在才用，否则继续往下找）
2. `<数据目录>/weights/*.safetensors`（名字含 `rfdetr` / `rf-detr` 的优先）
3. 都没有 → 随机初始化，训练日志打印提示

### 4. 确认生效

训练开始的日志会打印实际用的哪份权重：

```
rfdetr: 预训练权重 F:\code\Label2Train\src\Label2Train.App\bin\Debug\net10.0\data\weights\rf-detr-medium.safetensors
```

没配上时是这句（同时学习率会用 1e-3 而不是 1e-4）：

```
rfdetr: 未配置预训练权重（把 model.safetensors 放到 ...\data\weights，或设环境变量 LABEL2TRAIN_RFDETR_WEIGHTS）；本次从随机初始化开始
```

装载结果紧跟其后，一行说清载入 / 跳过多少张量（medium + 3 个类别的数据集实测）：

```
rfdetr: 载入 302 个张量（跳过 159 个，2 个参数保持随机初始化：class_embed.bias、class_embed.weight）
```

`class_embed` 保持随机是**正常的**：官方是 COCO 91 类，你的类别数不同，装载只按形状匹配，
对不上的层就留随机初始化。同理，跳过的 159 个是检查点里有、本项目结构里没有的（`mask_token`、
`enc_output`、分组辅助头等）。

### 5. 排错

| 现象 | 原因 |
|---|---|
| 日志还是「未配置预训练权重」 | 放错位置（仓库根的 `data`、`weights` 的子目录）、数据目录被 `--data` 改了、后缀不是 `.safetensors` |
| 报「不是官方 RF-DETR 权重（没有 backbone.0. 前缀的键）」 | 下成了纯 DINOv2 主干 / transformers 包装版（键名多一层前缀） |
| 训练跑了但指标还是长时间 0 | 权重没生效（看第 4 节日志），或 epoch 太少（24 张图每轮仅 6 步） |
| 文件大小对不上（medium 应 128.6 MB） | 下载中断，重下 |

## RF-DETR 训练超参与曲线判读

界面上只暴露 epoch / batch / imgsz / device，学习率与批次顺序由运行器决定
（`src/Label2Train.Train/Runners/DetrDatasetTrainingRunner.cs` 与 `Sessions/DetrTrainingSession.Run.cs`）：

| 项 | 取值 | 说明 |
|---|---|---|
| 学习率 | 有预训练 **1e-4**，无预训练 1e-3 | 官方 AdamW 微调量级；batch 被截到 4 时，2e-4 会让小数据集上的 loss 震荡回升；**1e-3 配官方权重实测约 1/4 的初始化会冻结**，见「数据增强对照实验」 |
| 随机头梯度 | 有效 lr 封顶 **1e-3**（`HeadGradientScale`：1e-4→×10，5e-4→×2，1e-3→×1） | 分类头 / query / 参考点是随机初始化的，不放大就学不动，放大过头会把 sigmoid 推爆 |
| batch | 上限 **4** | DETR 显存占用高，界面填更大也会被截断 |
| 调度 | 1 个 epoch 线性 warmup + 余弦退火到初始 lr 的 1% | 总步数 = `epochs × 每轮批次数` |
| 批次顺序 | **每个 epoch 重新打乱**（按 epoch 定种子，可复现） | `TrainerOptions.Shuffle` 在委托式 `Fit` 上不生效，所以打乱放在 `BuildEpochBatches` 里做 |
| 数据增强 | hflip 0.5、缩放 ±0.25、平移 ±0.1、HSV(0.015 / 0.7 / 0.4)，**无 mosaic** | 只作用于训练集，验证集是纯 letterbox；增强种子按 `(epoch, 样本序号)` 派生，同一个 seed 可复现；日志第一屏会打印 `augment: ...` |
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

## RF-DETR 分类为什么慢：过拟合实验记录

结论先说，这条线后来分成了两半：

- **分类本身确实收敛慢**（下面的分步表）：300 个 query 对几个目标，正负比 1:200，未匹配 query 的框
  又不受监督，分数只会缓慢爬升。这是模型/损失的特性，不是写错。
- **但慢的很大一部分是移植 bug**：用官方 Python 实现对照后发现，C# 前向的参考点与框解码语义不对
  （参考点没在 logit 空间维护、最终框没加回参考点）。修完之后同样的 8 图实验 **3 轮就有 mAP50**，
  修之前 3 轮恒为 0、L1 完全不降——详见「官方 Python 对照」一节。

### 实验设定

8 张图反复训同一批（官方 medium 权重、224×224、lr 1e-3、`(正样本 + 负样本) / 正样本数` 归一化）。
这种方式能把「学不动」和「学得慢」区分开：数据量小到必须背下来，背不下来就是实现问题。

| 步数 | L1 | GIoU | 最佳 query 的 IoU | top1 分数 | mAP50 |
|---|---|---|---|---|---|
| 1 | 3.25 | 2.63 | 0.05 | 0.49 | 0.000 |
| 400 | 0.77 | 1.48 | 0.30 | 0.30 | 0.001 |
| 800 | 0.11 | 0.41 | 0.69 | 0.22 | 0.129 |
| 1200 | 0.04 | 0.18 | 0.72 | 0.28 | 0.212 |
| 2000 | **0.002** | **0.008** | 0.74 | **0.34** | 0.283 |

- **框回归完全正常**：2000 步 L1 到 0.002、GIoU 0.008，每个 GT 都有 query 贴着它。
- **分类是慢，不是卡死**：top1 分数 0.34 还在缓慢上升；梯度范数 2~7、`class_embed` 权重范数在
  1.01 附近持续变动，说明梯度在传、参数在更新，只是缺少方向上的共识。

### 官方 Python 对照：参考点语义 bug

这张表只说明了「慢」，没法排除「写错」。于是用官方 `rfdetr`（venv + 官方 medium 权重，同一批 8 图、
224×224、bs 8、lr 1e-3、3 轮）跑同样的过拟合，再和自己的实现逐项对数值。

官方前向的关键点是**参考点全程活在 logit 空间**（`refpoints_unsigmoid`）：

- two-stage 提议：`grid_proposals`（`unsigmoid=True`）与 `enc_out_bbox_embed` 输出**都是 logit**，直接相加；
- 每层 decode 后 `references = references + bbox_embed(hs)`，累加前不 sigmoid；
- 采样点、位置编码用 `references.sigmoid()`；
- 最终框 `= (bbox_embed(decoder_norm(hs)) + refpoints_unsigmoid).sigmoid()`。

旧实现（错的三处）：

1. 把 `refpoint_embed` 的参数值当 0~1 坐标直接用，没经过 sigmoid；
2. 每层 refine 只在**中心两维**上做 `sigmoid(centers + delta)`，宽高永远沿常量；
3. 最终框只取 `sigmoid(bbox_embed(queries))`，**完全没加回参考点** → 框只能从 0.5 起步学绝对位置。

修好（`RfDetrDetector.forward` 里 `lastReference` + `AddReference`）后，同样的 3 轮：

| epoch | loss | cls | l1 | giou | top1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| 1 | 10.29 | 1.186 | 5.445 | 3.654 | 0.146 | 0.000 | 0.000 |
| 2 | 4.47 | 0.299 | 1.831 | 2.344 | 0.409 | 0.081 | — |
| 3 | 2.65 | 0.705 | 0.645 | 1.298 | — | **0.277** | 0.094 |

修之前的同一份数据：l1 稳在 5.43、mAP50 恒 0、top1 0.015。**所以「分类慢」的主因不是损失设计，
而是框根本没被解码到目标上，分类跟着一起学不动。**

顺带确认/修掉的两件事：

- **权重装载无缺口**：审计 `RfDetrPretrainedReport.Untouched` 只有 `class_embed.weight/bias`
  （官 COCO 91 类，类别数不同本来就该随机），`bbox_embed` 三层权重范数 21.07 / 37.11 / 3.50 都已装上。
- **IA-BCE 按官方口径实现并默认开启**（`IouAwareBce = true`）：`neg_w = p²`、
  `pos_w = clamp(p^0.25 · IoU^0.75, 0.01)`，`loss = Σ(neg_w·x − logσ(x)·(pos_w+neg_w)) / numPos`，
  不是软标签 BCE；有对照官方数值的单元测试（`DetrIouAwareBceTests`）。
- **ONNX 导出器同步**：把 logit 空间累加与「最终加回参考点」复刻进 `RfDetrOnnxExporter`，
  `RfDetrOnnxParityTests` 的小模型与官方权重两个用例都逐元素对齐。

### 根因：正负样本比例

300 个 query × 3 类，对每图 4~5 个目标就是约 **1:200**。更麻烦的是未匹配的 query **框不受监督**，
它们会挤在真目标旁边当高分重复框，分类头必须在这种几乎相同的特征里把几十个正样本挑出来。
这解释了「曲线缓慢但正确爬升」，也解释了为什么把学习率调大 10 倍、100 倍都只是原地打转。

### 途中修掉的归一化 bug

问题：上一版按「正负样本各自求均值」，等于把每个负样本的权重压到正样本的 1/200。

| 归一化 | 400 步 mAP50（8 图） | 现象 |
|---|---|---|
| 正负各自求均值 | 0.169 | 每图约 20 个未匹配 query 分数飙到 0.97，压在正确框前面 |
| `(posSum + negSum) / numPos`（现用，标准口径） | —（见右列） | 分数 ≥0.1 的框从 100 个（饱和）降到 4 个，负样本真正被压住 |

现在的实现是 `DetrLoss` 里的 `elementwise.sum() / Math.Max(1, positiveCount)`，与 DETR / YOLO 一致。

### query 数与官方权重切片

`DetrTrainingOptions.Queries` 默认 **50**（原为 20，且在有官方权重时被 `pretrained?.Queries`
强制覆盖成 300，界面上写的和实际用的并不一致）。

- 结构参数 `DetrArchitecture.Queries` 现在一律取用户设置，检查点元数据、ONNX 导出、评估都跟着走。
- 官方 `query_feat` 存的是 300 行（×13 组），装载时**按行切前 N 行**，组 0 排在最前。
- 切片**只允许** `query_embed` / `refpoint_embed`。`class_embed` 绝不能按行截：官方是 COCO 91 类，
  截前 N 行会把别人的类别语义套到你的类别上。
- 实测（640 张图、bs 8、224、3 轮）：Q=300 与 Q=50 都载入 302 个张量、只有 `class_embed` 随机，
  说明 query / refpoint 是切片装上而不是被跳过；起始分类损失 30.87 → **9.36**（负样本少了 6 倍）。
  3 轮内两者 mAP 都在 0 附近，**不足以判断谁最终更高**，别把这一段当结论。

### 数据增强对照实验（640 / 160 真实数据）

配置：3 类（square / triangle / circle，2559 个框、每图 3~5 个）、train 640 / val 160、imgsz 224、
batch 4、Q=50、官方 medium 权重、每次评估取 100 张验证图；`torch.random.manual_seed`
固定种子，同一份数据上只切「增强开关」。跑了 8 轮 / 20 轮两个档位，160 步/轮。

> **速度要看显存够不够**：同形状单步（前向 + 损失 + 反向 + AdamW）在显存宽裕时是 **0.10 s**
> （→ 约 16 s/轮）；但 batch 4 / 224 实跑时显存占到 **7885 / 8151 MiB**，一旦贴到上限，
> Windows 把显存分页到内存，单步掉到 **3.21 s（514 s/轮）**——36 倍的差距，且 `nvidia-smi`
> 上看利用率还是 100%，很容易被当成「模型就这个速度」。要复跑对照实验先看显存，
> 8 GB 卡建议把 batch 降到 2。cuda，评估另算。

> **配对种子的坑**：`torch.random.manual_seed` 在同一个进程里连续调用并不能把权重初始化完全复位
> （实测同一个 seed 连建两次模型，权重不同；ONNX 对拍测试因此偶发全 NaN，已改为用 .NET `Random`
> 按固定种子写权重，见 `RfDetrOnnxParityTests.FillDeterministically`）。所以上面的「配对」只在
> 「一组一个进程」时严格成立；同一进程里连跑的几组，初始化会依次漂移，不是真正同一起点。
> 好在两种增强设置之间的差距（0.13~0.16）远大于同配置换种子的波动（0.026），结论方向不受影响。

**生产档位（有官方权重时 runner 自动给 lr 1e-4）· 8 轮档位**：

| 种子 | 增强 | mAP50@e2 | @e4 | @e6 | @e8 | mAP50-95@e8 |
|---|---|---|---|---|---|---|
| 4 | 开 | 0.002 | 0.169 | 0.330 | **0.401** | 0.165 |
| 4 | 关 | 0.000 | 0.025 | 0.224 | 0.292 | 0.134 |
| 1234 | 开 | 0.000 | 0.130 | 0.266 | **0.327** | 0.139 |
| 1234 | 关 | 0.000 | 0.081 | 0.165 | 0.227 | 0.104 |

**但把轮数放到 20，结论就反过来了**（同样 lr 1e-4、`EvalEvery=2`、100 张验证图）：

| 种子 | 增强 | @e8 | @e12 | @e16 | @e18 | @e20 | mAP50-95@e20 |
|---|---|---|---|---|---|---|---|
| 4 | 开 | 0.508 | 0.622 | 0.694 | 0.718 | 0.735 | 0.478 |
| 1234 | 开 | 0.523 | 0.626 | 0.727 | 0.754 | 0.761 | 0.505 |
| 4 | 关 | 0.500 | 0.764 | 0.854 | 0.885 | **0.894** | **0.683** |

- 8 轮时「开增强 +0.11」是**调度造成的假象**：8 轮档位的 cosine 到 e8 就归零，等于「步数不够」，
  增强让前几轮学得更慢，但那时关增强那条也才 0.29，两边都没收敛，谁快谁好看而已。
  （同一份数据、20 轮档位下 e8 是 0.500 vs 0.508，其实持平。）
- 给够轮数之后**关增强明显更好**：0.894 vs 0.735 / 0.761（+0.13 ~ +0.16 mAP50，mAP50-95 差 0.18）。
  这份数据是程序渲染的：背景、光照、目标外观都高度一致，增强只会让模型「多学噪声少学分布」。
- 真实照片数据上未必如此（背景杂、光照变、目标姿态多），所以默认值暂时**保留 `true`**
  （对齐官方 RF-DETR 配方），等有真实照片集再跑同一套对照再定。

**怎么用**：训练页「训练目标设置」里有一个「数据增强」勾选框（只在选到 RF-DETR 时可用，
YOLO 的 hflip / mosaic 在它自己的 pipeline 里）；值从 `TrainingTarget` → `TrainViewModel.Augment`
→ `TrainingRunOptions.Augment` → `DetrTrainingOptions.Augment` 一路传下去；开训时会打印
`augment: hflip ... hsv ...` 或 `augment: 关闭（纯 letterbox，仅用于对照）`，事后翻日志能确认这一轮到底用没用。

**速度**：20 轮档位仍是 **0.09 s/步 → 14 s/轮**（cuda，评估另算），20 轮 ≈ 5 分钟。

**lr 1e-3（从零训的档位）更快，但会冻死**：

| 随机头梯度 | 种子 | 增强 | 8 轮 mAP50 |
|---|---|---|---|
| ×10（原始） | 4 | 开 | 0.628 |
| ×10 | 4 | 关 | **冻结**（loss 恒 10.238） |
| ×10 | 1234 | 开 | 0.608 |
| ×10 | 1234 | 关 | **冻结**（loss 恒 10.238） |
| 封顶 1e-3 | 4 | 开 | **冻结**（loss 恒 16.0） |
| 封顶 1e-3 | 4 | 关 | 0.646 |
| 封顶 1e-3 | 1234 | 开 | 0.660 |
| 封顶 1e-3 | 1234 | 关 | 0.769 / mAP50-95 0.557 |

读法：1e-3 不炸的时候确实快得多（0.65~0.77 vs 1e-4 的 0.40），但 **8 组里 3 组直接冻死**，
所以生产仍用 1e-4；想更快应该加轮数，别靠抬 lr。

**冻结长什么样**：从某一轮起 `loss` 几乎不动（10.238 / 16.0）、`det 0.0/img`、`top1 ≈ 0.01`、
mAP 恒 0。出事那一轮的梯度范数能冲到 **75**（正常在 1e-2 ~ 1e0），一步就把参考点推过 sigmoid
的肩部；之后范数掉到 **2e-2**，loss 只剩下 0.1~0.3% 的抖动，`l1` / `giou` 到小数点后三位都不变。
两组**不同初始化**能冻在同一个 loss 上，说明预测贴在饱和端之后，loss 只剩「与预测无关的目标项」
—— 继续跑不会恢复，降 lr 也救不回来，只能换配置重跑。
（这也解释了之前 ONNX 对拍偶发的一次 NaN：同一类数值飞掉。）

**这一轮做的两件事**（都不是根治，是把坑显形 + 收窄）：

1. `DetrTrainingSession.HeadGradientScale`：随机头的梯度放大从固定 ×10 改成「有效 lr 封顶 1e-3」
   （1e-4 → ×10，1e-3 → ×1，5e-4 → ×2），不再出现 1e-2 这种量级的头学习率；
   冻结率从 2/4 降到 1/4，但没消失。
2. 诊断：`Trainer.LastGradientNorm`（每轮平均、裁剪前，现在打在每轮日志末尾的 `grad` 上）；
   开训前若「有官方权重且 lr > 3e-4」提示一次；训练中出现「loss 连续两轮降不到 1% 且 mAP 还是 0」
   再提示一次「基本可以判定冻住了，换 lr 重跑」——不用用户对着 mAP 恒 0 的曲线猜。
   （判据没用「梯度恰好为 0」：冻结时梯度是「很小」而不是「为零」，loss 会抖百分之几，
   最初按 0 判的那一版实测漏报。）

**还没根治**：为什么 1e-3 下约 1/4 的初始化会在头几轮饱和。候选方向：`query_embed` 的 N(0,1)
初始化、参考点用 `uniform(0, 1)`（logit 空间！sigmoid 后只能落在 0.5~0.73）、
以及 `references + bbox_embed(...)` 逐层累加的幅值控制。

### 还能继续做的事（按性价比）

1. ~~打开 IA-BCE~~：已按官方口径实现并默认开启（`IouAwareBce = true`），见上一节。
2. ~~给 DETR 加数据增强~~：已完成。训练集用 `TrainingPresets.DetrTrainPreprocess`：hflip 0.5、
   缩放 ±0.25、平移 ±0.1、HSV(0.015 / 0.7 / 0.4)，**不开 mosaic**（官方 RF-DETR 配方里没有，
   小数据集拼贴后目标常只剩半截）；验证集仍是纯 letterbox；增强种子按 `(epoch, 样本序号)`
   派生，可复现。`DetrTrainingOptions.Augment`（默认 `true`）已暴露到「训练目标设置」的
   「数据增强」勾选框（只对 RF-DETR 可用）。对照结论见上一节：8 轮时开增强「更好」是调度欠训造成的
   假象，**20 轮下关增强 0.894 vs 开 0.735/0.761**；默认值暂时保留 `true`（对齐官方配方），
   **需要一份真实照片数据集复跑这套对照再决定**。
3. ~~加轮数看平台~~：已跑到 20 轮，关增强到 e20 仍在微涨（0.885 → 0.894），开增强 e20 是 0.735 / 0.761；
   再往上（30 轮）预计 +0.01~0.02，收益远小于第 2、4 条。
4. 想再快就试 lr 5e-4（新的封顶规则下随机头 ×2），但要盯着 `warn` 那一行，别让冻结白跑。
5. 根治冻结（上面「还没根治」的三条）。
6. 拿一份真实照片数据集（背景 / 光照有变化）复跑第 2 条的对照，再决定 `Augment` 的默认值。
