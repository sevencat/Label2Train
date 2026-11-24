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

## YOLO 训练参数（v8 / v11 / v26）：在哪改、改了会怎样

这一节的默认值都是**逐个与 ultralytics 8.4.210 数值对拍过**的。对拍不是看文档，而是同一份权重、
同一个 batch（4 图 320×320、3 类）、同一份 GT，比四个量：前向输出、loss 分项、整网梯度范数、
TAL 分配结果。v8 / v11 / v26 三家全部对齐（数字见下一节），所以下面的默认值可以直接当作「官方口径」，
改任何一个都会让训练偏离官方实现。

参数分三层，改的地方各不相同。

### 对齐核对结果（v8 / v11 / v26，2026-11-24 复测）

核对办法：把本实现建出来的网导出成 safetensors，官方 `DetectionModel(cfg=..., nc=3, ch=3)` 用
`load_state_dict(strict=False)` 载入同一组张量（v26 载 `yolo26.yaml`，两套头都载），两边吃同一份
输入和 GT，各自跑前向 + loss + `backward()`。TAL 分配结果靠在官方侧 patch
`TaskAlignedAssigner.forward` 抓。

| 量（官方 vs 本实现） | yolov8n | yolo11n | yolo26n |
|---|---|---|---|
| 前向输出和 | 相对差 < 1e-7 | 339477.9375 vs 339477.9（1e-7） | -95591.0703 vs -95591.06（1e-7） |
| loss 总项 | 差 0.02% | 20.2575569 vs 20.257563（3e-7） | 19.4518242 vs 19.451828（2e-7） |
| 整网梯度范数 | 差 0.12% | 350.69543 vs 350.69645（3e-6） | 127.40212 vs 127.40607（3e-5） |
| TAL 正样本数 | 90 / 90 | 90 / 90（one-to-one 那路不适用） | o2m 90/90，o2o 9/9 |
| TAL 目标分之和 | 逐点相同 | 57.850769 vs 57.85079 | o2o 7.702569 vs 7.702569 |
| TAL 正样本位置 | 90 个逐点相同 | 前 40 个逐点相同 | o2o 9 个逐点相同 |

**分项对不上是正常的，别当成 bug**：本实现报的分项是**没乘 gain** 的原始值，官方报的是乘过
`box=7.5 / cls=0.5 / dfl=1.5` 的。乘回去就逐项吻合：

| 分项 | v11 本实现 × gain | 官方 | v26 o2m 本实现 × gain | 官方 | v26 o2o 本实现 × gain | 官方 |
|---|---|---|---|---|---|---|
| box | 0.088324 × 7.5 = 0.66243 | 0.6624333 | 0.13275301 × 7.5 = 0.9956476 | 0.9956467 | 0.14128037 × 7.5 = 1.0596028 | 1.0596031 |
| cls | 7.0493765 × 0.5 = 3.524688 | 3.5246878 | 7.419924 × 0.5 = 3.709962 | 3.7099619 | 8.763583 × 0.5 = 4.381791 | 4.3817911 |
| dfl | 0.5848458 × 1.5 = 0.877269 | 0.8772685 | 0.006831013 × 1.5 = 0.0102465 | 0.0102465 | 0.006645188 × 1.5 = 0.0099678 | 0.0099678 |

为了让三家都对上，这一轮改掉了五处（前三处是真正的语义偏差，第四处是「选 best」的口径，
第五处只有训练曲线能暴露）：

1. **冲突锚点必须丢弃**（`TaskAlignedAssigner`）。一个锚点被多个 GT 的 top-k 同时选中时，官方是
   `count_tensor > 1 → mask 0`，直接变成负样本；本实现原来是「归给分数更高的那个 GT」。目标越密集
   偏得越厉害（同一个锚点被反复复用，正样本个数和分数和都对不上），已按官方改成丢弃，
   `YoloTalAssignerTests.Conflicting_Anchor_Is_Dropped_Not_Stolen` 守着这条。
2. **`pos_overlaps` 只在 `mask_pos` 内取 max**（同上）。官方是 `(overlaps * mask_pos).amax(-1)`，
   本实现原来在整张网格上取 max。v8 / v11 的 top-k 候选多，两种写法几乎等价；v26 的 one-to-one
   分支 `topk2=1` 只剩一个候选，整网格 max 会把归一化系数抬高约 4%，目标分和前景个数直接对不上。
3. **v26 双头权重按官方调度衰减**（`YoloEndToEndDetectionLoss`）。one-to-many 0.8 → 0.1、
   one-to-one 0.2 → 0.9，`decay(x) = max(1 - x/(epochs-1), 0) * (0.8 - 0.1) + 0.1`，每个 epoch 末
   更新一次（走 `IEpochAwareLoss`，`Trainer` 在 epoch 边界调，和日志节流无关）；one-to-one 分支用
   `tal_topk=7, tal_topk2=1`。原来是 1:1 直接相加、`topK=1`。
4. **选 best 用官方 fitness**。ultralytics 的 `DetMetrics.fitness = 0.1 * mAP50 + 0.9 * mAP50-95`，
   本实现原来只比 mAP50。6 张验证集上 mAP50 第一轮就饱和到 1.0，按 mAP50 选会把早期那轮
   （mAP50-95 只有 0.85）当成 best 导出。现在 `train.json` 里也多了 `bestMap50_95` 字段，
   而且是**best 那一轮**的值，不是最后一轮。
5. **AdamW 的衰减分组对齐**（`BuildOptimizer` + `TrainerOptions.DecayOnlyParameters`）。
   官方 `build_optimizer` 分三组：conv/linear 权重吃 5e-4，BN/LN 权重和所有 bias 的
   weight_decay 是 **0**。本实现原来是单组 AdamW，全部参数一起吃衰减——数值对拍查不出来
   （前向/loss/梯度都一样），但训练动力学不同：yolo11n 因此系统性低 0.06，改完立刻回到官方区间
   （见下面实测表）。TorchSharp 的 `ParamGroup` 不支持逐组 `weight_decay`，所以优化器按
   `weight_decay: 0` 建，衰减由 `Trainer` 在每步更新后补 `p *= 1 - lr·wd`
   （与 `torch.optim.AdamW` 内部逐位等价），只作用于 `named_parameters()` 里
   「名字不含 `bias` 且维度 ≥ 2」的那些参数。想退回旧行为：环境变量 `LABEL2TRAIN_WD_LEGACY=1`。
   `TrainerDecoupledWeightDecayTests` 钉住「只有配置的参数被乘、其余一动不动」。

### 1. 损失与正样本分配

改 `YoloLossOptions`（定义在 `src/Label2Train.Train/Training/Yolo/YoloDetectionLoss.cs` 顶部）：

| 参数 | 默认 | 作用 | 调大 / 调小的后果 |
|---|---|---|---|
| `BoxWeight` | 7.5 | 定位损失（CIoU）权重 | 调大→框更准但分类学得慢，小目标易抖动；调小→框松散 |
| `ClassWeight` | 0.5 | 分类（BCE）权重 | 类别多 / 目标小时调大到 1.0 左右；调太大易出现高置信度误检 |
| `DflWeight` | 1.5 | DFL 权重（`reg_max=1` 时退化成 L1，权重照用） | 调大→边界回归更细但更慢收敛 |
| `TopK` | **10** | TAL 每个 GT 取 top-k 个候选锚点 | 调大→正样本变多、召回高但定位变糙；调小→正样本稀疏，小目标容易整轮没有正样本。官方 8.4 的 `tal_topk` 就是 10（8.0/8.1 才是 13） |
| `Alpha` | 0.5 | 分配指标里分类分数的指数 | 调大→更偏向「分类置信度高」的锚点 |
| `Beta` | 6 | 分配指标里 IoU 的指数 | 调大→更偏向「框贴得准」的锚点，小目标更吃这个 |
| `ScaleByBatchSize` | `true` | loss 是否乘 batch size（官方行为） | 关掉只是让日志里的 loss 数值小一倍，对梯度方向无影响，不建议改 |

修改方式（`src/Label2Train.Train/Runners/YoloDatasetTrainingRunner.cs`，构造损失的调用处）：

```csharp
// 原样：全部走默认值（= 官方口径）
ILoss loss = YoloEndToEndDetectionLoss.CreateFor(
    model, options.ImageSize, options.ImageSize, classCount, trackComponents: true);

// 改法：传一份自定义 options，只覆盖要动的项
ILoss loss = YoloEndToEndDetectionLoss.CreateFor(
    model,
    options.ImageSize,
    options.ImageSize,
    classCount,
    new YoloLossOptions { TopK = 13, ClassWeight = 1.0f },
    trackComponents: true);
```

分配阶段用的相似度是 **CIoU 并 clamp 到 0**（`TaskAlignedAssigner`），这一点写死不可配——
官方 `iou_calculation` 就是这么做的，换成普通 IoU 会让分配结果偏移（正样本个数和梯度都对不上）。

### 2. 数据增强

改 `YoloPreprocessOptions.TrainDefault`（`src/Label2Train.Train/Data/Images/YoloPreprocessOptions.cs`）：

| 参数 | 默认 | 作用 | 调整建议 |
|---|---|---|---|
| `HorizontalFlipProbability` | 0.5 | 水平翻转 | 目标有明确左右语义（如左/右旋钮）时降到 0.2 或关掉 |
| `MosaicProbability` | 0.5 | 4 图拼贴 | 小数据集的主力增强；目标普遍很大（占半张图）时降到 0.2，否则贴完只剩半截 |
| `ScaleGain` | 0.5 | 随机缩放 ±50% | 同一相机固定距离拍摄时降到 0.2~0.3 |
| `TranslateGain` | 0.1 | 随机平移 ±10% | 一般不用动 |
| `HueGain` / `SaturationGain` / `ValueGain` | 0.015 / 0.7 / 0.4 | HSV 扰动 | 光照变化大的真实照片可保持；纯合成渲染图可降到 0 |
| `PadValue` | 114 | letterbox 填充灰度 | 不要改，导出端按 114 对齐 |

验证与推理走 `InferenceDefault`（增强全 0）。**验证集开增强会让 mAP 失去可比性**，
这一项不要动。只想关掉某一项增强时，用 `with` 表达式覆盖，不要改 `TrainDefault` 的默认值：

```csharp
YoloPreprocessOptions augment = YoloPreprocessOptions.TrainDefault(options.ImageSize)
    with { MosaicProbability = 0f };
```

### 3. 优化器与训练超参

`YoloDatasetTrainingRunner` 里写死，或在 `TrainingRunOptions`（`src/Label2Train.Core/Simulation/DatasetTrainingRunner.cs`）
里从界面 / CLI 传入：

| 项 | 默认 | 位置 | 说明 |
|---|---|---|---|
| 优化器 | AdamW | runner 内 | 官方 YOLO 默认是 SGD(lr0=0.01)，这里统一用 AdamW，**不要照搬 SGD 的 0.01** |
| 初始学习率 | 0.002 | `TrainingRunOptions.LearningRate` | AdamW 常用 1e-3 ~ 2e-3；> 5e-3 在小数据上容易发散 |
| 权重衰减 | 5e-4，只作用于 conv/linear 权重 | runner 内 `BuildOptimizer` + `TrainerOptions.DecayOnlyParameters` | 与官方 `build_optimizer` 的分组一致：BN/LN 权重与 bias 的衰减为 0。`LABEL2TRAIN_WD_LEGACY=1` 可退回「全部参数一起吃衰减」的旧行为 |
| warmup | 3 个 epoch | runner 内 | 前 3 轮从 0 线性爬升，之后进余弦段 |
| 学习率调度 | 余弦退火到 `lr × 0.01` | runner 内 | 末段 1% 收尾，避免最后几轮来回摆 |
| 梯度裁剪 | 范数 10 | runner 内 | 与官方一致 |
| EMA 衰减 | `clamp(1 - 3/总步数, 0.9, 0.999)` | runner 内 `EmaDecayFor` | 按步数自适应：小数据集几百步也能得到有意义的平均权重，不会停在初始值 |
| `Epochs` / `BatchSize` / `ImageSize` / `Scale` / `Family` | 100 / 8 / 640 / n / v8 | `TrainingRunOptions` 主构造参数 | 界面「训练目标设置」里填的就是这些 |
| 评估阈值 | conf 0.001、NMS IoU 0.7 | `YoloTrainingOptions` | 与官方 val 默认一致；改了就和其他结果不可比 |

`Scale` 与 `Family` 决定建哪个网：`family=v8` 建 `YoloV8DetectionModel`（C2f + SPPF + 三档 Detect），
`family=v11` 建 `YoloV11DetectionModel`（C3k2 + C2PSA），两者共用上面这套 loss / 增强 / 优化器参数。
`family=v26` 建 `YoloV26DetectionModel`，走双头（one-to-many + one-to-one）的 `YoloEndToEndDetectionLoss`，
权重按 0.8→0.1 / 0.2→0.9 衰减，见上面「对齐核对结果」第 3 条。

### 实测参考（20 epoch、batch 4、imgsz 320、24 张训练 / 6 张验证）

两边都是从官方 COCO 权重热启动（`yolov8n.pt` / `yolo11n.pt` / `yolo26n.pt`，分类头按 3 类重建）。
官方侧不是照 ultralytics 默认超参跑：一个一次性的 ultralytics 脚本（未入库）把超参按本实现的默认值改齐
（AdamW、lr0 0.002、lrf 0.01、cos、warmup 3 epoch、`nbs=batch` 让 ultralytics 不放大 wd、
box 7.5 / cls 0.5 / dfl 1.5、mosaic 0.5 且 `close_mosaic=0`、fliplr 0.5、translate 0.1、scale 0.5、
hsv 0.015/0.7/0.4，其余增强全关），再用 raw 权重（不是 EMA）测 mAP。

| mAP50-95 | 本实现 seed 20261124 | seed 20261125 | seed 20261126 | 均值 | 官方 seed 0 | seed 1 | 均值 |
|---|---|---|---|---|---|---|---|
| yolov8n | 0.9500 | 0.9458 | 0.9875 | 0.9611 | 0.9840 | 0.9075 | 0.9458 |
| yolo11n | 0.9458 | 0.9229 | 0.9667 | 0.9451 | 0.9637 | 0.9564 | 0.9601 |
| yolo26n | 0.9722 | 0.9250 | 0.9083 | 0.9352 | 0.9720 | 0.9371 | 0.9546 |

mAP50 三家全是 1.000（本实现）/ 0.995（官方），已经饱和，看不出口径差别，所以只看 mAP50-95。

**怎么读这张表**：6 张验证图，单张图框偏一点就够抖 0.03~0.06。本实现三个 seed 自身摆幅
0.042（v8）/ 0.044（v11）/ 0.064（v26），官方两个 seed 是 0.077 / 0.007 / 0.035。三家均值差
`+0.015 / -0.015 / -0.019`，**全部落在彼此的自摆幅之内**，没有系统性偏差。

修衰减分组前后的对照（同一批 seed，只有这一处改动）：

| 模型 | 修前 v11 均值 | 修后 v11 均值 | 官方均值 |
|---|---|---|---|
| yolo11n | 0.8986（0.9028 / 0.8944） | 0.9451（+0.047） | 0.9601 |

也就是说 v11 原来那 0.06 的缺口，主体就是「BN/LN 和 bias 也在吃 5e-4 衰减」。

### 已知差异（对齐后还剩的）

1. **增强的随机流不同**。mosaic / 裁剪采样用的是各自的随机数序列，同 seed 也不会逐 step 一致，
   所以 loss 曲线逐轮对比没有意义，只能比量级和收敛速度。表里 0.04~0.06 的自摆幅主要来自这里
   （再加上验证集只有 6 张图）。
2. **验证口径**：官方 `val` 默认用 EMA 权重，本实现导出的 best 是 raw 权重（best 按 fitness 选）。
   上表的官方数字已经是 raw，两者口径才可比——拿官方的 EMA 数字来比会平白多出 0.01~0.03。
3. 训练/验证都跑 CPU、单线程 DataLoader（官方 `workers=0`），吞吐不具可比性，这里只比精度。

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

界面上只暴露 epoch / batch / imgsz / queries，学习率与批次顺序由运行器决定
（`src/Label2Train.Train/Runners/DetrDatasetTrainingRunner.cs` 与 `Sessions/DetrTrainingSession.Run.cs`）。
训练页右上角还有一个「**参数说明**」按钮：一点就列出这次训练所有参数的当前值、作用与推荐范围（只读，改值仍走「训练目标设置」）。

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
| queries | 下拉 **50 / 100 / 150 / 300**，缺省 50，允许范围 1~300 | 一对一匹配下它同时是正样本上限与检测上限；越界（如手改代码填 400）会夹到 300 并在日志里打印 `query : 400 → 300`。详见「queries：怎么选、影响什么」 |

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
- **loss 上下震荡、不降反升**才是有问题：先降 lr（1e-4 → 5e-5；lr 不在界面上，改
  `DetrDatasetTrainingRunner` 里那行三元表达式），再看数据量与标签质量。
- 想更早确认「确实在学」，可用低阈值导出 ONNX 看预测框有没有贴住目标，比等 mAP 更直观。

### 结论与推荐配方（怎么训）

**一句话结论**：实现是对的。修完参考点语义 bug 后，8 张图过拟合 **3 轮就有 mAP50**，2000 步时 L1 0.002 / GIoU 0.008，
`class_embed` 的权重一直在变（梯度范数 2~7）；剩下的「慢」是 DETR 一对一匹配的分类特性——query 多、正样本少，
未匹配 query 的框又不受监督，分类分数只能慢慢爬。详见下面「分类为什么慢」一节。

**先确认权重，再谈参数**：`data/weights/` 下要有 `rf-detr-medium.safetensors`（或用 `LABEL2TRAIN_RFDETR_WEIGHTS` 指定），
训练页「模型」那一行会显示文件名。显示「随机初始化」时跑的是 lr 1e-3 的从零训练：mAP 长期贴 0，
而且同一批对照里 **lr 1e-3 的 8 组有 3 组直接冻结**（loss 恒定）。这时调别的参数都是白费。

| 参数 | 建议 | 依据 |
|---|---|---|
| epochs | 几百张 → **50 起**；≥1000 张 → 80~100 | 640 张 20 轮才到 0.86~0.93 mAP50，还在涨 |
| batch | 8 GB 卡 **2**；≥12 GB 用 4 | batch 4 / 224 已占 7885 MiB，贴上限后单步慢 30 倍 |
| imgsz | 小目标、密集场景 336/448；其余 **224** | 必须是 112 的倍数，非法值会被就近纠正 |
| 学习率 | 界面不暴露，自动 | 有官方权重 1e-4；无权重 1e-3 |
| 数据增强 | 真实照片**开**；程序渲染 / 背景一致的合成数据**关** | 见下面「数据增强：什么时候开、什么时候关」 |
| queries | 每图 1~30 个目标 → **50**；30~50 → 100；50~150 → 150；>150 → 300 | 它同时是召回上限，见「queries：怎么选、影响什么」 |
| 验证集 | 至少留 20%（导出时按文件名自动留出） | 少于 5 张图时没有 val，mAP 恒为 0 |

节奏参考：640 张 / batch 2 ≈ **29 s/轮**（显存 5999 MiB）→ 50 轮约 25 分钟。

> 弹窗里的「缺省 RF-DETR」预设给的是 **100 轮 / batch 4 / imgsz 560**——那是给 ≥12 GB 卡和 COCO 量级分辨率的。
> 8 GB 卡上套用它之后要手动改：batch 降到 2、imgsz 降到 224 或 336，否则很可能直接 OOM，
> 或者贴到显存上限后慢 30 倍（见上面的速度注）。

### 数据增强：什么时候开、什么时候关

**界面在哪**（三处都能确认这次到底用没用）：

- 训练页右上角「训练目标设置…」→ 弹窗里的「数据增强」勾选框（下面那行灰字就是同一条说明）。
- 主界面参数行的「aug」格子：开 / 关，训练中也一直显示。
- 训练日志第一屏：开着打印 `augment: hflip 0.5 · scale ±0.25 · translate ±0.1 · hsv 0.015,0.7,0.4`，
  关掉打印 `augment: 关闭（纯 letterbox，仅用于对照）`。事后翻日志能确认这一轮真实生效的是哪个。

三条使用须知：

- **只对 RF-DETR 生效**：模型家族选成 YOLO 时这个勾选框是灰的——YOLO 的 hflip / mosaic 在它自己的 pipeline 里，这里没有开关。
- **验证集永远不增强**：勾掉之后训练集也只做 letterbox，所以「开 / 关」比的是同一个验证集上的分数，可以直接对着看。
- **不被预设覆盖，但会随重启复位**：「缺省 RF-DETR」只重置 epochs / batch / imgsz；这个值和 epochs 一样没有持久化，
  重启回到默认「开」。每次开训前扫一眼「aug」格子。

**它具体做什么**：hflip 0.5、随机缩放 ±0.25、平移 ±0.1、HSV(0.015 / 0.7 / 0.4)，**没有 mosaic**；
增强种子按 `(epoch, 样本序号)` 派生，同一个 seed 能复现。开着等于告诉模型「目标可能左右翻转、大小和位置会变」，
关掉等于说「目标的位置、尺度、朝向都是稳定的」。

**什么时候该开着**（缺省值，先试这个）：

| 场景 | 为什么 |
|---|---|
| 真实照片：角度、光照、拍摄距离都在变 | 要学的是目标本身，不是「它在这张图里的像素坐标」 |
| 只有几十~几百张图 | 每轮看到的图都略有不同，相当于放大有效样本量，抑制背图 |
| 目标位置随机（手持、抓拍、移动相机） | 翻转 / 平移正好覆盖这种分布 |

**什么时候该关掉**：

| 场景 | 为什么 |
|---|---|
| 程序渲染 / 游戏截图 / 合成数据，背景与光照高度一致 | 增强只是把像素挪来挪去，模型要多花容量去拟合这些「假变化」，实测更差 |
| 固定机位、固定焦距的工业质检 / 监控 | 部署时目标的尺度、位置分布就是训练那一套，增强反而把输入分布拉歪 |
| 目标有强方向性（文字、有上下之分的零件、单向纹理） | hflip 会造出「反着」的样本，与真实分布冲突；要开也得在数据层面排除翻转 |
| 小目标 + 低分辨率 | 缩放 ±0.25 会把本来只有几个像素的目标直接缩没 |

**本机实测（640 张程序渲染图、官方 medium 权重）**：关掉更准，且差距远大于换种子的噪声：

| 种子 | 增强 | mAP50@e10 | @e16 | @e20 | mAP50-95@e20 |
|---|---|---|---|---|---|
| 4 | 开 | 0.614 | 0.843 | 0.856 | 0.623 |
| 4 | 关 | 0.795 | 0.920 | **0.930** | **0.767** |
| 1234 | 开 | 0.553 | 0.803 | 0.860 | 0.615 |
| 1234 | 关 | 0.632 | 0.913 | **0.929** | **0.756** |

（每组一个进程、batch 2、20 轮、建模前 `manual_seed`；同配置换种子只差 0.001~0.004。
batch 4 的旧表结论一致，见「数据增强对照实验」。）

**别把它当万能开关**：

- 关增强**不能**解决数据少的问题，只是让模型别去拟合增强噪声；数据不足时过拟合会更快
  （train loss 一路掉、val mAP 早早走平），这时该补数据而不是继续调开关。
- 上面的 0.93 **只在这个数据集上成立**：背景一致、目标外观一致。换成真实拍摄的图，默认开着更稳，别照搬数字。
- 想自己确认：同一份数据跑两组（一组一个进程、其余参数完全一致），看 **e10 的 mAP50** 谁高——
  20 轮太长时，这一步就能分出方向。

### 每图目标数的影响

queries 现在是**下拉可调**（50 / 100 / 150 / 300，见下一节），缺省 50。同一份数据下，
每图目标数决定三件事：能匹配多少正样本、负样本比多大、一批里最密的那张图要占多少显存。

| 每图目标数 | 推荐 queries | 负:正 | 会发生什么 |
|---|---|---|---|
| 1~10 | 50（默认） | 4:1 ~ 49:1 | 本仓库的默认场景，正常收敛 |
| 10~30 | 50 | 0.7:1 ~ 4:1 | 收敛更快；同图遮挡、同类叠放变多，imgsz 可提到 336 |
| 30~50 | 100 | 1:1 ~ 2.3:1 | 仍用 50 会开始漏检：超出的 GT 拿不到梯度 |
| 50~150 | 150 | 0:1 ~ 2:1 | 必须调大，否则召回天花板就是 50/GT |
| >150 | 300（官方） | 负样本最少 | 官方权重只有 300 行，再往上没有预训练可用 |

机制上就四条：

1. **一对一匹配、无 NMS**：一张图最多 min(GT, Q) 个正样本，多出来的目标在匹配里只能落到负样本上，学不到。
2. **负:正 = (Q − GT) / GT**：Q=50 时每图 1 个目标是 49:1（最慢），每图 10 个也只有 4:1；
   官方 300 个 query 在同数据上是 6 倍负样本——这就是「官方 query 数在稀疏数据上慢一个量级」的算术来源。
3. **目标张量按批内最密的那张图 padding**（`YoloBatchBuilder` 取 `maxBoxes`）：一张 80 框的图会把整批的匹配开销和显存
   都拖到 80 框的量级。所以密集数据集要**把 batch 压小**，别指望调大。
4. **推理侧低阈值 0.05、不 NMS、最多 Q 个框/图**：密集场景先确认是不是被 Q 卡住，再谈调参。

平均每图目标数在界面上能直接算：训练页数据集那一行显示「**N 张图 / M 个框**」，M/N 就是。

另外两点容易忽略：

- **小目标与密集是同一件事的两面**：letterbox 到 224 之后，密集图里的小目标可能只剩几个像素，先把 imgsz 加到 336/448 再看。
- **类别多、长尾重** 不改变正负比，但每类正样本更少，同样的轮数下尾部类别基本学不到——加轮数不如先补数据。

### queries：怎么选、影响什么

**它是什么**：decoder 里 query 的个数，每个 query 输出「一个框 + 每类分数」。所以它同时是**检测上限**
（推理时 score > 0.05、不做 NMS，一张图最多出 Q 个框）和**正样本上限**（训练时 Hungarian 一对一匹配，
一图最多匹配上 min(Q, GT) 个正样本，其余 query 全部计入负样本）。

**界面在哪**：

- 「训练目标设置…」→ `queries` 下拉，四档 50 / 100 / 150 / 300；选 YOLO 时整块隐藏（YOLO 没有 query 这个概念）。
- 主界面参数行的 `queries` 格子：只读显示当前值。
- 右上角「**参数说明**」按钮：一次性列出这次训练所有参数的当前值、作用与推荐范围（含 queries 与每图目标数的换算表）。
- 训练日志会打印 `query : 100（每图最多 100 个正样本 / 检测框）`。这个数字随权重一起存进 checkpoint，
  导出 ONNX 与检测端按它建图，**不需要在检测时再设**（见 `model.json` 的 `queries` 字段）。

**四条影响**：

| 影响 | 说明 |
|---|---|
| 召回上限 | 每图目标数 > Q 时必然漏检，而且漏掉的那些连梯度都没有；上限 = Q / GT |
| 分类头收敛速度 | 负:正 = (Q − GT)/GT。稀疏数据上 Q 越大越慢：300 对 4~5 个目标是 65:1，50 只有 10:1 |
| 显存与耗时 | 基本无感。query_feat 只有 Q×256 个数，多的只是解码器里一点矩阵乘；瓶颈始终在 ViT 主干（CPU 448×448 每步 2.55 s 就是它）。没有单独拿 Q=50 与 300 对跑计时，别把它当调参杠杆 |
| 与官方权重的关系 | Q ≤ 300 时从官方 `query_feat` / `refpoint_embed` 切前 Q 行；Q=300 是全量；>300 会被拦下（官方只给了 300 行，多出来的是随机初始化，等于没有预训练） |

**怎么选（三步）**：

1. 算平均每图框数：数据集行的「M 个框 / N 张图」，M/N 就是。
2. 查上一节的表选档位；落在两档之间**取小的那个**，先别一步跳到 300。
3. 开训后确认日志里的 `query : N`，跑 10~20 轮看两件事：mAP50 是不是比之前更早离开 0、密集图上的漏检是不是少了。

**代价与坑**：

- **调小（→50）**：检测与召回上限被压到 50/图，每图 60 个目标的数据一定有 10 个永远拿不到梯度；
  官方 300 行 query 里也只用上前 50 行。稀疏数据上这点损失远小于收益，密集数据上就是硬伤。
- **调大（→100/150/300）**：负样本变多、分类头收敛更慢，小数据集上可能几十轮 mAP 都贴着 0；
  每图目标数没到那个量级时，调大只会变慢。**先确认数据真的那么密再调。**
- **换 queries ≠ 换坏已有模型**：queries 记在 checkpoint 里，改下拉只影响下一次训练；
  已训练好的权重、已导出的 ONNX 不受影响。

**证据强度**（这一节别把推荐当成实测结论）：

| 说法 | 证据 |
|---|---|
| 一图目标数 > Q 必然漏检，漏掉的目标拿不到梯度 | 结构决定（一对一匹配 + 最多输出 Q 个框），与数据无关 |
| Q=50 的起始分类损失比 Q=300 低 6 倍（30.87 → 9.36），302 个张量正常落位 | 实测（640 张、imgsz 224、3 轮） |
| Q=300 与 Q=50 **谁最终 mAP 更高** | **没有结论**：3 轮内两者 mAP 都贴着 0，当时前向还有参考点 bug，那批数据不能用来比较 |
| 100 / 150 两档更合适 | 未实测，按上面的负:正比外推。数据每图 20~40 个目标时，建议自己跑一组 50 vs 100（一组一个进程、其余参数一致），用 e10 的 mAP50 判方向 |

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
强制覆盖成 300，界面上写的和实际用的并不一致）。现在它从「训练目标设置 → queries 下拉」
经 `TrainingTarget.Queries → TrainViewModel.Queries → TrainingRunOptions.Queries → DetrTrainingOptions.Queries`
一路传下去，越界值夹到 1~300，开训日志打印 `query : N` 可核对。

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
> 按「一组一个进程」重跑过的结果见下一节，结论一致。

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

**严格配对复跑（一组一个进程 · batch 2 · 20 轮）**：

上面的表有「同进程连跑、初始化会漂移」的瑕疵，而且 batch 4 在这台 8 GB 卡上会贴显存上限
（见前面的速度注）。按「一组一个进程、建模型前 `manual_seed`、官方 medium 权重、batch 2、
其余配置不变」重跑了四组，汇总（每组 6400 步，是 batch 4 那批的两倍步数）：

| 种子 | 增强 | mAP50@e10 | @e16 | @e20 | mAP50-95@e20 |
|---|---|---|---|---|---|
| 4 | 开 | 0.614 | 0.843 | 0.856 | 0.623 |
| 4 | 关 | 0.795 | 0.920 | **0.930** | **0.767** |
| 1234 | 开 | 0.553 | 0.803 | 0.860 | 0.615 |
| 1234 | 关 | 0.632 | 0.913 | **0.929** | **0.756** |

- 方向与 batch 4 的结论一致：**关增强更好，mAP50 差 0.069~0.074、mAP50-95 差 0.141~0.144**。
- 严格配对后**种子噪声几乎消失**：同配置换种子只差 0.001~0.004（0.856 vs 0.860、0.930 vs 0.929），
  比增强带来的差距小一个数量级，所以这个结论不受初始化漂移影响。
- 绝对值和 batch 4 那张表不能直接比：batch 2 的步数是两倍（6400 vs 3200），e20 分数自然更高。

**怎么用**：训练页「训练目标设置」里有一个「数据增强」勾选框（只在选到 RF-DETR 时可用，
YOLO 的 hflip / mosaic 在它自己的 pipeline 里）；值从 `TrainingTarget` → `TrainViewModel.Augment`
→ `TrainingRunOptions.Augment` → `DetrTrainingOptions.Augment` 一路传下去；开训时会打印
`augment: hflip ... hsv ...` 或 `augment: 关闭（纯 letterbox，仅用于对照）`，事后翻日志能确认这一轮到底用没用。

**速度**：显存够时 batch 4 是 0.10 s/步（约 16 s/轮、20 轮约 5 分钟）；8 GB 卡上 batch 4 会贴到
显存上限并降速到 3.21 s/步，实测改用 batch 2 后 29 s/轮、显存 5999 MiB，20 轮 10~11 分钟。

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
