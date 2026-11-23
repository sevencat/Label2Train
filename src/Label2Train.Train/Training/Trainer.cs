using System.Diagnostics;
using Label2Train.Train.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Label2Train.Train.Training;

/// <summary>训练超参数。</summary>
public sealed record TrainerOptions
{
    /// <summary>训练轮数。</summary>
    public int Epochs { get; init; } = 100;

    /// <summary>梯度全局范数裁剪阈值（&lt;= 0 表示不裁剪）。</summary>
    public float GradientClipNorm { get; init; } = 5f;

    /// <summary>每个 epoch 是否打乱批次顺序。</summary>
    public bool Shuffle { get; init; } = true;

    /// <summary>每多少个 epoch 回调一次（0 表示每个 epoch 都回调）。</summary>
    public int LogEvery { get; init; }

    /// <summary>起始 epoch（断点续训用；epoch 编号会从该值开始递增）。</summary>
    public int StartEpoch { get; init; }

    /// <summary>起始优化步（断点续训用；影响学习率调度的绝对位置）。</summary>
    public int StartStep { get; init; }

    /// <summary>
    /// 按参数名给出梯度倍率（用于给随机初始化的部分单独提高学习率）；null 表示全部 1。
    /// 见 <see cref="Trainer"/> 里 <c>ScaleGradients</c> 的说明。
    /// </summary>
    public Func<string, float>? GradientScale { get; init; }
}

/// <summary>单个 epoch 结束时的指标。</summary>
public sealed record EpochMetrics(int Epoch, int Step, float Loss, float LearningRate);

/// <summary>
/// 训练循环：前向 → 损失 → 反向 → 梯度裁剪 → 优化器更新。
/// <para>
/// 自动微分、优化器、梯度全部由 TorchSharp / LibTorch 负责；
/// 本项目只编排训练流程、日志、学习率调度与检查点。
/// </para>
/// </summary>
public sealed class Trainer
{
    private readonly Module<Tensor, Tensor> _model;
    private readonly OptimizerHelper _optimizer;
    private readonly ILoss _loss;

    /// <summary>
    /// 统一损失入口：若模型实现了 <see cref="ILossExtrasProvider"/>，把附加张量一并传给损失
    /// （损失的 extras 重载默认会忽略它们，只有需要深监督的损失才真正使用）。
    /// </summary>
    private Tensor ComputeLoss(Tensor prediction, Tensor target)
    {
        IReadOnlyDictionary<string, Tensor>? extras =
            (_model as ILossExtrasProvider)?.GetLossExtras();

        return extras is null
            ? _loss.Compute(prediction, target)
            : _loss.Compute(prediction, target, extras);
    }
    private readonly TrainerOptions _options;
    private readonly Rng _rng;
    private readonly Ema? _ema;

    /// <summary>
    /// 需要放大/缩小梯度的参数（<see cref="TrainerOptions.GradientScale"/> 的落地形式）。
    /// 构造时抓一次，避免每步遍历整张网络。
    /// </summary>
    private readonly (Parameter Parameter, float Scale)[] _gradientScales;

    public Trainer(
        Module<Tensor, Tensor> model,
        OptimizerHelper optimizer,
        ILoss loss,
        TrainerOptions? options = null,
        Rng? rng = null,
        Ema? ema = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(loss);

        _model = model;
        _optimizer = optimizer;
        _loss = loss;
        _options = options ?? new TrainerOptions();
        _rng = rng ?? new Rng(1);
        _ema = ema;
        Step = _options.StartStep;

        Func<string, float>? gradientScale = _options.GradientScale;

        _gradientScales = gradientScale is null
            ? Array.Empty<(Parameter, float)>()
            : _model.named_parameters()
                .Select(entry => (Parameter: entry.parameter, Scale: gradientScale(entry.name)))
                .Where(entry => entry.Scale != 1f)
                .ToArray();

        if (_options.Epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), _options.Epochs, "Epochs 必须大于 0");
    }

    /// <summary>已经执行的优化步数。</summary>
    public int Step { get; private set; }

    /// <summary>最近一个 epoch 的平均损失。</summary>
    public float LastLoss { get; private set; } = float.NaN;

    /// <summary>最近一个 epoch 里「等批次数据（解码 / letterbox / 增强 + 上传）」花掉的秒数。</summary>
    public double LastDataSeconds { get; private set; }

    /// <summary>最近一个 epoch 里「前向 + 损失 + 反向 + 优化器」花掉的秒数。</summary>
    public double LastComputeSeconds { get; private set; }

    /// <summary>最近一个 epoch 实际跑了多少个优化步；除以它才是每步耗时。</summary>
    public int LastBatchCount { get; private set; }

    /// <summary>
    /// 最近一个 epoch 里每步梯度全局范数的平均值（裁剪前）。
    /// <para>
    /// 恒为 0 是「模型已经学不动了」的硬证据：参数被推到 sigmoid 的饱和端之后
    /// 梯度精确为 0，loss 会一直停在原处，靠降低 lr 也救不回来，只能停训换配置。
    /// 比「loss 几轮没变」更准——冻结时 loss 仍会有百分之几的抖动。
    /// </para>
    /// </summary>
    public double LastGradientNorm { get; private set; }

    private double _epochGradNormSum;

    private int _epochGradSteps;

    /// <summary>
    /// 训练多个 epoch，返回每个 epoch 的指标。
    /// <para>批次集合固定；<see cref="TrainerOptions.Shuffle"/> 决定每个 epoch 是否重排。</para>
    /// </summary>
    public IReadOnlyList<EpochMetrics> Fit(
        IReadOnlyList<Batch> batches,
        ILearningRateSchedule? schedule = null,
        Action<EpochMetrics>? onEpoch = null,
        TrainControl? control = null,
        IProgress<TrainProgress>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(batches);
        if (batches.Count == 0)
            throw new ArgumentException("训练批次为空。", nameof(batches));

        int[] order = Enumerable.Range(0, batches.Count).ToArray();

        return Fit(
            _ =>
            {
                if (_options.Shuffle)
                    Shuffle(order);

                return order.Select(index => batches[index]);
            },
            schedule,
            onEpoch,
            control,
            progress);
    }

    /// <summary>
    /// 训练多个 epoch，批次由委托按 epoch 惰性提供：
    /// 用于 mosaic / 多尺度训练这种「每个 epoch 重新构建批次」的场景，避免把整轮数据常驻内存。
    /// </summary>
    /// <param name="control">可选的暂停 / 取消控制；null 表示一路跑到底。</param>
    /// <param name="progress">可选的进度回调：每个 epoch 上报一次（不受 <see cref="TrainerOptions.LogEvery"/> 影响）。</param>
    /// <returns>已完成的 epoch 指标；被取消时返回的是取消前的部分历史。</returns>
    public IReadOnlyList<EpochMetrics> Fit(
        Func<int, IEnumerable<Batch>> batchesForEpoch,
        ILearningRateSchedule? schedule = null,
        Action<EpochMetrics>? onEpoch = null,
        TrainControl? control = null,
        IProgress<TrainProgress>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(batchesForEpoch);

        var history = new List<EpochMetrics>(_options.Epochs);
        int lastEpoch = _options.StartEpoch + _options.Epochs;
        bool stopped = false;

        for (int epoch = _options.StartEpoch; epoch < lastEpoch && !stopped; epoch++)
        {
            _model.train();

            var dataWatch = new Stopwatch();
            var stepWatch = new Stopwatch();
            double dataSeconds = 0;
            double computeSeconds = 0;
            double epochLoss = 0;
            int batchCount = 0;

            // AdamW 的 exp_avg / exp_avg_sq 在第一次 step 里创建，必须活在 DisposeScope 之外。
            // 从检查点继续训练时 optimizer 已有 state（Step > 0），可以直接进 scope 分支。
            bool optimizerStateReady = Step > 0;

            try
            {
                using IEnumerator<Batch> iterator = batchesForEpoch(epoch).GetEnumerator();

                while (true)
                {
                    // MoveNext 就是「准备下一批」：解码图片、letterbox、增强、拷到设备。
                    // 它和下面的前向反向在同一线程里，所以这段时间 GPU 是闲着的。
                    dataWatch.Restart();
                    if (!iterator.MoveNext())
                        break;
                    dataSeconds += dataWatch.Elapsed.TotalSeconds;

                    Batch batch = iterator.Current;
                    ArgumentNullException.ThrowIfNull(batch);

                    // 暂停/取消只在批次边界生效：不打断已开始的 forward/backward，
                    // 参数与优化器状态始终停在一致的位置，继续时无需重新加载任何东西。
                    if (control is not null && !control.WaitUntilReady())
                    {
                        stopped = true;
                        break;
                    }

                    stepWatch.Restart();

                    // 一个 step 的前向激活是 GB 级的，不主动释放就要等 GC——而 GPU 显存不够时
                    // Windows 会把显存分页到系统内存，训练从 100 ms/step 掉到 3-4 s/step，
                    // 表现就是「上了 CUDA 还是很慢」。所以整步挂在一个 DisposeScope 上。
                    if (!optimizerStateReady)
                    {
                        epochLoss += RunStep(batch, schedule);
                        optimizerStateReady = true;

                        // 首步没有 scope，激活只能靠 GC；主动收一次，别让这几 GB 常驻。
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                    }
                    else
                    {
                        using (var scope = torch.NewDisposeScope())
                        {
                            epochLoss += RunStep(batch, schedule);
                            scope.Dispose();
                        }
                    }

                    batchCount++;
                    computeSeconds += stepWatch.Elapsed.TotalSeconds;
                }
            }
            finally
            {
                LastDataSeconds = dataSeconds;
                LastComputeSeconds = computeSeconds;
                LastBatchCount = batchCount;

                // 梯度范数是「模型还在不在学」的唯一硬指标，每轮记一次平均值
                LastGradientNorm = _epochGradSteps > 0 ? _epochGradNormSum / _epochGradSteps : 0;
                _epochGradNormSum = 0;
                _epochGradSteps = 0;
            }

            if (batchCount == 0)
            {
                // 刚进新的 epoch 就按了停止 / 取消：第一批还没开始跑，属于正常收尾，
                // 不能报「没有产生任何批次」把用户吓一跳（日志里会变成一条训练失败）
                if (stopped)
                {
                    break;
                }

                throw new InvalidOperationException($"第 {epoch} 个 epoch 没有产生任何批次。");
            }

            LastLoss = (float)(epochLoss / batchCount);
            var metrics = new EpochMetrics(epoch, Step, LastLoss, GetLearningRate());
            history.Add(metrics);

            // 进度上报与日志节流解耦：UI 每轮都要刷新曲线，而控制台日志可以少打。
            progress?.Report(new TrainProgress(epoch, lastEpoch, Step, LastLoss, metrics.LearningRate));

            if (onEpoch is not null
                && (_options.LogEvery <= 0 || epoch % _options.LogEvery == 0 || epoch == lastEpoch - 1))
            {
                onEpoch(metrics);
            }
        }

        return history;
    }

    /// <summary>在给定批次上评估（不更新参数），返回平均损失。</summary>
    public float Evaluate(IReadOnlyList<Batch> batches)
    {
        ArgumentNullException.ThrowIfNull(batches);
        if (batches.Count == 0)
            throw new ArgumentException("评估批次为空。", nameof(batches));

        _model.eval();
        double total = 0;

        using (torch.no_grad())
        {
            foreach (Batch batch in batches)
            {
                // 评估也是每批几 GB 的激活，同样要按批释放，否则验证集一大就爆显存。
                using (var scope = torch.NewDisposeScope())
                {
                    Tensor prediction = _model.forward(batch.Input);
                    total += ComputeLoss(prediction, batch.Target).item<float>();
                    scope.Dispose();
                }
            }
        }

        _model.train();
        return (float)(total / batches.Count);
    }

    /// <summary>执行一个训练步：前向 / 损失 / 反向 / 梯度裁剪 / 优化器 / EMA，返回该步损失。</summary>
    private float RunStep(Batch batch, ILearningRateSchedule? schedule)
    {
        _optimizer.zero_grad();
        Tensor prediction = _model.forward(batch.Input);
        Tensor loss = ComputeLoss(prediction, batch.Target);
        loss.backward();

        _epochGradNormSum += ClipGradients();
        _epochGradSteps++;
        ScaleGradients();

        if (schedule is not null)
            SetLearningRate(schedule.GetLearningRate(Step));

        _optimizer.step();
        _ema?.Update(_model);
        Step++;

        return loss.detach().item<float>();
    }

    /// <summary>裁剪梯度，返回裁剪前的全局范数（关掉裁剪时也会算，用于诊断模型是否已经学不动）。</summary>
    private double ClipGradients()
    {
        // 平方和在设备上累加，最后只同步一次。
        // 逐参数 item<T>() 会让每个参数都触发一次 GPU→CPU 同步（YOLOv8n 有上百个参数张量），
        // 等于每个 step 上百次「等 GPU 跑完再继续」，这是 GPU 训练最典型的自我拖慢。
        Tensor? sum = null;
        foreach (Parameter parameter in _model.parameters())
        {
            if (parameter.grad is null)
                continue;

            Tensor squares = parameter.grad.detach().pow(2).sum();
            sum = sum is null ? squares : sum + squares;
        }

        if (sum is null)
            return 0;

        double norm;
        using (sum)
        {
            norm = Math.Sqrt(sum.item<float>());
        }

        if (_options.GradientClipNorm <= 0f || norm <= _options.GradientClipNorm || norm <= 0)
            return norm;

        var scale = (float)(_options.GradientClipNorm / norm);
        foreach (Parameter parameter in _model.parameters())
            parameter.grad?.mul_(scale);

        return norm;
    }

    private void SetLearningRate(float learningRate)
    {
        foreach (ParamGroup group in _optimizer.ParamGroups)
            group.LearningRate = learningRate;
    }

    /// <summary>
    /// 按 <see cref="TrainerOptions.GradientScale"/> 放大/缩小指定参数的梯度。
    /// Adam 的更新量 ≈ lr·g/|g|，梯度乘以 k 等价于学习率乘以 k，
    /// 因此可以用来给「随机初始化的头」单独提高学习率。
    /// 放在梯度裁剪之后：免得放大的头梯度把预训练部分的裁剪阈值一起拉低。
    /// </summary>
    private void ScaleGradients()
    {
        foreach ((Parameter parameter, float scale) in _gradientScales)
            parameter.grad?.mul_(scale);
    }

    private float GetLearningRate() => (float)_optimizer.ParamGroups.First().LearningRate;

    private void Shuffle(int[] order)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = _rng.NextInt(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }
}
