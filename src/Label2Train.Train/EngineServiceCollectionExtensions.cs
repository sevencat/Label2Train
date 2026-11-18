using Microsoft.Extensions.DependencyInjection;
using Label2Train.Core.Simulation;
using Label2Train.Train.Export;
using Label2Train.Train.Runners;

namespace Label2Train.Train;

/// <summary>训练引擎的 DI 注册。</summary>
public static class EngineServiceCollectionExtensions
{
    /// <summary>
    /// 把 <see cref="IDatasetTrainingRunner"/> 换成真训练引擎。
    /// <para>必须在 <c>AddLabel2TrainCore</c> 之后调用（容器按最后注册解析单实例）。</para>
    /// </summary>
    public static IServiceCollection AddLabel2TrainTrainingEngine(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);

        services.AddSingleton<YoloDatasetTrainingRunner>();
        services.AddSingleton<DetrDatasetTrainingRunner>();
        services.AddSingleton<IDatasetTrainingRunner, DatasetTrainingRunnerDispatcher>();
        services.AddSingleton<TrainingExportService>();
        return services;
    }
}
