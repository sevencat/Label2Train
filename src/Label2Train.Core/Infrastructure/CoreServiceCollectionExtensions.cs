using FreeSql;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NLog.Extensions.Logging;
using Label2Train.Core.Dataset;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure.Logging;

namespace Label2Train.Core.Infrastructure;

public static class CoreServiceCollectionExtensions
{
    /// <summary>
    /// 按给定的库位置打开（不存在就建）当前工作库，并把 <see cref="TinyLabelDatabase"/> 注入成单例（随容器释放）。
    /// 位置本身也一并注册：调用方给谁就用谁——各宿主都 <see cref="StartupOptions.Parse"/> 一份自己的参数，
    /// 测试想单独指一个库时同样造一份，Core 不再自己编一个实现。
    /// 要在 <see cref="AddLabel2TrainCore"/> 之前调用：库路径只定这一次，配置 / 账号 / 项目共用这一个连接池。
    /// </summary>
    public static IServiceCollection AddLabel2TrainDatabase(this IServiceCollection services, IDatabaseLocation location)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(location);

        TinyLabelDatabase database = TinyLabelDatabase.Open(location.DatabasePath);
        services.AddSingleton(database);
        services.AddSingleton(location);

        return services;
    }

    /// <summary>注册两个应用共用的基础设施：库、配置、日志（NLog）。桌面端与 Web 端都从这里拿。</summary>
    public static IServiceCollection AddLabel2TrainCore(this IServiceCollection services, string applicationName)
    {
        ArgumentNullException.ThrowIfNull(services);

        AppPaths.EnsureCreated();

        var buffer = new LogBuffer();
        services.AddSingleton(buffer);

        // 库单例由 AddLabel2TrainDatabase 注册，这里只是把它的 IFreeSql 也挂出来：
        // 配置、账号、项目注入同一个连接池，谁都不 new、不 Dispose
        services.AddSingleton(provider => provider.GetRequiredService<TinyLabelDatabase>().Orm);
        services.AddSingleton<SettingsStore>();
        services.AddSingleton(provider => provider.GetRequiredService<SettingsStore>().Load());

        NLogSetup.Configure(buffer, AppPaths.LogDirectory, applicationName);

        services.AddLogging(builder =>
        {
            builder.SetMinimumLevel(LogLevel.Information);
            builder.AddNLog();
        });

        services.AddLabel2TrainDatasets();

        return services;
    }
}
