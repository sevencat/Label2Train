using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Label2Train.Label;

/// <summary>标注业务层的注册入口。Core 的存储/格式/导入导出链路由 <c>AddLabel2TrainCore</c> 负责。</summary>
public static class LabelServiceCollectionExtensions
{
    public static IServiceCollection AddLabel2TrainLabel(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);

        services.TryAddSingleton<LabelDatasetImportService>();
        services.TryAddSingleton<LabelProjectBrowser>();

        return services;
    }
}
