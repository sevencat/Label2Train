using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Infrastructure.Logging;
using Xunit;

namespace Label2Train.Label.Tests;

public class LoggingTests
{
    [Fact]
    public void AddLabel2TrainCore_RoutesNLogEventsIntoBuffer()
    {
        var services = new ServiceCollection();
        services.AddLabel2TrainCore("test");
        using ServiceProvider provider = services.BuildServiceProvider();

        LogBuffer buffer = provider.GetRequiredService<LogBuffer>();
        ILogger<LoggingTests> logger = provider.GetRequiredService<ILogger<LoggingTests>>();

        logger.LogInformation("探测完成：{Count} 条", 3);

        LogEntry entry = Assert.Single(
            buffer.Snapshot(),
            item => item.Message.Contains("探测完成：3", StringComparison.Ordinal));

        Assert.Equal(LogLevel.Information, entry.Level);
        Assert.Contains(nameof(LoggingTests), entry.Category, StringComparison.Ordinal);
    }
}
