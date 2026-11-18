using Avalonia;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;

[assembly: AvaloniaTestApplication(typeof(Label2Train.App.Tests.TestAppBuilder))]

// 语言、数据目录、BeanFactory 都是进程级静态状态，测试之间别并行
[assembly: Xunit.CollectionBehavior(DisableTestParallelization = true)]

namespace Label2Train.App.Tests;

/// <summary>
/// headless 测试的应用引导：走真正的 <see cref="Label2Train.App.App"/>，
/// 主题、样式、资源都和线上一致；因为 headless 没有桌面生命周期，
/// App 里那套容器初始化会被跳过，容器由 <see cref="AppTestHost"/> 按测试自己搭。
/// </summary>
public static class TestAppBuilder
{
    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<Label2Train.App.App>()
            .UseHeadless(new AvaloniaHeadlessPlatformOptions
            {
                // 不要像素：headless 排版/命中测试足够，也不需要 GPU 或 Skia
                UseHeadlessDrawing = true,
            });
}
