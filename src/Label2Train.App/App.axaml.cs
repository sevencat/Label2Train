using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Localization;
using Label2Train.Label;
using Label2Train.Train;
using Label2Train.App.ViewModels;
using Label2Train.App.Views;

namespace Label2Train.App;

public partial class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        // headless 测试（Avalonia.Headless）没有桌面生命周期：数据目录、容器、主窗口
        // 都交给测试自己安排；生产的桌面入口永远走下面的分支
        if (ApplicationLifetime is not IClassicDesktopStyleApplicationLifetime desktop)
        {
            base.OnFrameworkInitializationCompleted();
            return;
        }

        StartupOptions startup = StartupOptions.Parse(desktop.Args);

        // 数据目录（库、日志都落这里）先定，再打开库：文件不在就当场建空库
        AppDataPaths.Initialize(startup.DataDirectory);

        ServiceProvider services = new ServiceCollection()
            // 参数已经解析过，直接用这一份注册：窗口与启动逻辑看到的是同一个对象，
            // 它同时就是「库在哪」（IDatabaseLocation），开库时直接拿它
            .AddSingleton(startup)
            .AddLabel2TrainDatabase(startup)
            .AddLabel2TrainCore("label")
            .AddLabel2TrainLabel()
            // 训练引擎：替换 Core 里那个只出摘要的假 runner（必须在 Core 之后）
            .AddLabel2TrainTrainingEngine()
            // Injectio 按程序集生成：MainWindowViewModel / MainWindow 在特性里注册
            .AddLabel2TrainApp()
            .BuildServiceProvider();

        // 逃生舱口：少数拿不到构造函数注入的地方（XAML 预览、转换器回调）可以从这里偶尔取一次
        BeanFactory.Initialize(services);

        // 语言：设置里存 zh/en，空值跟随系统
        ApplyLanguage(services.GetRequiredService<AppSettings>());

        ILogger<App> logger = services.GetRequiredService<ILoggerFactory>().CreateLogger<App>();

        logger.LogInformation(
            "标注应用启动，数据目录：{Directory}，数据库：{Database}",
            AppDataPaths.DataDirectory,
            startup.DatabasePath);

        desktop.MainWindow = services.GetRequiredService<MainWindow>();

        desktop.Exit += (_, _) =>
        {
            logger.LogInformation("标注应用退出");

            // 容器的 DisposeAsync 会顺带关掉库；窗口已经关了，
            // 这里只等收尾（超时就不再等，进程反正要退）
            try
            {
                Task.Run(async () => await services.DisposeAsync()).Wait(TimeSpan.FromSeconds(5));
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "释放服务容器时报错");
            }
            finally
            {
                // 容器已经释放，再留着引用只会让后续解析调到已释放的 provider
                BeanFactory.Reset();
            }
        };

        base.OnFrameworkInitializationCompleted();
    }

    private static void ApplyLanguage(AppSettings settings) =>
        Localizer.Instance.SetLanguage(
            string.IsNullOrWhiteSpace(settings.Language)
                ? Localizer.DetectSystemLanguage()
                : settings.Language);
}
