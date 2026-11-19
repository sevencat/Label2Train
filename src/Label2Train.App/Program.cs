using Avalonia;
using ImageFacade;
using ImageFacade.SkiaSharp;
using Label2Train.Core.Imaging;
using Label2Train.Core.Infrastructure;
using SimdSkia.Avalonia;

namespace Label2Train.App;

internal static class Program
{
    [STAThread]
    public static int Main(string[] args)
    {
        if (!PlatformGuard.EnsureSupported())
        {
            return 1;
        }

        // exe 同级的 .env 先填进环境变量：--data / --db 之外的设置（端口等）都从这里来
        AppEnvironment.LoadDotEnv();

        // --data <目录>：数据根目录（项目库 / 配置库 / 日志），默认 exe 同级 data
        AppDataPaths.Initialize(AppDataPaths.ReadDataArgument(args));
        AppDataPaths.EnsureCreated();
        
        ImageRuntime.Use(new SkiaBackend());

        // --import <源目录> --output <目标目录>：把 YOLO / RF-DETR 数据集导成标注项目，不开窗口
        if (ImportCommand.TryParse(args, out ImportCommand command))
        {
            // 导入写的就是 --output 目录里的 label2train.db：库跟着它走（--db 显式指定时以 --db 为准）
            AppDataPaths.Initialize(command.DestinationPath);

            return ImportCommandRunner.RunAsync(command, StartupOptions.Parse(args))
                .GetAwaiter().GetResult();
        }

        return BuildSimdSkiaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp()
    {
        return AppBuilder.Configure<App>()
            .UseHarfBuzz()
            .UseWin32()
            .UseSkia()
            // 走 ANGLE（EGL → D3D11）；两条引擎线的 GLES3 着色器与浮点 FBO 都依赖它。
            // 顺序即优先级：ANGLE 起不来时由 Avalonia 回落到它自己的默认渲染路径。
            .With(new Win32PlatformOptions
            {
                RenderingMode = [Win32RenderingMode.AngleEgl],
            })
            .WithInterFont()
            .LogToTrace();
    }

    public static AppBuilder BuildSimdSkiaApp()
    {
        return AppBuilder.Configure<App>()
            .UseSimdSkiaDesktop()
            .LogToTrace();
    }
}
