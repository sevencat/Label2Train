using Avalonia;
using ImageFacade;
using ImageFacade.Backends;
using ImageFacade.SdcbSimdCodecs;
using SimdSkia.Avalonia;
using Label2Train.Core.Imaging;
using Label2Train.Core.Infrastructure;
using Sdcb.SimdCodecs.Bmp;
using Sdcb.SimdCodecs.Jpeg;
using Sdcb.SimdCodecs.Png;

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

        // 图像后端在 exe 入口注册：界面与数据管线共用同一个 SkiaSharp 后端（导入图片读尺寸也用它）
        ImageBackendBase.RegisterCodec(
            ImageFormat.Png,
            PngImageDecoder.Registration,
            PngImageEncoder.Registration);

        ImageBackendBase.RegisterCodec(
            ImageFormat.Jpeg,
            JpegImageDecoder.Registration,
            JpegImageEncoder.Registration,
            request => new JpegEncodeOptions { Quality = Math.Clamp(request.Quality, 1, 100) });

        ImageBackendBase.RegisterCodec(
            ImageFormat.Bmp,
            BmpImageDecoder.Registration,
            BmpImageEncoder.Registration,
            _ => new BmpEncodeOptions());
        ImageRuntime.Use(new SoftwareBackend());

        // --import <源目录> --output <目标目录>：把 YOLO / RF-DETR 数据集导成标注项目，不开窗口
        if (ImportCommand.TryParse(args, out ImportCommand command))
        {
            // 导入写的就是 --output 目录里的 label2train.db：库跟着它走（--db 显式指定时以 --db 为准）
            AppDataPaths.Initialize(command.DestinationPath);

            return ImportCommandRunner.RunAsync(command, StartupOptions.Parse(args))
                .GetAwaiter().GetResult();
        }

        return BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            // 渲染 + 窗口宿主都走 SimdSkia（纯托管，不再加载 Skia 原生库）
            .UseStandardRuntimePlatformSubsystem()
            .UseWin32().With(new Win32PlatformOptions()
            {
                RenderingMode = [Win32RenderingMode.Software]
            })
            .UseSimdSkia(options =>
            {
                //options.UseGpuPreferred();
            })
            .LogToTrace();
}
