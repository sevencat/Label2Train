using Microsoft.Extensions.DependencyInjection;
using Label2Train.App.ViewModels;
using Label2Train.App.Views;
using Label2Train.Core.Dataset.TinyLabel;
using Label2Train.Core.Infrastructure;
using Label2Train.Core.Localization;
using Label2Train.Label;
using Label2Train.Train.Export;

namespace Label2Train.App.Tests;

/// <summary>
/// 一次测试用的应用宿主：临时数据目录 + 真容器（和 <see cref="Label2Train.App.App"/> 同一套注册，
/// 只是不换真训练引擎，用 Core 里的摘要 runner，省得拉 TorchSharp）。
/// 语言固定中文，断言文案才稳定。
/// </summary>
public sealed class AppTestHost : IAsyncDisposable
{
    private readonly ServiceProvider _services;
    private readonly TinyLabelDatabase _database;

    private AppTestHost(string root, StartupOptions startup, ServiceProvider services)
    {
        Root = root;
        Startup = startup;
        _services = services;
        _database = services.GetRequiredService<TinyLabelDatabase>();
    }

    /// <summary>本次测试的数据目录（临时目录，销毁时删掉）。</summary>
    public string Root { get; }

    public StartupOptions Startup { get; }

    /// <summary>主窗口 ViewModel：每个宿主一份，容器单例。</summary>
    public MainWindowViewModel ViewModel => _services.GetRequiredService<MainWindowViewModel>();

    public IServiceProvider Services => _services;

    public static AppTestHost Create()
    {
        string root = Path.Combine(Path.GetTempPath(), "label2train-app-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);

        // 数据目录、日志、设置都落临时目录：测试不碰用户真实数据
        AppDataPaths.Initialize(root);
        Localizer.Instance.SetLanguage("zh-Hans");

        var startup = StartupOptions.Parse([AppDataPaths.DataArgument, root]);

        ServiceProvider services = new ServiceCollection()
            .AddSingleton(startup)
            .AddLabel2TrainDatabase(startup)
            .AddLabel2TrainCore("label-tests")
            .AddLabel2TrainLabel()
            // 只补训练页需要的导出服务；真训练引擎（TorchSharp）不注册，IDatasetTrainingRunner
            // 用 Core 里那个摘要 runner，测试不用等 LibTorch
            .AddSingleton<TrainingExportService>()
            .AddLabel2TrainApp()
            .BuildServiceProvider();

        BeanFactory.Initialize(services);

        return new AppTestHost(root, startup, services);
    }

    /// <summary>新建主窗口（会像线上一样在 Opened 里打开上次的工作区）。</summary>
    public MainWindow CreateMainWindow() => new(ViewModel, Startup);

    public async ValueTask DisposeAsync()
    {
        // 顺序反过来：先放容器引用，再关库（库是实例注册，容器不负责释放）
        BeanFactory.Reset();
        await _services.DisposeAsync();
        _database.Dispose();

        try
        {
            Directory.Delete(Root, recursive: true);
        }
        catch (IOException)
        {
        }
    }
}
