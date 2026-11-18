using Xunit;

namespace Label2Train.Label.Tests;

public class SmokeTests
{
    [Fact]
    public void ProductName_IsStable()
        => Assert.Equal("Label2Train", Label2Train.Core.Label2TrainInfo.ProductName);

    [Fact]
    public void LabelModule_IsRegistered()
        => Assert.Equal("Label", Label2Train.Label.LabelModule.Name);
}
