global using OpenCvSharp;
global using Microsoft.ML.OnnxRuntime;
global using Microsoft.ML.OnnxRuntime.Tensors;

namespace Lama_Onnx_Inference;

public static class InpaintEngineProvider
{
    public static IEngine CreateInstance() => new Engine();
}
