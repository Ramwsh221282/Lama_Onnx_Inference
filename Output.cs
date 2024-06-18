namespace Lama_Onnx_Inference;

internal class Output
{
    public float[] PixelArray { get; private set; }
    public Tensor<float> Tensor { get; private set; }

    public Output(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> tensor)
    {        
        DisposableNamedOnnxValue[] values = tensor.ToArray();
        Tensor = values[0].AsTensor<float>();
        PixelArray = Tensor.ToArray();
    }
}
