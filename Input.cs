namespace Lama_Onnx_Inference;

internal class Input
{
    private Tensor<float> _image;
    private Tensor<float> _mask;

    private readonly Mat _imageMat;
    private readonly Mat _maskMat;

    public Input(string imageFilePath, string maskFilePath)
    {        
        _imageMat = new Mat(imageFilePath);
        _maskMat = new Mat(maskFilePath);

        DefaultHeigth = _imageMat.Height;
        DefaultWidth = _imageMat.Width;

        InputPreProcessor processor = new InputPreProcessor();
        _image = processor.CreateImageTensor(_imageMat);
        _mask = processor.CreateMaskTensor(_maskMat);

        _imageMat.Dispose();
        _maskMat.Dispose(); 
    }
    
    public int DefaultHeigth { get; private set; }
    public int DefaultWidth { get; private set; }

    public List<NamedOnnxValue> CreateInputContainer()
    {
        return new List<NamedOnnxValue>()
        {
            NamedOnnxValue.CreateFromTensor("image", _image),
            NamedOnnxValue.CreateFromTensor("mask", _mask)
        };
    }    
}
