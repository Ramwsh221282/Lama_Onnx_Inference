namespace Lama_Onnx_Inference;

internal class Engine : IEngine
{    
    private readonly InferenceSession _session;
    private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _outputTensor;
    private Input _input;
    private Mat _processedImage;

    public Engine() => _session = new InferenceSession("lama_fp32.onnx");

    public void SetInput(string imageFilePath, string maskFilePath) => _input = new Input(imageFilePath, maskFilePath);

    public void ProcessImage()
    {
        _outputTensor = _session.Run(_input.CreateInputContainer());
        _processedImage = new OutputPostProcessor(new Output(_outputTensor)).PostProcessImage();
    }    

    public void ResizeImageToDefault()
    {                
        Size size = new Size(_input.DefaultWidth, _input.DefaultHeigth);        
        Cv2.Resize(_processedImage, _processedImage, size, 0, 0, InterpolationFlags.Cubic);           
    }

    public void SaveImage(string path)
    {
        Cv2.ImWrite(path, _processedImage);        
        _processedImage.Dispose();
    }

    public void Dispose() => _session.Dispose();
}
