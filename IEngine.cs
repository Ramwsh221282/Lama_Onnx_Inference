namespace Lama_Onnx_Inference;

public interface IEngine : IDisposable
{
    void SetInput(string imageFilePath, string maskFilePath);
    void ProcessImage();
    void ResizeImageToDefault();
    void SaveImage(string path);
}
