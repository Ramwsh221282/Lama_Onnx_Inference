namespace Lama_Onnx_Inference;

class Program
{
    static void Main()
    {                
        IEngine engine = InpaintEngineProvider.CreateInstance();

        engine.SetInput("image.png", "mask.png");
        engine.ProcessImage();
        engine.ResizeImageToDefault();
        engine.SaveImage("inpainted.png");        
        engine.Dispose();        
    }                
}