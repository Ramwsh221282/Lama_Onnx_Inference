namespace Lama_Onnx_Inference;

internal class OutputPostProcessor
{
    private readonly Output _output;
    private readonly int batchSize;
    private readonly int channels;
    private readonly int height;
    private readonly int width;

    public OutputPostProcessor(Output output)
    {
        _output = output;
        batchSize = _output.Tensor.Dimensions[0];
        channels = _output.Tensor.Dimensions[1];
        height = _output.Tensor.Dimensions[2];
        width = _output.Tensor.Dimensions[3];
    }

    public Mat PostProcessImage()
    {
        float[,,] imageData = new float[height, width, channels];
        SetPixels(imageData);
        SetColors(imageData);        
        return new Mat(height, width, MatType.CV_32FC3, imageData);        
    }

    private void SetPixels(float[,,] imageData)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int index = b * channels * height * width + c * height * width + h * width + w;
                        imageData[h, w, c] = _output.PixelArray[index];
                    }
                }
            }
        }
    }

    private void SetColors(float[,,] imageData)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int c = 0; c < channels; c++)
                {
                    imageData[h, w, c] = Math.Max(0, Math.Min(255, imageData[h, w, c]));
                }
            }
        }
    }
}
