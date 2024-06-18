namespace Lama_Onnx_Inference;

internal class InputPreProcessor
{
    private readonly Size _size;

    public InputPreProcessor() => _size = new Size(512, 512);

    public Tensor<float> CreateImageTensor(Mat imageMat)
    {
        Mat resized = new Mat();
        Cv2.Resize(imageMat, resized, _size, 0, 0, InterpolationFlags.Cubic);
        imageMat.Dispose();
        Tensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, 512, 512 });

        for (int y = 0; y < resized.Height; y++)
        {
            for (int x = 0; x < resized.Width; x++)
            {
                tensor[0, 0, y, x] = resized.At<Vec3b>(y, x)[0] / 255.0f;
                tensor[0, 1, y, x] = resized.At<Vec3b>(y, x)[1] / 255.0f;
                tensor[0, 2, y, x] = resized.At<Vec3b>(y, x)[2] / 255.0f;
            }
        }

        return tensor;
    }

    public Tensor<float> CreateMaskTensor(Mat maskMat)
    {
        Mat resized = new Mat();
        Cv2.Resize(maskMat, resized, _size, 0, 0, InterpolationFlags.Cubic);
        maskMat.Dispose();
        Tensor<float> tensor = new DenseTensor<float>(new[] { 1, 1, 512, 512 });

        Size size = new Size(11, 11);
        Mat element = Cv2.GetStructuringElement(MorphShapes.Rect, size);
        Cv2.Dilate(resized, resized, element);
        element.Dispose();

        for (int y = 0; y < resized.Height; y++)
        {
            for (int x = 0; x < resized.Width; x++)
            {
                float v = resized.At<Vec3b>(y, x)[0];
                if (v > 127)
                    tensor[0, 0, y, x] = 1.0f;
                else
                    tensor[0, 0, y, x] = 0.0f;
            }
        }

        return tensor;
    }        
}
