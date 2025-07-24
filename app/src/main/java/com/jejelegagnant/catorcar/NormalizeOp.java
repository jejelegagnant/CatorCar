package com.jejelegagnant.catorcar;

import android.graphics.PointF;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class NormalizeOp implements ImageOperator {

    private final float mean;
    private final float std;

    public NormalizeOp(float mean, float std) {
        this.mean = mean;
        this.std = std;
    }

    @Override
    public TensorImage apply(TensorImage image) {
        // S'assurer que le type est bien FLOAT32
        if (image.getDataType() != DataType.FLOAT32) {
            image = TensorImage.createFrom(image, DataType.FLOAT32);
        }

        float[] pixels = image.getTensorBuffer().getFloatArray();
        float[] normalizedPixels = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            normalizedPixels[i] = (pixels[i] - mean) / std;
        }

        // Recharger le buffer normalisé
        TensorBuffer normalizedBuffer = TensorBuffer.createFixedSize(
                image.getTensorBuffer().getShape(), DataType.FLOAT32);
        normalizedBuffer.loadArray(normalizedPixels);

        image.load(normalizedBuffer);
        return image;
    }

    @Override
    public int getOutputImageHeight(int inputHeight, int inputWidth) {
        return inputHeight;
    }

    @Override
    public int getOutputImageWidth(int inputHeight, int inputWidth) {
        return inputWidth;
    }

    @Override
    public PointF inverseTransform(PointF point, int inputImageHeight, int inputImageWidth) {
        return point;
    }
}
