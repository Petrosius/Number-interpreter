package com.amitshekhar.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;



public class TensorFlowImageRegressionModel implements RegressionModel {

    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    //private static final float THRESHOLD = 0.1f;
    private static final float PIXEL_DEPTH = 255.0f;

    private Interpreter interpreter;
    private int INPUT_WIDTH;
    private int INPUT_HEIGHT;


    private TensorFlowImageRegressionModel() {

    }

    static RegressionModel create(AssetManager assetManager,
                             String modelPath,
                             int INPUT_HEIGHT, int INPUT_WIDTH) throws IOException {

        TensorFlowImageRegressionModel regressionModel = new TensorFlowImageRegressionModel();
        regressionModel.interpreter = new Interpreter(regressionModel.loadModelFile(assetManager, modelPath));
        regressionModel.INPUT_HEIGHT = INPUT_HEIGHT;
        regressionModel.INPUT_WIDTH = INPUT_WIDTH;

        return regressionModel;
    }



    @Override
    public Recognition recognizeImage(Bitmap bitmap) {

        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        Object[] inputArray = {byteBuffer};

        float[] predictions = getPredictionsArray(inputArray);
        Recognition recognition = new Recognition(predictions[0],
                predictions[1], predictions[2], predictions[3]);

        return recognition;
    }

    public float[] getPredictionsArray(Object[] inputArray) {

        float[][] boxCoordinates = new float[1][4];
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, boxCoordinates);
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        float[] results = new float[4];
        results[0] = boxCoordinates[0][0];
        results[1] = boxCoordinates[0][1];
        results[2] = boxCoordinates[0][2];
        results[3] = boxCoordinates[0][3];

        return results;

    }
    public float softmax(float data[][]){
        return (float) (Math.exp(data[0][0]) / (Math.exp(data[0][0])+ Math.exp(data[0][1])) );
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }



    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * INPUT_WIDTH * INPUT_HEIGHT * PIXEL_SIZE * 4);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_WIDTH * INPUT_HEIGHT];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_HEIGHT; ++i) {
            for (int j = 0; j < INPUT_WIDTH; ++j) {
                final int val = intValues[pixel++];
                int red = ((val >> 16) & 0xFF);
                int green = ((val >> 8) & 0xFF);
                int blue = val & 0xFF;

                byteBuffer.putFloat((red-PIXEL_DEPTH/2)/ PIXEL_DEPTH);
                byteBuffer.putFloat((green-PIXEL_DEPTH/2)/ PIXEL_DEPTH);
                byteBuffer.putFloat((blue-PIXEL_DEPTH/2)/ PIXEL_DEPTH);

            }
        }
        return byteBuffer;
    }
}

