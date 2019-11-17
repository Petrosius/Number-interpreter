package com.amitshekhar.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by amitshekhar on 17/03/18.
 */

public class TensorFlowImageClassifier implements Classifier {


    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    //private static final float THRESHOLD = 0.1f;
    private static final float PIXEL_DEPTH = 255.0f;

    private Interpreter interpreter;
    private int inputSize;


    private TensorFlowImageClassifier() {

    }

    static Classifier create(AssetManager assetManager,
                             String modelPath,
                             //String labelPath,
                             int inputSize) throws IOException {

        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath));
        //classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        classifier.inputSize = inputSize;

        return classifier;
    }

    @Override
    public Bitmap convertBitmapIntoGreyscale(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        byte[] imageBytes = new byte[byteBuffer.remaining()];
        byteBuffer.get(imageBytes);
        Bitmap bmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        return bmp;
    }


    @Override
    public Recognition recognizeImage(Bitmap bitmap) {

        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        Object[] inputArray = {byteBuffer};

        int[] predictions = getPredictionsArray(inputArray);
        Recognition recognition = new Recognition(predictions[0], predictions[1],
                predictions[2], predictions[3]);

        return recognition;

    }

    public int[] getPredictionsArray(Object[] inputArray) {

        float[][] digit1 = new float[1][11];
        float[][] digit2 = new float[1][11];
        float[][] digit3 = new float[1][11];
        float[][] length = new float[1][5];


        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, digit1);
        outputMap.put(1, digit2);
        outputMap.put(2, digit3);
        outputMap.put(3, length);


        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        int[] results = new int[4];
        results[0] = argmax(digit1);
        results[1] = argmax(digit2);
        results[2] = argmax(digit3);
        results[3] = argmax(length);
       /* for(int i = 0;i < 11;i++) {
            Log.d("Tikimybės 1| " ,"Tikimybes "+ Integer.toString(i) + ": " +
                    Float.toString(result1[0][i]));
        }
        for(int i = 0;i < 11;i++) {
            Log.d("Tikimybės 2| " ,"Tikimybes "+ Integer.toString(i) + ": " +
                    Float.toString(result2[0][i]));
        }*/
        // Log.d("Preidctions", Float.toString(results[0]));
        // Log.d("Preidctions", Float.toString(results[1]));
        return results;

    }

    public int argmax(float[][] array) {

        int best = -1;
        float best_confidence = 0.0f;

        for (int i = 0; i < array[0].length; i++) {

            float value = array[0][i];
            if (value > best_confidence) {
                best_confidence = value;
                best = i;
            }
        }

        return best;


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
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE * 4);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                int red = ((val >> 16) & 0xFF);
                int green = ((val >> 8) & 0xFF);
                int blue = val & 0xFF;
                //byteBuffer.put((byte)((red + green + blue)/3));
                byteBuffer.putFloat((red - PIXEL_DEPTH / 2) / PIXEL_DEPTH);
                byteBuffer.putFloat((green - PIXEL_DEPTH / 2) / PIXEL_DEPTH);
                byteBuffer.putFloat((blue - PIXEL_DEPTH / 2) / PIXEL_DEPTH);
                //byteBuffer.putFloat(((red + green + blue)/3-PIXEL_DEPTH/2)/ PIXEL_DEPTH);
                //byteBuffer.putFloat( ((val & 255)-PIXEL_DEPTH/2)/ PIXEL_DEPTH);=
                // String msg = Integer.toString(i) +","+ Integer.toString(j) + "|" +
                //Integer.toString( val )+"| "+ Integer.toString( val & 255) + "| " +
                //         Float.toString( ((red + green + blue)/3-PIXEL_DEPTH/2)/ PIXEL_DEPTH);
                // Float.toString( ((val & 255)-PIXEL_DEPTH/2)/ PIXEL_DEPTH);
                // Log.d("Bufferis", msg);

                //  byteBuffer.put((byte) ((val >> 16) & 0xFF));
                //  byteBuffer.put((byte) ((val >> 8) & 0xFF));
                //  byteBuffer.put((byte) (val & 0xFF));

            }
        }
        return byteBuffer;
    }
}

