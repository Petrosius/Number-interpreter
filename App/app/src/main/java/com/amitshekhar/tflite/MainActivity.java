package com.amitshekhar.tflite;

import android.content.pm.ActivityInfo;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Bitmap;
import android.graphics.Paint;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;


import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_PATH = "model.tflite";
    private static final String REG_MODEL_PATH = "reg_model.tflite";
    //private static final String LABEL_PATH = "labels.txt";
    private static final int INPUT_SIZE = 54;
    private static final int INPUT_HEIGHT = 48;
    private static final int INPUT_WIDTH = 64;
    private static final int MARGIN = 7;

    private Classifier classifier;
    private RegressionModel regressionModel;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnToggleCamera;
    private ImageView imageViewResult;
    private CameraView cameraView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        cameraView = findViewById(R.id.cameraView);
        cameraView.setLockVideoAspectRatio(true);
        cameraView.setFocus(100);




        imageViewResult = findViewById(R.id.imageViewResult);
        textViewResult = findViewById(R.id.textViewResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        btnDetectObject = findViewById(R.id.btnDetectObject);

        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {

                Bitmap bitmap = cameraKitImage.getBitmap();
                Bitmap regBitmap = Bitmap.createScaledBitmap(bitmap,INPUT_WIDTH, INPUT_HEIGHT, false);

                final RegressionModel.Recognition predictions = regressionModel.recognizeImage(regBitmap);
                predictions.setBitmapSize(bitmap.getHeight(), bitmap.getWidth());
                //Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                //showImageBitmap(mutableBitmap, predictions);

                int[] coordinates = addMargin(MARGIN, predictions); //[top, height, left, width]
                Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, coordinates[2], coordinates[0],
                        coordinates[3], coordinates[1]);
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, INPUT_SIZE, INPUT_SIZE, false); //Resize bitmap
                //imageViewResult.setImageBitmap(croppedBitmap);
                final Classifier.Recognition results = classifier.recognizeImage(scaledBitmap);
                textViewResult.setText( results.toString());

                //display bitmap
                if (results.getLength() != 0){
                    Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    showImageBitmap(mutableBitmap, predictions);
                }
                else{
                    imageViewResult.setImageBitmap(bitmap);
                }
            }

            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {

            }
        });

//        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                cameraView.toggleFacing();
//            }
//        });
        btnDetectObject.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.captureImage();
            }
        });

        initTensorFlowAndLoadModel();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
                regressionModel.close();
            }
        });
    }

    private int[] addMargin(int margin, RegressionModel.Recognition predictions) {

        int[] coordinates = new int[4]; //[top, height, left, width]

        if (predictions.getTop() + predictions.getHeight() + margin < predictions.getBitmapHeight() )
            coordinates[1] = predictions.getHeight() + margin;
        else
            coordinates[1] = predictions.getBitmapHeight() - predictions.getTop();

        if (predictions.getTop() - margin > 0)
            coordinates[0] = predictions.getTop() - margin;
        else
            coordinates[2] = 0;

        if (predictions.getLeft() + predictions.getWidth() + margin < predictions.getBitmapWidth() )
            coordinates[3] = predictions.getWidth() + margin;
        else
            coordinates[3] = predictions.getBitmapWidth() - predictions.getLeft();

        if (predictions.getLeft() - margin > 0)
            coordinates[2] = predictions.getLeft() - margin;
        else
            coordinates[2] = 0;

        return coordinates;

    }

    private void showImageBitmap(Bitmap bitmap, RegressionModel.Recognition predictions){

        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStrokeWidth(20f);
        Canvas c = new Canvas(bitmap);

        int top = predictions.getTop();
        int bottom = predictions.getBottom();
        int left = predictions.getLeft();
        int right = predictions.getRight();

        c.drawLine(left, top, right, top, paint);//top line
        c.drawLine(left, bottom, right, bottom, paint);;//bottom line
        c.drawLine(left, top, left, bottom, paint);;//left line
        c.drawLine(right, top, right, bottom, paint);;//right line

        imageViewResult.setImageBitmap(bitmap);

    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            //LABEL_PATH,
                            INPUT_SIZE);

                    regressionModel = TensorFlowImageRegressionModel.create(
                            getAssets(),
                            REG_MODEL_PATH,
                            INPUT_HEIGHT,
                            INPUT_WIDTH);

                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }


    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnDetectObject.setVisibility(View.VISIBLE);
            }
        });
    }



}
