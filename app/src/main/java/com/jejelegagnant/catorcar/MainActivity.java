package com.jejelegagnant.catorcar;

import android.media.Image;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import android.content.Intent;
import android.net.Uri;


public class MainActivity extends AppCompatActivity {
    private View mainLayout;
    private TextView textView;
    private ImageView backgroundImage;
    private static final int REQUEST_IMAGE_PICK = 1;
    private static final int REQUEST_IMAGE_CAPTURE = 2;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        mainLayout = findViewById(R.id.main);
        textView = findViewById(R.id.textView);
        backgroundImage = findViewById(R.id.backgroundImage);


        setupAnalyzeButton();
        setupWindowInsets();
    }

    private void setupAnalyzeButton() {
        Button analyzeButton = findViewById(R.id.button);
        analyzeButton.setOnClickListener(v -> onAnalyzeButtonClick());
    }

    private void onAnalyzeButtonClick() {
        Log.d("MainActivity", "Le bouton a été cliqué !");
        String[] options = {"Choisir dans la galerie", "Prendre une photo"};
        new android.app.AlertDialog.Builder(this)
                .setTitle("Sélectionner une image")
                .setItems(options, (dialog, which) -> {
                    if (which == 0) {
                        // Galerie
                        Intent pickIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                        startActivityForResult(pickIntent, REQUEST_IMAGE_PICK);
                    } else {
                        // Appareil photo
                        Intent takePictureIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                    }
                })
                .show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) {
            Toast.makeText(this, "Aucune image sélectionnée", Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap bitmap = null;

        if (requestCode == REQUEST_IMAGE_PICK) {
            try {
                Uri imageUri = data.getData();
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            } catch (IOException e) {
                Toast.makeText(this, "Erreur lors du chargement de l'image", Toast.LENGTH_SHORT).show();
                Log.e("Image", "Erreur chargement image", e);
                return;
            }
        } else if (requestCode == REQUEST_IMAGE_CAPTURE) {
            Bundle extras = data.getExtras();
            if (extras != null && extras.get("data") instanceof Bitmap) {
                bitmap = (Bitmap) extras.get("data");
            } else {
                Toast.makeText(this, "Erreur lors de la capture", Toast.LENGTH_SHORT).show();
                return;
            }
        }

        if (bitmap != null) {
            runModel(bitmap); // exécute l’analyse avec le modèle
        }
    }
    private void runModel(Bitmap bitmap) {
        Toast.makeText(MainActivity.this, "Analyse en cours...", Toast.LENGTH_SHORT).show();

        int inputSize = 260;
        Interpreter tflite = null;

        try {
            tflite = new Interpreter(FileUtil.loadMappedFile(this, "2.tflite"));
            List<String> labels = FileUtil.loadLabels(this, "labels.txt");

            int[] inputShape = tflite.getInputTensor(0).shape();
            DataType inputType = tflite.getInputTensor(0).dataType();

            TensorImage tensorImage = preprocessImage(bitmap, inputSize, inputType);

            int[] outputShape = tflite.getOutputTensor(0).shape();
            DataType outputType = tflite.getOutputTensor(0).dataType();
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);

            tflite.run(tensorImage.getBuffer(), outputBuffer.getBuffer());

            handlePredictionResult(outputBuffer, labels);

        } catch (IOException e) {
            Toast.makeText(this, "Erreur lors du chargement du modèle", Toast.LENGTH_LONG).show();
            Log.e("Model", "Erreur chargement modèle", e);
        } finally {
            if (tflite != null) {
                tflite.close();
            }
        }
    }


    private TensorImage preprocessImage(Bitmap bitmap, int inputSize, DataType inputType) {
        TensorImage tensorImage = new TensorImage(inputType);
        tensorImage.load(bitmap);

        ImageProcessor.Builder processorBuilder = new ImageProcessor.Builder()
                .add(new ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR));

        if (inputType == DataType.FLOAT32) {
            processorBuilder.add(new NormalizeOp(0f, 255f));
        }

        ImageProcessor imageProcessor = processorBuilder.build();
        return imageProcessor.process(tensorImage);
    }

    private void handlePredictionResult(TensorBuffer outputBuffer, List<String> labels) {
        float[] scores = outputBuffer.getFloatArray();

        int maxIdx = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[maxIdx]) maxIdx = i;
        }

        String result = labels.get(maxIdx);
        float confidence = scores[maxIdx];

        Log.d("Prediction", "Top label: " + result + " (score=" + confidence + ")");
        Log.d("Prediction", "All scores: " + Arrays.toString(scores));

        Toast.makeText(this, "Prédit : " + result + "\n(Confiance : " + confidence + ")", Toast.LENGTH_LONG).show();
        if (isIndexCar(maxIdx)) {
            mainLayout.setBackgroundColor(getResources().getColor(android.R.color.holo_blue_dark));
            backgroundImage.setImageResource(R.drawable.car_background);
            textView.setText("It is a car!" + "\n" + result);
        } else if (isIndexCat(maxIdx)) {
            mainLayout.setBackgroundColor(getResources().getColor(android.R.color.holo_orange_light));
            backgroundImage.setImageResource(R.drawable.cat_background);
            textView.setText("It is a cat!" + "\n" + result);
        } else {
            mainLayout.setBackgroundColor(getResources().getColor(android.R.color.darker_gray));
            backgroundImage.setImageResource(R.drawable.idk_background);
            textView.setText("It is neither a cat nor a car."+ "\n" + result);
        }
    }
    private boolean isIndexCat(int index) {
        // Vérifie si l'index correspond à un label de chat
        return index > 280 && index < 294 || index == 383;
    }
    private boolean isIndexCar(int index) {
        // Vérifie si l'index correspond à un label de voiture
        return index == 436 || index == 468 || index == 511 || index == 609 || index == 619 || index == 656
                || index == 661 || index == 705 || index == 717 || index == 751 || index == 757 || index == 817;
    }

    private void setupWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

}