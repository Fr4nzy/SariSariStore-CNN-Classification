package org.lu.sarisaristore_cnn_classification;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.lu.sarisaristore_cnn_classification.ml.ModelProducts;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imgview;
    TextView result;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.camera);
        gallery = findViewById(R.id.gallery);

        result = findViewById(R.id.result);
        imgview = findViewById(R.id.imgview);

        camera.setOnClickListener(v -> {
            if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent,3);
            }else {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
            }
        });
        gallery.setOnClickListener(v -> {
            Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(galleryIntent,1);
        });
    }

    public void classifyImage(Bitmap image){
        try {
            ModelProducts model = ModelProducts.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());

            int pixel = 0;
            // Iterates over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for (int i=0;i<imageSize;i++){
                for(int j=0;j<imageSize;j++){
                    int val = intValues[pixel++]; // RGB Values
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelProducts.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // finds the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++){
                if (confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {
                    "Carne Norte", "Tuna Adobo", "Tuna Afritada", "Tuna Caldereta", "Tuna Mechado", "Alaska Evaporada",
                    "Argentina Corned Beef", "Argentina Meatloaf", "Baby Powder", "Birch Tree", "Camel Yellow", "Century Tuna 155g",
                    "CloseUp Sachet", "Commando Matches", "Cup Noodles Beef", "Cup Noodles Seafood", "Datu Puti Patis",
                    "Datu Puti Soysauce Bottle", "Datu Puti Soysauce Sachet", "Datu Puti Vinegar Sachet", "DelMonte TomatoSauce", "Energen Chocolate",
                    "Energen Vanilla", "Great Taste Brown", "Great Taste Classic", "Hapee Toothpaste", "Hokkaido", "Ketchup Sachet",
                    "Kopiko Black", "Kopiko Blanca", "Kopiko Brown", "Lucky 7 100g", "Lucky 7 150g", "LuckyMe Beef", "LuckyMe Chicken",
                    "LuckyMe Pancit Canton", "LuckyMe SpicyBeef", "Mang Tomas", "Marlboro Red", "Mega Sardines Green", "Mega Sardines Red",
                    "Mighty Green", "Milo", "Nescafe Creamylatte", "Nescafe Decaf", "Nescafe White", "Nissin Beef", "Nissin Seafood",
                    "Nissin Spicy Seafood", "Olivenza Matches", "Payless Pancit Canton", "Plus Juice", "San Marino Corned Tuna 100g", "San Marino Corned Tuna 150g",
                    "StarMargarin Sweetblend", "Wow Ulam Caldereta", "Wow Ulam Mechado"
            };
            result.setText(classes[maxPos]);


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) { // Handles the image being taken by the camera
                Bitmap image = null;
                if (data != null && data.getExtras() != null) {
                    image = (Bitmap) data.getExtras().get("data");
                }
                if (image != null) {
                    int dimension = Math.min(image.getWidth(), image.getHeight());
                    image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                    imgview.setImageBitmap(image);

                    image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                    classifyImage(image); // This proceeds to resize it
                }
            } else {
                Uri dat = data.getData();
                if (dat != null) {
                    Bitmap image = null;
                    try {
                        image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                        imgview.setImageBitmap(image); // This handles the image in the gallery

                        image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                        classifyImage(image); // Then resize it
                    } catch (IOException e) {
                        // Handle the exception if image retrieval fails
                        e.printStackTrace();
                    }
                }else {
                    // Handle the case when the 'data' extra is null or the image is not provided
                    Toast.makeText(this, "Loading Image Failed", Toast.LENGTH_SHORT).show();
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }




}