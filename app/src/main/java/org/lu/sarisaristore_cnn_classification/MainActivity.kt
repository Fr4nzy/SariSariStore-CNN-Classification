package org.lu.sarisaristore_cnn_classification

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.lu.sarisaristore_cnn_classification.ml.ModelProducts
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    var camera: Button? = null
    var gallery: Button? = null
    var imgview: ImageView? = null
    var result: TextView? = null
    var imageSize = 224
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        camera = findViewById(R.id.camera)
        gallery = findViewById(R.id.gallery)
        result = findViewById(R.id.result)
        imgview = findViewById(R.id.imgview)
        camera.apply {
            this?.setOnClickListener {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                    startActivityForResult(cameraIntent, 3)
                } else {
                    requestPermissions(
                        arrayOf(Manifest.permission.CAMERA),
                        100
                    )
                }
            }
        }

        gallery.apply {
            this?.setOnClickListener {
                val galleryIntent =
                    Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                startActivityForResult(galleryIntent, 1)
            }
        }
    }

    fun classifyImage(image: Bitmap) {
        try {
            val model = ModelProducts.newInstance(applicationContext)

            // Creates inputs for reference.
            val inputFeature0 =
                TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())
            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            var pixel = 0
            // Iterates over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val `val` = intValues[pixel++] // RGB Values
                    byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 255))
                    byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 255))
                    byteBuffer.putFloat((`val` and 0xFF) * (1f / 255))
                }
            }
            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val confidences = outputFeature0.floatArray
            // finds the index of the class with the biggest confidence.
            var maxPos = 0
            var maxConfidence = 0f
            for (i in confidences.indices) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i]
                    maxPos = i
                }
            }
            val classes = arrayOf(
                "Carne Norte",
                "Tuna Adobo",
                "Tuna Afritada",
                "Tuna Caldereta",
                "Tuna Mechado",
                "Alaska Evaporada",
                "Argentina Corned Beef",
                "Argentina Meatloaf",
                "Baby Powder",
                "Birch Tree",
                "Camel Yellow",
                "Century Tuna 155g",
                "CloseUp Sachet",
                "Commando Matches",
                "Cup Noodles Beef",
                "Cup Noodles Seafood",
                "Datu Puti Patis",
                "Datu Puti Soysauce Bottle",
                "Datu Puti Soysauce Sachet",
                "Datu Puti Vinegar Sachet",
                "DelMonte TomatoSauce",
                "Energen Chocolate",
                "Energen Vanilla",
                "Great Taste Brown",
                "Great Taste Classic",
                "Hapee Toothpaste",
                "Hokkaido",
                "Ketchup Sachet",
                "Kopiko Black",
                "Kopiko Blanca",
                "Kopiko Brown",
                "Lucky 7 100g",
                "Lucky 7 150g",
                "LuckyMe Beef",
                "LuckyMe Chicken",
                "LuckyMe Pancit Canton",
                "LuckyMe SpicyBeef",
                "Mang Tomas",
                "Marlboro Red",
                "Mega Sardines Green",
                "Mega Sardines Red",
                "Mighty Green",
                "Milo",
                "Nescafe Creamylatte",
                "Nescafe Decaf",
                "Nescafe White",
                "Nissin Beef",
                "Nissin Seafood",
                "Nissin Spicy Seafood",
                "Olivenza Matches",
                "Payless Pancit Canton",
                "Plus Juice",
                "San Marino Corned Tuna 100g",
                "San Marino Corned Tuna 150g",
                "StarMargarin Sweetblend",
                "Wow Ulam Caldereta",
                "Wow Ulam Mechado"
            )
            result!!.text = classes[maxPos]


            // Releases model resources if no longer used.
            model.close()
        } catch (e: IOException) {
            // TODO Handle the exception
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) { // Handles the image being taken by the camera
                var image: Bitmap? = null
                if (data != null && data.extras != null) {
                    image = data.extras!!["data"] as Bitmap?
                }
                if (image != null) {
                    val dimension = Math.min(image.width, image.height)
                    image = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                    imgview!!.setImageBitmap(image)
                    image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                    classifyImage(image) // This proceeds to resize it
                }
            } else {
                val dat = data!!.data
                if (dat != null) {
                    var image: Bitmap? = null
                    try {
                        image = MediaStore.Images.Media.getBitmap(this.contentResolver, dat)
                        imgview!!.setImageBitmap(image) // This handles the image in the gallery
                        image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                        classifyImage(image) // Then resize it
                    } catch (e: IOException) {
                        // Handle the exception if image retrieval fails
                        e.printStackTrace()
                    }
                } else {
                    // Handle the case when the 'data' extra is null or the image is not provided
                    Toast.makeText(this, "Loading Image Failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}