package com.android.example.plantclassifier.main;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.widget.Toast;

import com.android.example.plantclassifier.R;
import com.android.example.plantclassifier.ml.Classifier;
import com.android.example.plantclassifier.ml.ModelConfig;
import com.android.example.plantclassifier.utils.ImageUtils;
import com.android.example.plantclassifier.models.Plant;
import com.android.example.plantclassifier.models.Result;
import com.google.android.material.button.MaterialButton;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatImageView;
import androidx.appcompat.widget.AppCompatTextView;

public class MainActivity extends AppCompatActivity {

    private List<Plant> mPlantList;

    private Plant mPlant;

    private int mIndex = 0;

    private AppCompatImageView mPlantImageView;
    private AppCompatTextView mResultTextView;
    private MaterialButton mChooseButton, mClassifyButton;

    private Classifier mClassifier;

    public static Intent getIntent(Context packageContext) {
        return new Intent(packageContext, MainActivity.class);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        loadClassifier();
        setupPlantList();
        wireUpWidgets();
        showImage();
        setListeners();
    }

    private void showImage() {
        mPlant = mPlantList.get(mIndex);
        mPlantImageView.setImageResource(mPlant.getImageResourceId());
    }

    private void loadClassifier() {
        try {
            mClassifier = Classifier.createClassifier(getAssets(), ModelConfig.MODEL_FILENAME);
        } catch (IOException e) {
            Toast.makeText(this, "Couldn't load model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    private void setupPlantList() {
        mPlantList = new ArrayList<>();
        mPlantList.add(new Plant(R.drawable.test122, R.string.fat_hen_label));
        //mPlantList.add(new Plant(R.drawable.test11, R.string.wheat_label));
        mPlantList.add(new Plant(R.drawable.test223, R.string.shepherds_purse_label));
        mPlantList.add(new Plant(R.drawable.test355, R.string.cranesbill_label));
    }

    private void setListeners() {

        mChooseButton.setOnClickListener(v -> {
            mIndex = (mIndex + 1) % mPlantList.size();
            showImage();
        });

        mClassifyButton.setOnClickListener(v -> detectImage());
    }

    private void detectImage() {
        Bitmap bitmap = ((BitmapDrawable) mPlantImageView.getDrawable()).getBitmap();
        bitmap = ThumbnailUtils.extractThumbnail(bitmap, getScreenWidth(), getScreenWidth());
        bitmap = ImageUtils.prepareImageForClassification(bitmap);
        List<Result> recognitions = mClassifier.recognizeImage(bitmap);
        Result result = recognitions.get(0);
        String confidence = String.format(Locale.getDefault(), "%.2f %%", result.mConfidence * 100);
        String actualLabel = getResources().getString(mPlant.getClassLabelId());
        mResultTextView.setText(getString(R.string.result_string, result.mTitle, confidence, actualLabel));
    }

    private void wireUpWidgets() {
        mPlantImageView = findViewById(R.id.plant_image_view);
        mResultTextView = findViewById(R.id.result_text_view);
        mChooseButton = findViewById(R.id.random_choose_button);
        mClassifyButton = findViewById(R.id.classifiy_button);
    }

    private int getScreenWidth() {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return displayMetrics.widthPixels;
    }
}
