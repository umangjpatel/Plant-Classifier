package com.android.example.plantclassifier.ml;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ModelConfig {

    public static final String MODEL_FILENAME = "converted_model.tflite";

    public static final int INPUT_IMG_SIZE_WIDTH = 70, INPUT_IMG_SIZE_HEIGHT = 70;
    private static final int FLOAT_TYPE_SIZE = 4, PIXEL_SIZE = 3;
    static final int MAX_CLASSIFICATION_RESULTS = 1;

    //public static final int IMAGE_MEAN = 70;

    static final int MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE;

    static final List<String> OUTPUT_LABELS = Collections.unmodifiableList(Arrays.asList("Black Grass",
            "Charlock", "Cleavers", "Common Chickweed", "Common Wheat", "Fat Hen",
            "Loose Silky-bent", "Maize", "Scentless Mayweed",
            "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"));


    static final float CLASSIFICATION_THRESHOLD = 0.5f, IMAGE_STD = 255.0f;

}
