package com.android.example.plantclassifier.utils;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;

import com.android.example.plantclassifier.ml.ModelConfig;

public class ImageUtils {

//    private static final ColorMatrix INVERT = new ColorMatrix(
//            new float[]{
//                    -1, 0, 0, 0, 255,
//                    0, -1, 0, 0, 255,
//                    0, 0, -1, 0, 255,
//                    0, 0, 0, 1, 0
//            });
//
//    private static final ColorMatrix BLACK_WHITE = new ColorMatrix(
//            new float[]{
//                    0.5f, 0.5f, 0.5f, 0, 0,
//                    0.5f, 0.5f, 0.5f, 0, 0,
//                    0.5f, 0.5f, 0.5f, 0, 0,
//                    0, 0, 0, 1, 0,
//                    -1, -1, -1, 0, 1
//            }
//    );

    /**
     * Make bitmap appropriate of appropriate dimensions.
     */
    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        ColorMatrix colorMatrix = new ColorMatrix();
        //colorMatrix.setSaturation(0);
        //colorMatrix.postConcat(BLACK_WHITE);
        //colorMatrix.postConcat(INVERT);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(colorMatrix);

        Paint paint = new Paint();
        paint.setColorFilter(f);

        Bitmap bmp = Bitmap.createScaledBitmap(
                bitmap,
                ModelConfig.INPUT_IMG_SIZE_WIDTH,
                ModelConfig.INPUT_IMG_SIZE_HEIGHT,
                false);
        Canvas canvas = new Canvas(bmp);
        canvas.drawBitmap(bmp, 0, 0, paint);
        return bmp;
    }
}
