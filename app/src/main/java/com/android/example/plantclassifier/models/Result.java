package com.android.example.plantclassifier.models;

import java.util.Locale;

public class Result {

    public final String mTitle;
    public final float mConfidence;

    public Result(String title, float confidence) {
        this.mTitle = title;
        this.mConfidence = confidence;
    }

    @Override
    public String toString() {
        return mTitle + " " + String.format(Locale.getDefault(), "(%.1f%%) ", mConfidence * 100.0f);
    }

}
