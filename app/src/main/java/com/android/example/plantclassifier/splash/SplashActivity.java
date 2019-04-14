package com.android.example.plantclassifier.splash;

import android.os.Bundle;

import com.android.example.plantclassifier.main.MainActivity;
import com.android.example.plantclassifier.R;

import androidx.appcompat.app.AppCompatActivity;

public class SplashActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTheme(R.style.SplashTheme);
        startActivity(MainActivity.getIntent(this));
        finish();
    }
}
