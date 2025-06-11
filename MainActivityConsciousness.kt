package com.antonio.my.ai.girlfriend.free.consciousnessui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.example.consciousnessui.ui.theme.ConsciousnessUITheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ConsciousnessUITheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    ConsciousnessStateVisualizer()
                }
            }
        }
    }
}
