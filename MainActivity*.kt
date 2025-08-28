package com.antonio.my.ai.girlfriend.free

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.antonio.my.ai.girlfriend.free.util.AppContextHolder

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppContextHolder.init(this)

        // Initialize Python if it hasn't been started yet
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        setContentView(R.layout.activity_main)

        // Example: Call a Python function
        val py = Python.getInstance()
        val module = py.getModule("my_enhanced_module") // Replace with your actual module name
        val result = module.callAttr("function_name", "argument1", "argument2")

        // Process the result as needed...
    }
}
