package com.antonio.my.ai.girlfriend.free

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

object PythonInitializer {
    fun initialize(context: Context) {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
    }
}
