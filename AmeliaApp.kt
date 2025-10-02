// app/src/main/java/com/amelia/AmeliaApp.kt
package com.amelia

import android.app.Application
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.amelia.bridge.AmeliaReflectionBridge

class AmeliaApp : Application() {
    override fun onCreate() {
        super.onCreate()
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
                Log.i("AmeliaApp", "Chaquopy started")
            }
            val py = Python.getInstance()
            val hook = py.getModule("python_hook")   // your python_hook.py
            val reflection = AmeliaReflectionBridge(this)
            hook.callAttr("initialize", this, reflection)
            Log.i("AmeliaApp", "python_hook.initialize OK")
        } catch (t: Throwable) {
            Log.e("AmeliaApp", "Init failed", t)
        }
    }
}
