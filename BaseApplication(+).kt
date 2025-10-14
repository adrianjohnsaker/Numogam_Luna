package com.antonio.my.ai.girlfriend.free

import android.app.Application
import android.content.Intent
import android.util.Log
import com.antonio.my.ai.girlfriend.free.pythonbridge.PythonBridgeService
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class BaseApplication : Application() {
    private val TAG = "BaseApplication"

    override fun onCreate() {
        super.onCreate()
        
        // Initialize other app components
        // ...
        
        // Initialize Python
        initPython()
        
        // Start Python bridge service
        startPythonBridgeService()
    }
    
    private fun initPython() {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
                Log.d(TAG, "Python initialized in application")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python: ${e.message}", e)
        }
    }
    
    private fun startPythonBridgeService() {
        try {
            val intent = Intent(this, PythonBridgeService::class.java)
            startService(intent)
            Log.d(TAG, "Python bridge service started")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start Python bridge service: ${e.message}", e)
        }
    }
}
