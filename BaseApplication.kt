package com.antonio.my.ai.girlfriend.free

import android.app.Application
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.util.Log

// This is an extension of your existing BaseApplication
class BaseApplication : Application() {
    private val TAG = "BaseApplication"
    private var pythonBridge: PythonBridge? = null
    
    override fun onCreate() {
        super.onCreate()
        
        Log.d(TAG, "BaseApplication.onCreate() - Initializing")
        
        // Initialize Python bridge with delay to ensure app is fully loaded
        Handler(Looper.getMainLooper()).postDelayed({
            initializePythonBridge()
            
            // Launch PythonHookActivity to initialize all modules
            val intent = Intent(this, PythonHookActivity::class.java)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            startActivity(intent)
        }, 2000) // 2-second delay
    }
    
    private fun initializePythonBridge() {
        try {
            Log.d(TAG, "Initializing Python bridge")
            pythonBridge = PythonBridge(this)
            Log.d(TAG, "Python bridge initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python bridge: ${e.message}")
            e.printStackTrace()
        }
    }
    
    fun getPythonBridge(): PythonBridge {
        if (pythonBridge == null) {
            Log.d(TAG, "Creating new Python bridge instance")
            pythonBridge = PythonBridge(this)
        }
        return pythonBridge!!
    }
}
