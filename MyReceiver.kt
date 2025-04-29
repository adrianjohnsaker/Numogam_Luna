package com.antonio.my.ai.girlfriend.free

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * Broadcast receiver to handle system events and initialize Python environment
 * when needed
 */
class MyReceiver : BroadcastReceiver() {
    private val TAG = "MyReceiver"
    
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED -> {
                Log.d(TAG, "Boot completed - initializing Python")
                initializePython(context)
            }
            "com.antonio.my.ai.girlfriend.free.INITIALIZE_PYTHON" -> {
                Log.d(TAG, "Manual initialization request")
                initializePython(context)
            }
        }
    }
    
    private fun initializePython(context: Context) {
        try {
            // Initialize Python bridge
            val bridge = ChaquopyBridge.getInstance(context)
            
            // Pre-warm modules by loading them
            val testModule = bridge.getModule("test_module")
            val result = testModule?.callAttr("test_function", "Initialization from BroadcastReceiver")
            
            Log.d(TAG, "Python initialization result: $result")
            
            // Report success to any listeners
            val resultIntent = Intent("com.antonio.my.ai.girlfriend.free.PYTHON_INITIALIZED")
            resultIntent.putExtra("success", true)
            context.sendBroadcast(resultIntent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python: ${e.message}")
            
            // Report failure
            val resultIntent = Intent("com.antonio.my.ai.girlfriend.free.PYTHON_INITIALIZED")
            resultIntent.putExtra("success", false)
            resultIntent.putExtra("error", e.message)
            context.sendBroadcast(resultIntent)
        }
    }
}
