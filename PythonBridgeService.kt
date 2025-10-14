package com.antonio.my.ai.girlfriend.free.pythonbridge

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Service that maintains the Python bridge connection
 * This allows the Python modules to continue running even when activities are destroyed
 */
class PythonBridgeService : Service() {
    private val TAG = "PythonBridgeService"
    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)
    private lateinit var pythonBridge: PythonBridge
    
    // Binder for client communication
    private val binder = LocalBinder()
    
    inner class LocalBinder : Binder() {
        fun getService(): PythonBridgeService = this@PythonBridgeService
    }
    
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "PythonBridgeService created")
        pythonBridge = PythonBridge.getInstance(applicationContext)
        
        // Initialize Python environment
        serviceScope.launch {
            if (pythonBridge.initialize()) {
                Log.d(TAG, "Python environment initialized successfully in service")
            } else {
                Log.e(TAG, "Failed to initialize Python environment in service")
            }
        }
    }
    
    override fun onBind(intent: Intent): IBinder {
        return binder
    }
    
    /**
     * Process a query through the Python bridge
     */
    suspend fun processQuery(query: String): JSONObject {
        return pythonBridge.processQuery(query)
    }
    
    /**
     * Get router status
     */
    suspend fun getRouterStatus(): JSONObject {
        return pythonBridge.getRouterStatus()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "PythonBridgeService destroyed")
        pythonBridge.cleanup()
        serviceScope.cancel()
    }
}
