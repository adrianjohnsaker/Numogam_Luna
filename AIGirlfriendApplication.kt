package com.antonio.my.ai.girlfriend.free

import android.app.Application
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class AIGirlfriendApplication : Application() {
    
    companion object {
        private const val TAG = "AIGirlfriendApp"
    }
    
    override fun onCreate() {
        super.onCreate()
        
        Log.d(TAG, "Initializing AI Girlfriend Application...")
        
        // Initialize Python first
        initializePython()
        
        // Pre-load Python modules in background
        preloadPythonModules()
        
        Log.d(TAG, "AI Girlfriend Application initialized successfully")
    }
    
    private fun initializePython() {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
                Log.d(TAG, "Python started successfully")
            } else {
                Log.d(TAG, "Python was already started")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start Python", e)
        }
    }
    
    private fun preloadPythonModules() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val python = Python.getInstance()
                
                // Test basic Python functionality
                val builtins = python.getModule("builtins")
                val testResult = builtins.callAttr("len", "test").toInt()
                Log.d(TAG, "Python basic test passed: $testResult")
                
                // Pre-load our AI module promoter
                val aiModule = python.getModule("ai_assistant_module_promoter")
                Log.d(TAG, "AI Module Promoter loaded successfully")
                
                // Test the module functionality
                val androidModules = aiModule.callAttr("get_android_optimized_modules")
                    .asList()
                    .map { it.toString() }
                Log.d(TAG, "Android optimized modules: $androidModules")
                
                // Pre-load common Python modules that our system uses
                preloadCommonModules(python)
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to preload Python modules", e)
            }
        }
    }
    
    private fun preloadCommonModules(python: Python) {
        val commonModules = listOf(
            "pandas",
            "numpy", 
            "requests",
            "json",
            "datetime"
        )
        
        commonModules.forEach { moduleName ->
            try {
                python.getModule(moduleName)
                Log.d(TAG, "Successfully preloaded: $moduleName")
            } catch (e: Exception) {
                Log.w(TAG, "Could not preload $moduleName: ${e.message}")
                // This is fine - some modules might not be available
            }
        }
    }
}
