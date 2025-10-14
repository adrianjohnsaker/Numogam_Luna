package com.antonio.my.ai.girlfriend.free.amelia

import android.content.Context
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

/**
 * KotlinBridge manages communication between Kotlin (Android) and Python modules
 * This class handles initialization of the Python environment and module catalyst system
 */
class KotlinBridge(private val context: Context) {
    
    private val TAG = "KotlinBridge"
    private var py: Python? = null
    private var moduleCatalyst: PyObject? = null
    private var moduleStatusListener: ((JSONObject) -> Unit)? = null
    private var isInitialized = false
    
    /**
     * Send status information back to Android UI
     */
    fun sendStatus(statusJson: String) {
        try {
            val status = JSONObject(statusJson)
            Log.d(TAG, "Module catalyst status: $status")
            moduleStatusListener?.invoke(status)
        } catch (e: Exception) {
            Log.e(TAG, "Error processing status: ${e.message}")
        }
    }
    
    /**
     * Set a listener for module status updates
     */
    fun setModuleStatusListener(listener: (JSONObject) -> Unit) {
        moduleStatusListener = listener
    }
    
    /**
     * Stimulate all modules with input data
     */
    fun stimulateModules(inputData: JSONObject, callback: (JSONObject?) -> Unit) {
        if (!isInitialized) {
            Log.e(TAG, "Cannot stimulate: ModuleCatalyst not initialized")
            callback(null)
            return
        }
        
        CoroutineScope(Dispatchers.IO).launch {
            val result = try {
                val pyInputData = py?.getBuiltins()?.callAttr("dict")
                
                // Convert JSON to Python dict
                val keys = inputData.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    when (val value = inputData.get(key)) {
                        is String -> pyInputData?.callAttr("__setitem__", key, value)
                        is Int -> pyInputData?.callAttr("__setitem__", key, value)
                        is Double -> pyInputData?.callAttr("__setitem__", key, value)
                        is Boolean -> pyInputData?.callAttr("__setitem__", key, value)
                        else -> {
                            // For complex objects, convert to string representation
                            pyInputData?.callAttr("__setitem__", key, value.toString())
                        }
                    }
                }
                
                // Call stimulate_all_modules on the catalyst
                val pyResult = moduleCatalyst?.callAttr("stimulate_all_modules", pyInputData)
                
                // Convert Python dict to JSON
                val jsonResult = pyResult?.toString()
                if (jsonResult != null) {
                    JSONObject(jsonResult)
                } else {
                    JSONObject().put("error", "No result from stimulate_all_modules")
                }
                
            } catch (e: PyException) {
                Log.e(TAG, "Python error during stimulation: ${e.message}", e)
                JSONObject().put("error", "Python error: ${e.message}")
            } catch (e: Exception) {
                Log.e(TAG, "Error during stimulation: ${e.message}", e)
                JSONObject().put("error", "Error: ${e.message}")
            }
            
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }
    
    /**
     * Call a specific method on a module
     */
    fun callModuleMethod(moduleName: String, methodName: String, params: JSONObject?, callback: (Any?) -> Unit) {
        if (!isInitialized) {
            Log.e(TAG, "Cannot call method: ModuleCatalyst not initialized")
            callback(null)
            return
        }
        
        CoroutineScope(Dispatchers.IO).launch {
            val result = try {
                // Get the module from the catalyst
                val module = moduleCatalyst?.callAttr("modules")?.__getitem__(moduleName)
                
                // Prepare parameters as Python objects if provided
                val args = if (params != null) {
                    val pyDict = py?.getBuiltins()?.callAttr("dict")
                    
                    val keys = params.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        pyDict?.callAttr("__setitem__", key, params.get(key).toString())
                    }
                    
                    arrayOf(pyDict)
                } else {
                    arrayOf<Any>()
                }
                
                // Call the method
                if (module != null) {
                    module.callAttr(methodName, *args)
                } else {
                    Log.e(TAG, "Module $moduleName not found")
                    null
                }
                
            } catch (e: PyException) {
                Log.e(TAG, "Python error calling $moduleName.$methodName: ${e.message}", e)
                null
            } catch (e: Exception) {
                Log.e(TAG, "Error calling $moduleName.$methodName: ${e.message}", e)
                null
            }
            
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }
    
    /**
     * Shutdown the Python environment and catalyst system
     */
    fun shutdown() {
        if (isInitialized) {
            try {
                moduleCatalyst?.callAttr("shutdown")
                isInitialized = false
            } catch (e: Exception) {
                Log.e(TAG, "Error shutting down: ${e.message}")
            }
        }
    }
    
    /**
     * Initialize the Python environment and module catalyst system
     */
    fun initialize(onComplete: (Boolean) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            val success = try {
                // Initialize Python if not already done
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(context))
                }
                
                py = Python.getInstance()
                
                // Make sure assets directory exists
                val assetsDir = File(context.filesDir, "assets")
                if (!assetsDir.exists()) {
                    assetsDir.mkdirs()
                }
                
                // Make modules directory if it doesn't exist
                val modulesDir = File(assetsDir, "modules")
                if (!modulesDir.exists()) {
                    modulesDir.mkdirs()
                }
                
                // Path to modules directory
                val modulesDirPath = modulesDir.absolutePath
                
                // Import the catalyst module
                val catalystModule = py?.getModule("module_catalyst") ?: run {
                    Log.e(TAG, "Failed to import module_catalyst")
                    null
                }
                
                // Initialize the catalyst system with this bridge
                moduleCatalyst = catalystModule?.callAttr(
                    "initialize_catalyst_system",
                    this@KotlinBridge,
                    modulesDirPath
                )
                
                isInitialized = moduleCatalyst != null
                isInitialized
                
            } catch (e: PyException) {
                Log.e(TAG, "Python error during initialization: ${e.message}", e)
                false
            } catch (e: Exception) {
                Log.e(TAG, "Error during initialization: ${e.message}", e)
                false
            }
            
            withContext(Dispatchers.Main) {
                onComplete(success)
            }
        }
    }
    
    /**
