package com.example.myapp

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

class KotlinBridge(private val context: Context) {

    private val TAG = "KotlinBridge"
    private var py: Python? = null
    private var moduleCatalyst: PyObject? = null
    private var moduleStatusListener: ((JSONObject) -> Unit)? = null
    private var isInitialized = false

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
                val catalystModule = py?.getModule("unified_coordinator_module") ?: run {
                    Log.e(TAG, "Failed to import unified_coordinator_module")
                    null
                }

                // Initialize the catalyst system with this bridge
                moduleCatalyst = catalystModule?.callAttr(
                    "coordinate_modules",
                    this@KotlinBridge
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
     * Coordinate interaction with the Python module
     */
    fun coordinateInteraction(inputData: Map<String, Any>, onSuccess: (JSONObject) -> Unit, onError: (String) -> Unit) {
        if (isInitialized) {
            try {
                // Send the interaction payload to the Python module
                val result = moduleCatalyst?.callAttr("coordinate_modules", inputData)
                val jsonResponse = JSONObject(result.toString())

                // Process the response from the Python module
                onSuccess(jsonResponse)
            } catch (e: Exception) {
                Log.e(TAG, "Error during interaction: ${e.message}")
                onError("Error processing interaction: ${e.message}")
            }
        } else {
            onError("Python module is not initialized.")
        }
    }
}
