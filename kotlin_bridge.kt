package com.antonio.my.ai.girlfriend.free

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
 * KotlinBridge manages communication between Kotlin (Android) and Python modules.
 * This class handles initialization of the Python environment and also includes
 * logic for triggering the Unified Coordinator module for cognitive interaction.
 */
class KotlinBridge(private val context: Context) {

    private val TAG = "KotlinBridge"
    private var py: Python? = null
    private var moduleCatalyst: PyObject? = null
    private var isInitialized = false

    /**
     * Initialize the Python environment and catalyst system.
     */
    fun initialize(onComplete: (Boolean) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            val success = try {
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(context))
                }

                py = Python.getInstance()

                val assetsDir = File(context.filesDir, "assets")
                if (!assetsDir.exists()) assetsDir.mkdirs()

                val modulesDir = File(assetsDir, "modules")
                if (!modulesDir.exists()) modulesDir.mkdirs()

                val modulesDirPath = modulesDir.absolutePath

                val catalystModule = py?.getModule("module_catalyst") ?: run {
                    Log.e(TAG, "Failed to import module_catalyst")
                    null
                }

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
     * Call the Unified Coordinator Module and get Amelia’s response with reasoning and zone.
     */
    fun runCoordinatorInteraction(
        userInput: String,
        memoryElements: List<String>,
        emotionalTone: String,
        tags: List<String>,
        reinforcement: Map<Int, Float>,
        onResult: (String) -> Unit
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            val result = try {
                val coordinatorModule = py?.getModule("unified_coordinator_module") ?: throw Exception("Module not found")

                val jsonInput = JSONObject().apply {
                    put("user_input", userInput)
                    put("memory_elements", memoryElements)
                    put("emotional_tone", emotionalTone)
                    put("tags", tags)
                    put("reinforcement", JSONObject(reinforcement.mapKeys { it.key.toString() }))
                }

                val output = coordinatorModule.callAttr("coordinate_modules", jsonInput.toMap())
                JSONObject(output.toString()).toString(2) // Pretty print
            } catch (e: Exception) {
                Log.e(TAG, "Coordinator interaction failed: ${e.message}")
                "{\"status\": \"error\", \"message\": \"${e.message}\"}"
            }

            withContext(Dispatchers.Main) {
                onResult(result)
            }
        }
    }

    // Extension: convert JSONObject to Map<String, Any>
    private fun JSONObject.toMap(): Map<String, Any> {
        val map = mutableMapOf<String, Any>()
        val keys = keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[key] = get(key)
        }
        return map
    }
}
