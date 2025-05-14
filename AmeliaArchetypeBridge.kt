package com.antonio.my.ai.girlfriend.free.archetypal

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

class AmeliaArchetypeBridge private constructor(private val context: Context) {
    private val TAG = "AmeliaArchetypeBridge"
    private var pythonModule: PyObject? = null
    private var ameliaArchetypeModule: PyObject? = null

    init {
        initializePython(context)
    }

    private fun initializePython(context: Context) {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        try {
            val py = Python.getInstance()
            pythonModule = py.getModule("amelia_archetype_module")
            // Instantiate the Unified Archetype Module in Python
            ameliaArchetypeModule = pythonModule?.callAttr("AmeliaArchetypeModule")
            Log.d(TAG, "Python archetype module initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python archetype module: ${e.message}")
            e.printStackTrace()
        }
    }

    suspend fun createSession(sessionName: String? = null): JSONObject = withContext(Dispatchers.IO) {
        try {
            val inputData = JSONObject().apply {
                put("operation", "create_session")
                put("session_name", sessionName)
            }
            val resultString = ameliaArchetypeModule?.callAttr("process_kotlin_input", inputData.toString())?.toString()
                ?: "{\"status\":\"error\",\"error\":\"Module not initialized\"}"
            return@withContext JSONObject(resultString)
        } catch (e: Exception) {
            Log.e(TAG, "Error creating session: ${e.message}")
            return@withContext JSONObject().apply {
                put("status", "error")
                put("error", e.message)
            }
        }
    }

    suspend fun endSession(): JSONObject = withContext(Dispatchers.IO) {
        try {
            val inputData = JSONObject().apply {
                put("operation", "end_session")
            }
            val resultString = ameliaArchetypeModule?.callAttr("process_kotlin_input", inputData.toString())?.toString()
                ?: "{\"status\":\"error\",\"error\":\"Module not initialized\"}"
            return@withContext JSONObject(resultString)
        } catch (e: Exception) {
            Log.e(TAG, "Error ending session: ${e.message}")
            return@withContext JSONObject().apply {
                put("status", "error")
                put("error", e.message)
            }
        }
    }

    suspend fun generateComplexArchetype(
        baseArchetype: String,
        emotionalTone: String,
        zoneDepth: Int
    ): JSONObject = withContext(Dispatchers.IO) {
        try {
            val inputData = JSONObject().apply {
                put("operation", "generate_complex_archetype")
                put("base_archetype", baseArchetype)
                put("emotional_tone", emotionalTone)
                put("zone_depth", zoneDepth)
            }
            val resultString = ameliaArchetypeModule?.callAttr("process_kotlin_input", inputData.toString())?.toString()
                ?: "{\"status\":\"error\",\"error\":\"Module not initialized\"}"
            return@withContext JSONObject(resultString)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating complex archetype: ${e.message}")
            return@withContext JSONObject().apply {
                put("status", "error")
                put("error", e.message)
            }
        }
    }

    suspend fun getComponentStatus(): JSONObject = withContext(Dispatchers.IO) {
        try {
            val resultString = ameliaArchetypeModule?.callAttr("get_component_status")?.toString()
                ?: "{\"status\":\"error\",\"error\":\"Module not initialized\"}"
            return@withContext JSONObject(resultString)
        } catch (e: Exception) {
            Log.e(TAG, "Error getting component status: ${e.message}")
            return@withContext JSONObject().apply {
                put("status", "error")
                put("error", e.message)
            }
        }
    }

    companion object {
        @Volatile
        private var instance: AmeliaArchetypeBridge? = null

        fun initialize(context: Context) {
            if (instance == null) {
                synchronized(this) {
                    if (instance == null) {
                        instance = AmeliaArchetypeBridge(context)
                    }
                }
            }
        }

        fun getInstance(context: Context): AmeliaArchetypeBridge {
            return instance
                ?: throw IllegalStateException("AmeliaArchetypeBridge is not initialized. Call initialize(context) first.")
        }
    }
}
