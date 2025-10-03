package com.antonio.my.ai.girlfriend.free.bridge

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject

/**
 * PipelineBridge
 * ---------------
 * A robust Kotlin interface to Amelia's pipeline.py
 *
 * Exposes:
 *  - processText(userInput: String, headers: Map<String,String>? = null): JSONObject?
 *  - Utility init() to bootstrap Python if needed.
 *
 * Ensures every response is wrapped in JSON and errors are logged.
 */
class PipelineBridge private constructor(private val context: Context) {

    companion object {
        private const val TAG = "PipelineBridge"
        @Volatile private var instance: PipelineBridge? = null

        fun getInstance(context: Context): PipelineBridge {
            return instance ?: synchronized(this) {
                instance ?: PipelineBridge(context.applicationContext).also { it.init() ; instance = it }
            }
        }
    }

    private lateinit var py: Python

    private fun init() {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
                Log.i(TAG, "Chaquopy Python started")
            }
            py = Python.getInstance()
            Log.i(TAG, "PipelineBridge initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "PipelineBridge init failed", e)
            throw e
        }
    }

    /**
     * Calls pipeline.process(userText, headers).
     * Returns JSONObject or null if failure.
     */
    fun processText(userInput: String, headers: Map<String, String>? = null): JSONObject? {
        return try {
            val module: PyObject = py.getModule("pipeline")
            val pyHeaders: PyObject? = headers?.let {
                val jsonString = JSONObject(it).toString()
                py.getModule("json").callAttr("loads", jsonString)
            }

            val result: PyObject = if (pyHeaders != null) {
                module.callAttr("process", userInput, pyHeaders)
            } else {
                module.callAttr("process", userInput, null)
            }

            val json = JSONObject(result.toString())
            Log.i(TAG, "Pipeline.processText OK: ${json.optString("id")} (${json.optJSONObject("choices")})")
            json
        } catch (e: Exception) {
            Log.e(TAG, "Pipeline.processText failed", e)
            null
        }
    }
}
