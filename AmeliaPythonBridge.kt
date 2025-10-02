// app/src/main/java/com/amelia/bridge/AmeliaPythonBridge.kt
package com.amelia.bridge

import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

/**
 * Definitive bridge to Python pipeline:
 * calls python_hook.process_input(text, headers) → pipeline.process(text, headers)
 * and returns choices[0].message.content for the UI to render.
 */
object AmeliaPythonBridge {

    private const val TAG = "AmeliaPythonBridge"

    fun processText(text: String): String = process(text, emptyMap())

    fun process(text: String, headers: Map<String, String> = emptyMap()): String {
        return runCatching {
            val py = Python.getInstance()
            val hook = py.getModule("python_hook")

            val pyHeaders: PyObject = PyObject.fromJava(headers)
            val resPy = hook.callAttr("process_input", text, pyHeaders)
            val resStr = resPy.toString() // OpenAI-style JSON

            val root = JSONObject(resStr)
            val choices = root.getJSONArray("choices")
            val content = choices
                .getJSONObject(0)
                .getJSONObject("message")
                .optString("content", "")
                .ifBlank { "[Amelia] (no content)" }

            Log.d(TAG, "pipeline ok; content length=${content.length}")
            content
        }.getOrElse { t ->
            Log.w(TAG, "pipeline error", t)
            "[Amelia·Bridge] error: ${t.message ?: "unknown"}"
        }
    }
}
