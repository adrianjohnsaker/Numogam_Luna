package com.antonio.my.ai.girlfriend.free.visual

import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class GlyphVisualBridge {

    private val TAG = "GlyphVisualBridge"
    private val py: Python = Python.getInstance()
    private val glyphModule: PyObject? = py.getModule("glyph_visual_generator")

    fun generateVisual(inputJson: JSONObject): JSONObject {
        return try {
            val result = glyphModule
                ?.callAttr("generate_visual", inputJson.toString())
                ?.toJava(String::class.java)
                ?: return errorJson("Null response from Python")
            JSONObject(result)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating visual", e)
            errorJson(e.message ?: "Unknown error")
        }
    }

    private fun errorJson(msg: String): JSONObject {
        return JSONObject().apply {
            put("status", "error")
            put("message", msg)
        }
    }
}
