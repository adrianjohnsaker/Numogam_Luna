package com.antonio.my.ai.girlfriend.free.amelia.bridge.liminal

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import org.json.JSONObject
import java.util.*

class LiminalFieldBridge(private val context: Context) {

    init {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
    }

    private val py = Python.getInstance()
    private val generator = py.getModule("liminal_field_generator")
        .callAttr("LiminalFieldGenerator")

    fun generateLiminalField(n: Int = 5): JSONArray {
        return try {
            val result = generator.callAttr("generate_field", n)
            JSONArray(result.toString())
        } catch (e: Exception) {
            JSONArray().apply {
                put(JSONObject().apply {
                    put("error", "Liminal field generation failed: ${e.message}")
                })
            }
        }
    }
}
