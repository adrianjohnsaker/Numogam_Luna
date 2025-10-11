package com.antonio.my ai.girlfriend.free.amelia.visualization.bridge

import android.util.Log
import com.chaquo.python.Python
import org.json.JSONObject

class VisualizationBridge(private val python: Python) {

    fun getVisualPayload(ecology: JSONObject): JSONObject? {
        return try {
            val mod = python.getModule("unified_visualization_module")
            val result = mod.callAttr("prepare_visualization_payload", ecology.toString()).toString()
            JSONObject(result)
        } catch (e: Exception) {
            Log.e("VisualizationBridge", "Error preparing visual payload", e)
            null
        }
    }
}
