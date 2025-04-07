package com.antonio.my.ai.girlfriend.bridge

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class VisualAutonomyBridge(private val context: Context) {

    private val py: Python = Python.getInstance()
    private val stackModule: PyObject = py.getModule("visual_autonomy_stack")
    private val stackInstance: PyObject = stackModule.callAttr("VisualAutonomyStack")

    fun runVisualAutonomyStack(
        userId: String,
        zone: Int,
        emotion: String,
        input: String
    ): JSONObject {
        return try {
            val result = stackInstance.callAttr(
                "process_stack",
                userId,
                zone,
                emotion,
                input
            )
            JSONObject(result.toString())
        } catch (e: Exception) {
            JSONObject(
                mapOf(
                    "status" to "error",
                    "message" to (e.message ?: "Unknown error")
                )
            )
        }
    }
}
