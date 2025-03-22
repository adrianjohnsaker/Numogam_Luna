package com.antonio.my.ai.girlfriend.free

import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class PoeticExpressionBridge {

    private val py: Python = Python.getInstance()
    private val poeticModule: PyObject = py.getModule("poetic_expression_generator")

    fun generatePoeticExpression(
        theme: String,
        tone: String,
        activeArchetype: String
    ): JSONObject? {
        return try {
            val result: PyObject = poeticModule.callAttr(
                "generate_poetic_expression",
                theme,
                tone,
                activeArchetype
            )
            JSONObject(result.toString())
        } catch (e: Exception) {
            Log.e("PoeticExpression", "Error generating poetic expression", e)
            null
        }
    }
}
