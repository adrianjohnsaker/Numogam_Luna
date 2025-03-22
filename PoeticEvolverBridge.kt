package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.Python
import com.chaquo.python.PyObject
import org.json.JSONObject

class PoeticEvolverBridge {

    private val python: Python = Python.getInstance()
    private val poeticModule: PyObject = python.getModule("poetic_language_evolver")

    fun generatePoeticExpression(
        basePhrase: String,
        zone: Int,
        emotion: String
    ): String {
        val input = JSONObject().apply {
            put("base_phrase", basePhrase)
            put("zone", zone)
            put("emotion", emotion)
        }

        val result = poeticModule.callAttr("generate_poetic_phrase", input.toString())
        val resultJson = JSONObject(result.toString())

        return resultJson.getString("poetic_expression")
    }
}
