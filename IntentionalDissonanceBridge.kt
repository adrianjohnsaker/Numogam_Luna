package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class IntentionalDissonanceBridge {
    private val py: Python = Python.getInstance()
    private val dissonanceModule: PyObject = py.getModule("intentional_symbolic_dissonance_module")

    fun generateSymbolicDissonance(emotion: String, motifs: List<String>): JSONObject {
        val kwargs = mapOf(
            "emotion" to emotion,
            "motif_pool" to motifs
        )
        val result: PyObject = dissonanceModule.callAttr("generate_dissonant_expression", kwargs)
        return JSONObject(result.toString())
    }
}
