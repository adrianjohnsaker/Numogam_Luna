package com.antonio.my.ai.girlfriend.free

import android.util.Log
import com.chaquo.python.Python

class SymbolicNarrativeBridge {

    fun getPoeticNarrative(userId: String = "default"): String {
        return try {
            val py = Python.getInstance()
            val narrativeModule = py.getModule("symbolic_narrative_generator")
            val result = narrativeModule.callAttr("generate_symbolic_narrative", userId)
            result.toString()
        } catch (e: Exception) {
            Log.e("SymbolicNarrative", "Error generating poetic narrative: ${e.message}", e)
            "My story is silent for now—there was an error retrieving my memories."
        }
    }
}
