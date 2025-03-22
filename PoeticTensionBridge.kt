package com.antonio.my.ai.girlfriend.free

import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.Python
import org.json.JSONObject

class PoeticTensionBridge {

    private val TAG = "PoeticTensionBridge"
    private val py = Python.getInstance()

    fun evaluateTension(
        emotionalTone: String,
        recentPhrases: List<String>,
        rhythmFluctuation: Double
    ): String {
        return try {
            val poeticModule = py.getModule("poetic_tension_modulation_engine")
            val result = poeticModule.callAttr(
                "evaluate_poetic_tension",
                emotionalTone,
                recentPhrases,
                rhythmFluctuation
            )
            result.toString()
        } catch (e: PyException) {
            Log.e(TAG, "Python exception: ${e.message}", e)
            "Error evaluating poetic tension"
        } catch (e: Exception) {
            Log.e(TAG, "General exception: ${e.message}", e)
            "Error evaluating poetic tension"
        }
    }
}
