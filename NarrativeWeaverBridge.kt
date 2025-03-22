package com.antonio.my.ai.girlfriend.free.modules

import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class NarrativeWeaverBridge {

    private val py: Python = Python.getInstance()
    private val narrativeModule: PyObject = py.getModule("narrative_weaver_module")

    fun generateNarrativeArc(
        userInput: String,
        memoryHighlights: List<String>,
        currentZone: Int,
        emotion: String
    ): NarrativeResult {
        return try {
            val payload = JSONObject().apply {
                put("user_input", userInput)
                put("memory_highlights", memoryHighlights)
                put("current_zone", currentZone)
                put("emotion", emotion)
            }

            val result = narrativeModule.callAttr("generateNarrativeArc", payload.toString())
            val json = JSONObject(result.toString())

            NarrativeResult(
                narrative = json.getString("narrative"),
                emotion = json.getString("emotion"),
                zone = json.getInt("zone"),
                rhythmTag = json.getString("rhythm_tag"),
                motif = json.getString("motif")
            )
        } catch (e: Exception) {
            Log.e("NarrativeWeaver", "Error generating narrative arc", e)
            NarrativeResult(error = "Failed to generate narrative: ${e.message}")
        }
    }

    data class NarrativeResult(
        val narrative: String = "",
        val emotion: String = "",
        val zone: Int = -1,
        val rhythmTag: String = "",
        val motif: String = "",
        val error: String? = null
    )
}
