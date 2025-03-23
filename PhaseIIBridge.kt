package com.antonio.my.ai.girlfriend.free.

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject

class PhaseIIBridge(private val context: Context) {

    private var py: Python? = null
    private var mirrorModule: PyObject? = null
    private var narrationModule: PyObject? = null
    private var metaModule: PyObject? = null

    fun initialize(): Boolean {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }

        py = Python.getInstance()

        return try {
            mirrorModule = py?.getModule("symbolic_mirror_engine")
            narrationModule = py?.getModule("self_narration_generator")
            metaModule = py?.getModule("recursive_meta_analysis_loop")
            mirrorModule != null && narrationModule != null && metaModule != null
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    fun reflectSymbolically(userInput: String, zone: Int, emotion: String): String {
        return try {
            val result = mirrorModule?.callAttr("generate_reflection", userInput, zone, emotion)
            result?.toString() ?: "Symbolic reflection unavailable."
        } catch (e: Exception) {
            "Error during symbolic reflection."
        }
    }

    fun generateSelfNarration(userInput: String, archetype: String, zone: Int): String {
        return try {
            val result = narrationModule?.callAttr("generate_narration", userInput, archetype, zone)
            result?.toString() ?: "Narration generation failed."
        } catch (e: Exception) {
            "Narrative processing error."
        }
    }

    fun performMetaAnalysis(lastOutput: String, tone: String): String {
        return try {
            val result = metaModule?.callAttr("analyze_and_adjust", lastOutput, tone)
            result?.toString() ?: "Meta analysis inconclusive."
        } catch (e: Exception) {
            "Meta-analysis error."
        }
    }

    fun fullPhaseIICycle(
        userInput: String,
        zone: Int,
        emotion: String,
        archetype: String
    ): JSONObject {
        val json = JSONObject()
        try {
            val reflection = reflectSymbolically(userInput, zone, emotion)
            val narration = generateSelfNarration(userInput, archetype, zone)
            val refined = performMetaAnalysis(narration, emotion)

            json.put("symbolic_reflection", reflection)
            json.put("self_narration", narration)
            json.put("refined_output", refined)
        } catch (e: Exception) {
            json.put("error", "Phase II processing failed.")
        }
        return json
    }
}
