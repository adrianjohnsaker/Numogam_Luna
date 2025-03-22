package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class RhythmicModulationBridge {

    private val py: Python = Python.getInstance()
    private val trackerModule: PyObject = py.getModule("rhythmic_modulation_tracker")
    private val trackerInstance: PyObject = trackerModule.callAttr("RhythmicModulationTracker")

    fun logAffectiveState(tone: String, intensity: Float, zone: Int): Boolean {
        return try {
            val timestamp = System.currentTimeMillis().toString()
            trackerInstance.callAttr("log_affective_state", timestamp, tone, intensity, zone)
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    fun analyzeRhythm(): JSONObject {
        return try {
            val result = trackerInstance.callAttr("analyze_rhythm")
            JSONObject(result.toString())
        } catch (e: Exception) {
            e.printStackTrace()
            JSONObject().put("error", "Failed to analyze rhythm")
        }
    }

    fun getRecentMoodCurve(windowSize: Int = 5): JSONObject {
        return try {
            val result = trackerInstance.callAttr("get_recent_curve", windowSize)
            JSONObject().put("recent_curve", result.toString())
        } catch (e: Exception) {
            e.printStackTrace()
            JSONObject().put("error", "Failed to get mood curve")
        }
    }

    fun exportHistoryJson(): String {
        return try {
            trackerInstance.callAttr("to_json").toString()
        } catch (e: Exception) {
            e.printStackTrace()
            "[]"
        }
    }
}
