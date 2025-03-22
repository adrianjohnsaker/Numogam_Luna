package com.antonio.my.ai.girlfriend.free

import android.util.Log
import com.chaquo.python.Python
import org.json.JSONObject

class ArchetypeEvolutionBridge {

    fun getEvolutionReport(userId: String = "default"): Map<String, Any> {
        return try {
            val py = Python.getInstance()
            val trackerModule = py.getModule("archetype_evolution_tracker")
            val result = trackerModule.callAttr("track_archetype_evolution", userId)

            val resultJson = JSONObject(result.toString())

            val report = mutableMapOf<String, Any>()
            report["status"] = resultJson.getString("status")

            if (report["status"] == "success") {
                report["totalEvolutions"] = resultJson.getInt("total_evolutions")
                report["mostCommonTitle"] = resultJson.getString("most_common_title")
                report["mostCommonZone"] = resultJson.getInt("most_common_zone")
                report["zoneDistribution"] = resultJson.getJSONObject("zone_distribution").toString()
                report["archetypeNames"] = resultJson.getJSONArray("archetype_names").toString()
                report["currentArchetype"] = resultJson.getJSONObject("current_archetype").toString()
            } else {
                report["message"] = resultJson.getString("message")
            }

            report
        } catch (e: Exception) {
            Log.e("ArchetypeEvolution", "Error tracking evolution: ${e.message}", e)
            mapOf("status" to "error", "message" to e.message.toString())
        }
    }
}
