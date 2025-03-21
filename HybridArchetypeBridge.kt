package com.antonio.my.ai.girlfriend.free

import android.util.Log
import com.chaquo.python.Python
import org.json.JSONObject

class HybridArchetypeBridge {

    fun generateHybridArchetype(clusterZoneMap: Map<Int, Map<String, Any>>): Pair<String, String> {
        return try {
            val py = Python.getInstance()
            val archetypeModule = py.getModule("hybrid_archetype_generator")

            // Convert Kotlin map to Python-compatible JSON
            val jsonInput = JSONObject()
            for ((clusterId, dataMap) in clusterZoneMap) {
                val clusterObject = JSONObject()
                for ((key, value) in dataMap) {
                    clusterObject.put(key, value)
                }
                jsonInput.put(clusterId.toString(), clusterObject)
            }

            val result = archetypeModule.callAttr("generate_hybrid_archetype", jsonInput.toMap())
            val hybridName = result.get("hybrid_archetype").toString()
            val description = result.get("description").toString()

            Pair(hybridName, description)

        } catch (e: Exception) {
            Log.e("HybridArchetypeBridge", "Error generating archetype: ${e.message}", e)
            Pair("Unknown Archetype", "Could not generate archetype due to an error.")
        }
    }

    // Optional helper for debugging/logging
    fun printArchetypeSummary(hybridPair: Pair<String, String>) {
        Log.d("HybridArchetype", "Name: ${hybridPair.first}")
        Log.d("HybridArchetype", "Description: ${hybridPair.second}")
    }

    // Helper extension to convert JSONObject to Map<String, Any>
    private fun JSONObject.toMap(): Map<String, Any> {
        val map = mutableMapOf<String, Any>()
        val keys = keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[key] = get(key)
        }
        return map
    }
}
