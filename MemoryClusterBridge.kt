package com.antonio.my.ai.girlfriend.free

import android.util.Log
import com.chaquo.python.Python
import org.json.JSONArray
import org.json.JSONObject

class MemoryClustererBridge {

    fun clusterMemories(memoryEntries: List<String>, numClusters: Int = 3): String {
        return try {
            val py = Python.getInstance()
            val clustererModule = py.getModule("memory_clusterer")
            val clustererClass = clustererModule.callAttr("MemoryClusterer", numClusters)

            // Pass memory entries to cluster
            val pyClusters = clustererClass.callAttr("cluster_memories", memoryEntries)

            // Summarize cluster content
            val summaries = clustererClass.callAttr("summarize_clusters")

            // Format result
            val result = JSONObject()
            result.put("clusters", JSONObject(pyClusters.toString()))
            result.put("summaries", JSONObject(summaries.toString()))

            result.toString(2) // Pretty printed JSON string
        } catch (e: Exception) {
            Log.e("MemoryClustererBridge", "Error in clustering: ${e.message}", e)
            "{\"status\":\"error\",\"message\":\"${e.message}\"}"
        }
    }

    fun clusterAsMap(memoryEntries: List<String>, numClusters: Int = 3): Pair<Map<Int, List<String>>, Map<Int, String>> {
        val result = clusterMemories(memoryEntries, numClusters)
        val json = JSONObject(result)
        val clustersMap = mutableMapOf<Int, List<String>>()
        val summariesMap = mutableMapOf<Int, String>()

        val clustersJson = json.getJSONObject("clusters")
        val summariesJson = json.getJSONObject("summaries")

        for (key in clustersJson.keys()) {
            val list = mutableListOf<String>()
            val items = clustersJson.getJSONArray(key)
            for (i in 0 until items.length()) {
                list.add(items.getString(i))
            }
            clustersMap[key.toInt()] = list
        }

        for (key in summariesJson.keys()) {
            summariesMap[key.toInt()] = summariesJson.getString(key)
        }

        return clustersMap to summariesMap
    }
}
