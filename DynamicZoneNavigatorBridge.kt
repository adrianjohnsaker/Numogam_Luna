package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.Python
import org.json.JSONObject

class DynamicZoneNavigatorBridge {

    fun getNextZone(tags: List<String>, reinforcement: Map<Int, Float>): String {
        return try {
            val py = Python.getInstance()
            val module = py.getModule("dynamic_zone_navigator")

            val inputJson = JSONObject().apply {
                put("tags", tags)
                put("reinforcement", JSONObject(reinforcement.mapKeys { it.key.toString() }))
            }

            val result = module.callAttr("navigate_zone", inputJson.toMap())
            JSONObject(result.toString()).toString(2)
        } catch (e: Exception) {
            "{\"status\": \"error\", \"message\": \"${e.message}\"}"
        }
    }

    // Convert JSONObject to Map<String, Any>
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
