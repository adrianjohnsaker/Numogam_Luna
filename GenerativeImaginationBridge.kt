package com.yourai.companion

import org.json.JSONObject
import com.chaquo.python.Python

class GenerativeImaginationBridge {

    fun generateImaginativeResponse(userInput: String, memoryList: List<String>, emotionalTone: String): String {
        return try {
            val py = Python.getInstance()
            val module = py.getModule("generative_imagination_module")

            val inputJson = JSONObject().apply {
                put("user_input", userInput)
                put("memory_elements", memoryList)
                put("emotional_tone", emotionalTone)
            }

            val response = module.callAttr("generate_imaginative_response", inputJson.toMap())
            JSONObject(response.toString()).toString(2) // pretty print
        } catch (e: Exception) {
            "{\"status\": \"error\", \"message\": \"${e.message}\"}"
        }
    }

    // Extension function to convert JSONObject to Python dict
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
