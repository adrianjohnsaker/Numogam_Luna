package com.antonio.my.ai.girlfriend.free.agency.bridge

import org.json.JSONObject

class MythicCausalityBridge {
    fun linkCauseToEffect(cause: String, effect: String): String {
        val payload = JSONObject()
        payload.put("cause", cause)
        payload.put("effect", effect)
        return payload.toString()
    }

    fun parseEngineOutput(response: String): String {
        val json = JSONObject(response)
        val cause = json.getJSONObject("data").getString("cause")
        val effect = json.getJSONObject("data").getString("effect")
        return "Cause: $cause -> Effect: $effect"
    }
}
