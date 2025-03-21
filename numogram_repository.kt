package com.antonio.my.ai.girlfriend.free.repository

import com.yourapp.bridge.NumogramBridge
import org.json.JSONObject

class NumogramRepository(private val bridge: NumogramBridge = NumogramBridge()) {

    fun transition(userId: String, currentZone: String, feedback: Double): JSONObject {
        return bridge.transition(userId, currentZone, feedback)
    }

    fun asyncTransition(userId: String, currentZone: String, feedback: Double): JSONObject {
        return bridge.asyncTransition(userId, currentZone, feedback)
    }

    fun getSerializedState(): JSONObject {
        return bridge.serializeState()
    }

    fun loadFromJson(jsonData: String): Boolean {
        return bridge.loadFromJson(jsonData)
    }

    fun resetMemory(): Boolean {
        return bridge.clearMemory()
    }

    fun cleanup(): Boolean {
        return bridge.cleanup()
    }

    fun safeExecute(methodName: String, args: Map<String, Any>): JSONObject {
        return bridge.safeExecute(methodName, args)
    }
}
