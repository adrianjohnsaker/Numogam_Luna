package com.antonio.my.ai.girlfriend.free

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject

/**
 * Bridge to connect Kotlin with the NumogramAI Python module
 */
class NumogramBridge(private val context: Context) {
    private val TAG = "NumogramBridge"
    private var python: Python
    private var numogramModule: PyObject
    private var aiInstance: PyObject
    
    init {
        // Initialize Python
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
            Log.d(TAG, "Python started")
        }
        
        python = Python.getInstance()
        
        // Load the numogram module
        numogramModule = python.getModule("numogram")
        Log.d(TAG, "NumogramAI module loaded")
        
        // Create an instance of NumogramAI
        aiInstance = numogramModule.callAttr("NumogramAI")
        Log.d(TAG, "NumogramAI instance created")
    }
    
    /**
     * Process user input through the NumogramAI
     * @param userInput The text input from the user
     * @return AI response
     */
    fun processInput(userInput: String): String {
        try {
            val response = aiInstance.callAttr("process_input", userInput)
            Log.d(TAG, "Response: ${response.toString()}")
            return response.toString()
        } catch (e: Exception) {
            Log.e(TAG, "Error processing input: ${e.message}")
            return "I'm having trouble processing that. ${e.message}"
        }
    }
    
    /**
     * Get the current state of the NumogramAI
     * @return A map containing the current state
     */
    fun getState(): Map<String, Any> {
        val stateObj = aiInstance.callAttr("get_state")
        return convertPyObjectToMap(stateObj)
    }
    
    /**
     * Get the current zone of the NumogramAI
     * @return The current zone
     */
    fun getCurrentZone(): String {
        return getState()["current_zone"].toString()
    }
    
    /**
     * Get the personality traits of the NumogramAI
     * @return A map of personality traits and their values
     */
    fun getPersonality(): Map<String, Double> {
        val state = getState()
        @Suppress("UNCHECKED_CAST")
        return state["personality"] as Map<String, Double>
    }
    
    /**
     * Force a transition to a new zone
     * @return The new zone
     */
    fun transition(): String {
        val newZone = aiInstance.callAttr("transition")
        return newZone.toString()
    }
    
    /**
     * Helper function to convert PyObject to Kotlin Map
     */
    private fun convertPyObjectToMap(pyObject: PyObject): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        
        for (key in pyObject.asMap().keys) {
            val value = pyObject.asMap()[key]
            
            result[key.toString()] = when {
                value == null -> "null"
                value.toString() == "None" -> "null"
                value.asMap() != null -> convertPyObjectToMap(value)
                value.asList() != null -> value.asList().map { 
                    if (it.asMap() != null) convertPyObjectToMap(it) else it.toString() 
                }
                else -> {
                    try {
                        value.toJava(Double::class.java) ?: value.toString()
                    } catch (e: Exception) {
                        value.toString()
                    }
                }
            }
        }
        
        return result
    }
}
