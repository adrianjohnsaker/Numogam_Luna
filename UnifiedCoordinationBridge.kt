package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class UnifiedCoordinatorBridge {

    private val py: Python = Python.getInstance()
    private val coordinatorModule: PyObject = py.getModule("unified_coordinator_module")

    fun coordinateInteraction(
        userInput: String,
        memoryElements: List<String>,
        emotionalTone: String,
        tags: List<String>,
        reinforcement: Map<Int, Float>
    ): JSONObject {
        val inputData = JSONObject().apply {
            put("user_input", userInput)
            put("memory_elements", memoryElements)
            put("emotional_tone", emotionalTone)
            put("tags", tags)
            put("reinforcement", JSONObject(reinforcement))
        }

        val result: PyObject = coordinatorModule.callAttr("coordinate_modules", inputData)
        return JSONObject(result.toString())
    }
}
