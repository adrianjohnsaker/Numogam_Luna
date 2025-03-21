package com.yourapp.bridge

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

class OntologicalRecursionBridge {

    private val py: Python = Python.getInstance()
    private val module: PyObject = py.getModule("ontological_recursion")
    private val instance: PyObject = module.callAttr("OntologicalRecursion")

    fun initialize(numSocieties: Int = 3, occasionsPerSociety: Int = 5): JSONObject {
        return callSafe("initialize", numSocieties, occasionsPerSociety)
    }

    fun update(): JSONObject {
        return callSafe("update")
    }

    fun asyncUpdate(): JSONObject {
        return callSafe("async_update")
    }

    fun getState(): JSONObject {
        return try {
            val json = instance.callAttr("to_json").toString()
            JSONObject(json)
        } catch (e: Exception) {
            errorJson(e)
        }
    }

    fun safeExecute(method: String, args: Map<String, Any>): JSONObject {
        return try {
            val result = instance.callAttr("safe_execute", method, PyObject.fromJava(args))
            JSONObject(result.toString())
        } catch (e: Exception) {
            errorJson(e)
        }
    }

    fun reset() {
        instance.callAttr("clear_history")
    }

    fun cleanup() {
        instance.callAttr("cleanup")
    }

    private fun callSafe(method: String, vararg args: Any): JSONObject {
        return try {
            val result = instance.callAttr(method, *args)
            JSONObject(result.toString())
        } catch (e: Exception) {
            errorJson(e)
        }
    }

    private fun errorJson(e: Exception): JSONObject {
        return JSONObject().apply {
            put("status", "error")
            put("message", e.message ?: "Unknown error")
        }
    }
}
