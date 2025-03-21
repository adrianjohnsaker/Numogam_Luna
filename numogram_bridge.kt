package com.yourapp.bridge

import org.json.JSONObject
import org.json.JSONArray
import com.chaquo.python.PyObject
import com.chaquo.python.Python

class NumogramBridge {

    private val py: Python = Python.getInstance()
    private val module: PyObject = py.getModule("numogram_system")  // Ensure this matches your .py file
    private val numogram: PyObject = module.callAttr("NumogramSystem")

    fun transition(userId: String, currentZone: String, feedback: Double): JSONObject {
        return try {
            val result: PyObject = numogram.callAttr("transition", userId, currentZone, feedback)
            JSONObject(result.toString())
        } catch (e: Exception) {
            JSONObject().apply {
                put("status", "error")
                put("error_message", e.message)
            }
        }
    }

    fun asyncTransition(userId: String, currentZone: String, feedback: Double): JSONObject {
        return try {
            val loop = module.callAttr("asyncio").callAttr("get_event_loop")
            val coroutine = numogram.callAttr("async_process", userId, currentZone, feedback)
            val result: PyObject = loop.callAttr("run_until_complete", coroutine)
            JSONObject(result.toString())
        } catch (e: Exception) {
            JSONObject().apply {
                put("status", "error")
                put("error_message", e.message)
            }
        }
    }

    fun serializeState(): JSONObject {
        return try {
            val jsonStr: String = numogram.callAttr("to_json").toString()
            JSONObject(jsonStr)
        } catch (e: Exception) {
            JSONObject().apply {
                put("status", "error")
                put("error_message", e.message)
            }
        }
    }

    fun loadFromJson(jsonData: String): Boolean {
        return try {
            module.callAttr("NumogramSystem").callAttr("from_json", jsonData)
            true
        } catch (e: Exception) {
            false
        }
    }

    fun clearMemory(): Boolean {
        return try {
            numogram.callAttr("clear_memory")
            true
        } catch (e: Exception) {
            false
        }
    }

    fun cleanup(): Boolean {
        return try {
            numogram.callAttr("cleanup")
            true
        } catch (e: Exception) {
            false
        }
    }

    fun safeExecute(methodName: String, args: Map<String, Any>): JSONObject {
        return try {
            val pyArgs = PyObject.fromJava(args)
            val result: PyObject = numogram.callAttr("safe_execute", methodName, *pyArgs.asList().toTypedArray())
            JSONObject(result.toString())
        } catch (e: Exception) {
            JSONObject().apply {
                put("status", "error")
                put("error_message", e.message)
            }
        }
    }
}
