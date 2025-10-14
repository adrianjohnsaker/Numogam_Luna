package com.antonio.my.ai.girlfriend.free.amelia.bridge

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONObject

object PythonModuleController {
    fun runSystemVisualizer(metricsJson: String): JSONObject {
        val py = Python.getInstance()
        val module = py.getModule("system_visualizer")
        val data = JSONObject(metricsJson)
        val pyDict = PyObject.fromJava(data.toMap())
        val result = module.callAttr("run_visualizer", pyDict, "/sdcard/Download").toString()
        return JSONObject(result)
    }
}
