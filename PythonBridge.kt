package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.PyObject
import com.chaquo.python.Python

object PythonBridge {
    private val py: Python by lazy { Python.getInstance() }
    private val api: PyObject by lazy { py.getModule("api") }

    fun listModules(): List<String> =
        api.callAttr("list_modules").asList().map { it.toString() }

    fun listFunctions(module: String): List<String> =
        api.callAttr("list_functions", module).asList().map { it.toString() }

    fun call(module: String, function: String, vararg args: Any): String {
        return api.callAttr("call", module, function, *args).toString()
    }
}
