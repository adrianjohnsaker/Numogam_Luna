package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.PyObject
import com.chaquo.python.Python

object PythonBridge {
    private val py: Python by lazy { Python.getInstance() }
    private val api: PyObject by lazy { py.getModule("api") }

    /** Returns all module names. */
    fun listModules(): List<String> {
        return api.callAttr("list_modules").asList().map { it.toString() }
    }

    /** Returns all function names in a module. */
    fun listFunctions(module: String): List<String> {
        return api.callAttr("list_functions", module).asList().map { it.toString() }
    }

    /** Call any function: module, function name, and args. */
    fun call(module: String, function: String, vararg args: Any): String {
        val pyArgs = args.map { it as Any }.toTypedArray()
        return api.callAttr("call", module, function, *pyArgs).toString()
    }
}
