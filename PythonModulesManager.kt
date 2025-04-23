package com.antonio.my.ai.girlfriend.free

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.chaquo.python.PyObject

/**
 * Single entry point for Chaquopy-based Python integration.
 * Drop this into your Kotlin sources—no further setup needed.
 */
class PythonModules(context: Context) {
    init {
        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
    }

    private val py: Python = Python.getInstance()

    /** Access any Python module by name. */
    fun getModule(moduleName: String): PyObject =
        py.getModule(moduleName)

    /** MultiZoneMemory module wrapper. */
    val memory = MemoryModule()

    inner class MemoryModule {
        private val module: PyObject = getModule("MultiZoneMemory")
        private var instance: PyObject = module.callAttr("MultiZoneMemory")

        fun initialize(memoryFile: String) {
            instance = module.callAttr("MultiZoneMemory", memoryFile)
        }

        fun updateMemory(userId: String, zone: String, info: Any): Boolean =
            instance.callAttr("update_memory", userId, zone, info).toBoolean()

        fun retrieveMemory(userId: String, zone: String): String =
            instance.callAttr("retrieve_memory", userId, zone).toString()
    }

    /** Example: another module wrapper. */
    val api = ApiModule()

    inner class ApiModule {
        private val module: PyObject = getModule("api")
        fun greet(name: String): String =
            module.callAttr("greet", name).toString()

        fun add(a: Int, b: Int): Int =
            module.callAttr("add", a, b).toInt()
    }
}
