package com.antonio.my.ai.girlfriend.free.modules

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

/**
 * Direct access to Python modules through Chaquopy.
 *
 * Just instantiate this once (e.g. in your Application class),
 * then access Python modules and their methods directly.
 */
class PythonModules(context: Context) {

    init {
        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
    }

    private val py: Python = Python.getInstance()

    /**
     * Access any Python module by name.
     *
     * @param moduleName Name of the Python module (filename without .py) under assets/python/
     * @return The PyObject representing the module
     */
    fun getModule(moduleName: String): PyObject {
        return py.getModule(moduleName)
    }

    /**
     * MultiZoneMemory module accessor.
     */
    val memory: MemoryModule by lazy { MemoryModule(getModule("MultiZoneMemory")) }

    /**
     * Example template for another module.
     * Just copy this inner class and update module/class names.
     */
    val yourModule: YourModule by lazy { YourModule(getModule("YourModuleName")) }

    /**
     * Wrapper for the Python MultiZoneMemory class.
     */
    class MemoryModule(private val module: PyObject) {
        private var instance: PyObject = module.callAttr("MultiZoneMemory")

        /** (Re‑)initialize with a specific file path. */
        fun initialize(memoryFile: String) {
            instance = module.callAttr("MultiZoneMemory", memoryFile)
        }

        /** Update memory: returns true on success. */
        fun updateMemory(userId: String, zone: String, info: Any): Boolean {
            return instance.callAttr("update_memory", userId, zone, info).toBoolean()
        }

        /** Retrieve memory from a zone. */
        fun retrieveMemory(userId: String, zone: String): String {
            return instance.callAttr("retrieve_memory", userId, zone).toString()
        }

        /** Raw PyObject instance. */
        fun getInstance(): PyObject = instance
    }

    /**
     * Template for an arbitrary Python module.
     */
    class YourModule(private val module: PyObject) {
        private var instance: PyObject = module.callAttr("YourClassName")

        /** Example method—customize per your Python code. */
        fun yourFunction(param1: String, param2: Int): String {
            return instance.callAttr("your_function", param1, param2).toString()
        }

        /** Raw instance access. */
        fun getInstance(): PyObject = instance
    }
}
