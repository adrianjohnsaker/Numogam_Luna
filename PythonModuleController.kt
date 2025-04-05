// PythonModuleController.kt
package com.antonio.my.ai.girlfriend.free.python

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import java.io.File

/**
 * Handles dynamic loading and caching of Python modules, with version control.
 */
object PythonModuleController {
    private const val TAG = "PythonModuleController"
    private const val MODULE_VERSION = "1.2"
    private const val VERSION_FILE_NAME = "version.txt"

    private var cachedModules = mutableMapOf<String, PyObject>()

    fun initialize(context: Context, moduleNames: List<String>) {
        val python = Python.getInstance()
        val modulesDir = File(context.filesDir, "assets/modules")
        val versionFile = File(modulesDir, VERSION_FILE_NAME)

        if (!versionFile.exists() || versionFile.readText() != MODULE_VERSION) {
            Log.d(TAG, "Extracting modules for version: $MODULE_VERSION")
            // Normally: extractModules(context, moduleNames)
            versionFile.writeText(MODULE_VERSION)
        } else {
            Log.d(TAG, "Modules are up-to-date.")
        }
    }

    fun getModule(name: String): PyObject? {
        if (cachedModules.containsKey(name)) {
            return cachedModules[name]
        }
        return try {
            val module = Python.getInstance().getModule(name)
            cachedModules[name] = module
            module
        } catch (e: Exception) {
            Log.e(TAG, "Error loading module: $name", e)
            null
        }
    }

    fun clearCache() {
        cachedModules.clear()
    }
}
