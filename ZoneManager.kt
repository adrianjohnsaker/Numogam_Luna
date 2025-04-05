// ZoneManager.kt
package com.antonio.my.ai.girlfriend.free.zone

/**
 * Handles symbolic and functional transitions between zones in Amelia.
 * Zones can represent different AI modes, moods, behaviors, or modules.
 */
object ZoneManager {
    private var currentZone: String = "default"
    private val listeners = mutableListOf<(String) -> Unit>()

    /**
     * Get the current active zone.
     */
    fun getCurrentZone(): String = currentZone

    /**
     * Set a new zone and notify listeners.
     */
    fun setZone(zone: String) {
        if (zone != currentZone) {
            currentZone = zone
            notifyListeners(zone)
        }
    }

    /**
     * Register a callback to be triggered on zone change.
     */
    fun registerZoneChangeListener(listener: (String) -> Unit) {
        listeners.add(listener)
    }

    private fun notifyListeners(zone: String) {
        listeners.forEach { it.invoke(zone) }
    }

    /**
     * Check if a given zone is active.
     */
    fun isZoneActive(zone: String): Boolean = zone == currentZone
}


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
