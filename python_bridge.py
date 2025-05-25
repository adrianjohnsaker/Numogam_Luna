package com.antonio.my.ai.girlfriend.free

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * PythonBridge - A high-level API for Python module operations using Chaquopy.
 *
 * This class copies the bundled "python" folder (packaged as an asset)
 * to the app's internal storage and injects its location into Python’s sys.path,
 * making your modules available at runtime.
 *
 * Ensure that your build configuration includes the "python" folder as an asset.
 *
 * This implementation relies on a working ChaquopyBridge which handles conversions
 * from PyObject to Kotlin types as well as module and instance creation.
 */
class PythonBridge private constructor(private val context: Context) {

    // Instance of ChaquopyBridge that wraps Chaquopy functions.
    private val bridge = ChaquopyBridge.getInstance(context)

    // Cache for loaded modules and created class instances.
    private val moduleCache = mutableMapOf<String, PyObject>()
    private val TAG = "PythonBridge"

    companion object {
        @Volatile
        private var instance: PythonBridge? = null

        fun getInstance(context: Context): PythonBridge {
            return instance ?: synchronized(this) {
                instance ?: PythonBridge(context.applicationContext).also { instance = it }
            }
        }
    }

    init {
        // Start Python if it isn't already running.
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        // Copy the bundled 'python' folder from assets into internal storage and inject its path.
        injectPythonModules()
        // Validate the environment by ensuring a critical module exists.
        validateEnvironment()
    }

    /**
     * Inject Python modules by copying the "python" folder from assets into internal storage,
     * then inserting its absolute path into Python’s sys.path.
     */
    private fun injectPythonModules() {
        try {
            // Destination folder in internal storage.
            val destDir = File(context.filesDir, "python")
            if (!destDir.exists()) {
                destDir.mkdirs()
                copyAssetFolder("python", destDir)
            }
            // Inject the destination folder path into the Python module search path.
            val sys = Python.getInstance().getModule("sys")
            sys["path"]?.callAttr("insert", 0, destDir.absolutePath)
            Log.d(TAG, "Python modules injected from: ${destDir.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to inject Python modules: ${e.message}", e)
        }
    }

    /**
     * Recursively copy an asset folder to a destination folder.
     *
     * @param assetFolder The name of the folder in the APK assets.
     * @param dest The destination directory in internal storage.
     */
    private fun copyAssetFolder(assetFolder: String, dest: File) {
        val assetManager = context.assets
        try {
            val files = assetManager.list(assetFolder) ?: return
            for (fileName in files) {
                val assetPath = if (assetFolder.isEmpty()) fileName else "$assetFolder/$fileName"
                val outFile = File(dest, fileName)
                // Check if this assetPath is a folder (it will have contents).
                val subFiles = assetManager.list(assetPath)
                if (subFiles != null && subFiles.isNotEmpty()) {
                    outFile.mkdirs()
                    copyAssetFolder(assetPath, outFile)
                } else {
                    copyAssetFile(assetPath, outFile)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error copying asset folder '$assetFolder' to '${dest.absolutePath}': ${e.message}", e)
        }
    }

    /**
     * Copy an individual asset file to the specified destination file.
     *
     * @param assetPath Path of the asset within the APK.
     * @param outFile Destination file.
     */
    private fun copyAssetFile(assetPath: String, outFile: File) {
        val assetManager = context.assets
        assetManager.open(assetPath).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

    /**
     * Validate the Python environment by printing sys.path and verifying that
     * a critical module (in this case, "numogram") can be imported.
     */
    private fun validateEnvironment() {
        try {
            val sys = Python.getInstance().getModule("sys")
            Log.d(TAG, "Python sys.path: ${sys["path"]}")
            // Replace "numogram" with any module critical to your application.
            Python.getInstance().getModule("numogram")
            Log.d(TAG, "Environment validation succeeded")
        } catch (e: Exception) {
            Log.e(TAG, "Environment validation failed: ${e.message}", e)
            throw RuntimeException("Python environment broken: ${e.message}")
        }
    }

    /**
     * Execute a Python function asynchronously.
     *
     * @param moduleName   The name of the Python module.
     * @param functionName The function within that module.
     * @param args         Arguments to pass to the Python function.
     * @return The result converted to Kotlin or null in case of error.
     */
    suspend fun executeFunction(moduleName: String, functionName: String, vararg args: Any?): Any? {
        return withContext(Dispatchers.IO) {
            try {
                val module = getModuleCached(moduleName)
                val result = module?.callAttr(functionName, *args)
                bridge.pyToKotlin(result)
            } catch (e: Exception) {
                Log.e(TAG, "Error executing $moduleName.$functionName: ${e.message}", e)
                null
            }
        }
    }

    /**
     * Create (and cache) an instance of a Python class defined in the provided module.
     *
     * @param moduleName The module where the class is defined.
     * @param className  The name of the Python class.
     * @param args       Arguments to pass to the class constructor.
     * @return The created Python object instance or null if creation fails.
     */
    suspend fun createInstance(moduleName: String, className: String, vararg args: Any?): PyObject? {
        return withContext(Dispatchers.IO) {
            val cacheKey = "$moduleName.$className(${args.joinToString()})"
            try {
                moduleCache[cacheKey] ?: bridge.createInstance(moduleName, className, *args)?.also {
                    moduleCache[cacheKey] = it
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error creating instance $cacheKey: ${e.message}", e)
                null
            }
        }
    }

    /**
     * Retrieve a Python module using a cache to improve performance.
     *
     * @param moduleName The name of the Python module.
     * @return The Python module or null if loading fails.
     */
    private fun getModuleCached(moduleName: String): PyObject? {
        return moduleCache[moduleName] ?: bridge.getModule(moduleName)?.also {
            moduleCache[moduleName] = it
        }
    }

    /**
     * Clear the cached modules and instances.
     */
    fun clearCache() {
        moduleCache.clear()
    }

    /**
     * Extract a property from a Python object.
     *
     * @param obj      The Python object.
     * @param property The property name to extract.
     * @return The converted property value or null if not available.
     */
    fun extractProperty(obj: PyObject?, property: String): Any? {
        return try {
            if (obj?.hasAttr(property) == true) {
                bridge.pyToKotlin(obj.get(property))
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting property '$property': ${e.message}", e)
            null
        }
    }

    /**
     * Call a method on a Python object.
     *
     * @param obj        The Python object.
     * @param methodName The method name to call.
     * @param args       Arguments for the method.
     * @return The method’s result converted to Kotlin or null on error.
     */
    fun callMethod(obj: PyObject?, methodName: String, vararg args: Any?): Any? {
        return try {
            if (obj?.hasAttr(methodName) == true) {
                bridge.pyToKotlin(obj.callAttr(methodName, *args))
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error calling method '$methodName': ${e.message}", e)
            null
        }
    }

    /**
     * Retrieve the Python sys.path as a list of string values.
     */
    fun getPythonPath(): List<String> {
        return try {
            val sys = Python.getInstance().getModule("sys")
            val pathObj = sys["path"]
            bridge.pyToKotlin(pathObj) as? List<String> ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "Error retrieving Python path: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Check whether a given Python module exists.
     *
     * @param moduleName The name of the Python module.
     * @return True if the module loads successfully, false otherwise.
     */
    fun moduleExists(moduleName: String): Boolean {
        return try {
            Python.getInstance().getModule(moduleName) != null
        } catch (e: Exception) {
            false
        }
    }

    /**
     * (Optional Enhancement) Reload a Python module to update its code.
     *
     * @param moduleName The module to reload.
     * @return The reloaded module or null if reloading fails.
     */
    fun reloadModule(moduleName: String): PyObject? {
        return try {
            // Remove the module from cache.
            moduleCache.remove(moduleName)
            val python = Python.getInstance()
            val importlib = python.getModule("importlib")
            val originalModule = python.getModule(moduleName)
            val reloadedModule = importlib.callAttr("reload", originalModule)
            // Cache and return the reloaded module.
            moduleCache[moduleName] = reloadedModule
            Log.d(TAG, "Module $moduleName reloaded successfully.")
            reloadedModule
        } catch (e: Exception) {
            Log.e(TAG, "Error reloading module $moduleName: ${e.message}", e)
            null
        }
    }
}
