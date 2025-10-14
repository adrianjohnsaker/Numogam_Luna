package com.antonio.my.ai.girlfriend.free.amelia

import android.content.Context
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * PythonConnector handles the interaction between Kotlin/Java code and Python scripts
 * in the Android app assets folder. It uses the Chaquopy library to execute Python code.
 */
class PythonConnector(private val context: Context) {

    companion object {
        private const val TAG = "PythonConnector"
        private const val PYTHON_MODULE_DIR = "python_modules"
        
        // List of Python modules to extract from assets
        private val REQUIRED_MODULES = listOf(
            "python_hook.py",
            "main_script.py"
            // Add other Python scripts as needed
        )
        
        private var instance: PythonConnector? = null
        
        @Synchronized
        fun getInstance(context: Context): PythonConnector {
            if (instance == null) {
                instance = PythonConnector(context.applicationContext)
            }
            return instance!!
        }
    }

    private var python: Python? = null
    private var initialized = false

    /**
     * Initialize the Python environment and extract necessary modules
     * from assets to internal storage for execution
     */
    fun initialize() {
        if (initialized) return
        
        try {
            // Initialize the Python interpreter if not already done
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            
            python = Python.getInstance()
            
            // Extract Python modules from assets to internal storage
            extractPythonModules()
            
            // Add the modules directory to Python's import path
            val sysModule = python?.getModule("sys")
            val modulesPath = context.getDir(PYTHON_MODULE_DIR, Context.MODE_PRIVATE).absolutePath
            sysModule?.callAttr("path.insert", 0, modulesPath)
            
            initialized = true
            Log.d(TAG, "Python environment initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python environment", e)
            throw RuntimeException("Failed to initialize Python environment", e)
        }
    }

    /**
     * Execute a Python function from a specified module
     * 
     * @param moduleName the name of the Python module (without .py extension)
     * @param functionName the name of the function to call
     * @param args arguments to pass to the Python function
     * @return the result of the Python function call
     */
    fun callPythonFunction(moduleName: String, functionName: String, vararg args: Any?): Any? {
        if (!initialized) {
            initialize()
        }
        
        return try {
            val module = python?.getModule(moduleName)
            if (module == null) {
                Log.e(TAG, "Failed to import Python module: $moduleName")
                throw RuntimeException("Failed to import Python module: $moduleName")
            }
            
            val result = module.callAttr(functionName, *args)
            convertPyObjectToJava(result)
        } catch (e: PyException) {
            Log.e(TAG, "Python error in $moduleName.$functionName: ${e.message}", e)
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error calling Python function $moduleName.$functionName", e)
            throw e
        }
    }
    
    /**
     * Execute a Python script from a string
     * 
     * @param pythonCode the Python code to execute
     * @return the result of the execution (if any)
     */
    fun executePythonCode(pythonCode: String): Any? {
        if (!initialized) {
            initialize()
        }
        
        return try {
            val builtins = python?.getModule("builtins")
            val result = builtins?.callAttr("exec", pythonCode)
            convertPyObjectToJava(result)
        } catch (e: PyException) {
            Log.e(TAG, "Python error in code execution: ${e.message}", e)
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error executing Python code", e)
            throw e
        }
    }

    /**
     * Get a Python module for direct interaction
     * 
     * @param moduleName the name of the Python module to get
     * @return the Python module object
     */
    fun getPythonModule(moduleName: String): PyObject? {
        if (!initialized) {
            initialize()
        }
        
        return try {
            python?.getModule(moduleName)
        } catch (e: PyException) {
            Log.e(TAG, "Python error importing module $moduleName: ${e.message}", e)
            null
        }
    }
    
    /**
     * Extract Python modules from assets to internal storage
     */
    private fun extractPythonModules() {
        val moduleDir = context.getDir(PYTHON_MODULE_DIR, Context.MODE_PRIVATE)
        
        for (moduleName in REQUIRED_MODULES) {
            try {
                val moduleFile = File(moduleDir, moduleName)
                
                // Skip if file exists and is not older than the app installation
                if (moduleFile.exists()) {
                    val appInstallTime = context.packageManager
                        .getPackageInfo(context.packageName, 0).lastUpdateTime
                        
                    if (moduleFile.lastModified() >= appInstallTime) {
                        Log.d(TAG, "Module $moduleName is up to date, skipping extraction")
                        continue
                    }
                }
                
                // Extract module from assets
                context.assets.open(moduleName).use { input ->
                    FileOutputStream(moduleFile).use { output ->
                        input.copyTo(output)
                    }
                }
                
                Log.d(TAG, "Extracted Python module: $moduleName")
            } catch (e: IOException) {
                Log.e(TAG, "Failed to extract Python module: $moduleName", e)
                throw RuntimeException("Failed to extract Python module: $moduleName", e)
            }
        }
    }
    
    /**
     * Convert Python objects to their Java/Kotlin equivalents
     */
    private fun convertPyObjectToJava(pyObject: PyObject?): Any? {
        if (pyObject == null) return null
        
        return try {
            when {
                pyObject.isCallable -> pyObject  // Return callable objects as is
                pyObject.check("__iter__") -> {
                    // Convert iterables to lists
                    val list = mutableListOf<Any?>()
                    val iterator = pyObject.iter()
                    while (true) {
                        try {
                            list.add(convertPyObjectToJava(iterator.next()))
                        } catch (e: PyException) {
                            // StopIteration, end of sequence
                            break
                        }
                    }
                    list
                }
                pyObject.check("__dict__") -> {
                    // Convert objects with __dict__ to maps
                    val dict = mutableMapOf<String, Any?>()
                    val pyDict = pyObject.getAttr("__dict__")
                    val keys = pyDict.callAttr("keys").iter()
                    while (true) {
                        try {
                            val key = keys.next().toString()
                            dict[key] = convertPyObjectToJava(pyDict.callAttr("get", key))
                        } catch (e: PyException) {
                            // StopIteration, end of sequence
                            break
                        }
                    }
                    dict
                }
                else -> pyObject.toJava(Any::class.java) // Default conversion
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to convert Python object to Java, returning as PyObject", e)
            pyObject // Return the original PyObject if conversion fails
        }
    }
    
    /**
     * Class for handling reflection-related operations from Python
     */
    inner class ReflectionBridge {
        /**
         * Get a Kotlin class by its fully qualified name
         */
        fun getKClassByName(className: String): Any? {
            return try {
                val clazz = Class.forName(className)
                clazz.kotlin
            } catch (e: ClassNotFoundException) {
                Log.e(TAG, "Class not found: $className", e)
                null
            }
        }
        
        /**
         * Call a method on an object using reflection
         */
        fun callMethod(obj: Any, methodName: String, vararg args: Any?): Any? {
            return try {
                val methods = obj.javaClass.methods
                val method = methods.firstOrNull { it.name == methodName }
                    ?: throw NoSuchMethodException("Method $methodName not found in ${obj.javaClass.name}")
                
                method.isAccessible = true
                method.invoke(obj, *args)
            } catch (e: Exception) {
                Log.e(TAG, "Error calling method $methodName", e)
                null
            }
        }
        
        /**
         * Get a property value from an object using reflection
         */
        fun getProperty(obj: Any, propertyName: String): Any? {
            return try {
                val property = obj.javaClass.kotlin.members
                    .firstOrNull { it.name == propertyName }
                    ?: throw NoSuchFieldException("Property $propertyName not found in ${obj.javaClass.name}")
                
                if (property is kotlin.reflect.KProperty1<*, *>) {
                    @Suppress("UNCHECKED_CAST")
                    (property as kotlin.reflect.KProperty1<Any, Any?>).get(obj)
                } else {
                    throw IllegalArgumentException("$propertyName is not a property")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting property $propertyName", e)
                null
            }
        }
        
        /**
         * Set a property value on an object using reflection
         */
        fun setProperty(obj: Any, propertyName: String, value: Any?): Boolean {
            return try {
                val property = obj.javaClass.kotlin.members
                    .firstOrNull { it.name == propertyName }
                    ?: throw NoSuchFieldException("Property $propertyName not found in ${obj.javaClass.name}")
                
                if (property is kotlin.reflect.KMutableProperty1<*, *>) {
                    @Suppress("UNCHECKED_CAST")
                    (property as kotlin.reflect.KMutableProperty1<Any, Any?>).set(obj, value)
                    true
                } else {
                    throw IllegalArgumentException("$propertyName is not a mutable property")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error setting property $propertyName", e)
                false
            }
        }
    }

    /**
     * Get a bridge for reflection operations
     */
    fun getReflectionBridge(): ReflectionBridge {
        return ReflectionBridge()
    }
    
    /**
     * Release Python resources when no longer needed
     */
    fun shutdown() {
        python = null
        initialized = false
        Log.d(TAG, "Python connector shut down")
    }
}
