package com.antonio.my.ai.girlfriend.free.universal.python.bridge

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.json.JSONArray
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.ConcurrentHashMap
import kotlin.reflect.KClass

/**
 * UniversalPythonBridge - A flexible bridge to interact with any Python module
 * This bridge uses Chaquopy to communicate with Python modules dynamically
 */
class UniversalPythonBridge private constructor(private val context: Context) {
    private val TAG = "UniversalPythonBridge"
    
    // Python instance
    private val py by lazy { Python.getInstance() }
    
    // Module cache
    private val moduleCache = ConcurrentHashMap<String, PyObject>()
    
    // Module instances cache
    private val instanceCache = ConcurrentHashMap<String, PyObject>()
    
    // Type converters
    private val typeConverters = TypeConverters()
    
    companion object {
        @Volatile
        private var INSTANCE: UniversalPythonBridge? = null
        
        fun getInstance(context: Context): UniversalPythonBridge {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: UniversalPythonBridge(context).also {
                    INSTANCE = it
                }
            }
        }
        
        fun initialize(context: Context) {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
        }
    }
    
    /**
     * Get or load a Python module
     * @param moduleName Name of the Python module
     * @return PyObject representing the module
     */
    @Throws(PythonBridgeException::class)
    private fun getModule(moduleName: String): PyObject {
        return moduleCache.getOrPut(moduleName) {
            try {
                py.getModule(moduleName)
            } catch (e: Exception) {
                throw PythonBridgeException("Failed to load module: $moduleName", e)
            }
        }
    }
    
    /**
     * Call a module-level function
     * @param moduleName Name of the Python module
     * @param functionName Name of the function to call
     * @param args Arguments to pass to the function
     * @return Result as PyObject
     */
    suspend fun callModuleFunction(
        moduleName: String,
        functionName: String,
        vararg args: Any?
    ): PyObject = withContext(Dispatchers.IO) {
        try {
            val module = getModule(moduleName)
            val convertedArgs = args.map { typeConverters.convertToPython(it) }.toTypedArray()
            module.callAttr(functionName, *convertedArgs)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to call $moduleName.$functionName: ${e.message}")
            throw PythonBridgeException("Failed to call $moduleName.$functionName", e)
        }
    }
    
    /**
     * Call a module-level function and convert result to Kotlin type
     * @param moduleName Name of the Python module
     * @param functionName Name of the function to call
     * @param resultType Expected Kotlin type of the result
     * @param args Arguments to pass to the function
     * @return Result converted to Kotlin type
     */
    suspend fun <T : Any> callModuleFunctionTyped(
        moduleName: String,
        functionName: String,
        resultType: KClass<T>,
        vararg args: Any?
    ): T = withContext(Dispatchers.IO) {
        val result = callModuleFunction(moduleName, functionName, *args)
        typeConverters.convertFromPython(result, resultType)
    }
    
    /**
     * Create an instance of a Python class
     * @param moduleName Name of the Python module
     * @param className Name of the class to instantiate
     * @param instanceId Unique identifier for this instance
     * @param args Arguments to pass to the constructor
     * @return Instance identifier
     */
    suspend fun createInstance(
        moduleName: String,
        className: String,
        instanceId: String,
        vararg args: Any?
    ): String = withContext(Dispatchers.IO) {
        try {
            val module = getModule(moduleName)
            val classObj = module.get(className) ?: throw PythonBridgeException("Class $className not found in module $moduleName")
            val convertedArgs = args.map { typeConverters.convertToPython(it) }.toTypedArray()
            val instance = classObj.callAttr("__call__", *convertedArgs)
            instanceCache[instanceId] = instance
            Log.d(TAG, "Created instance: $instanceId of $moduleName.$className")
            instanceId
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create instance of $moduleName.$className: ${e.message}")
            throw PythonBridgeException("Failed to create instance of $moduleName.$className", e)
        }
    }
    
    /**
     * Call a method on a Python instance
     * @param instanceId Instance identifier
     * @param methodName Name of the method to call
     * @param args Arguments to pass to the method
     * @return Result as PyObject
     */
    suspend fun callInstanceMethod(
        instanceId: String,
        methodName: String,
        vararg args: Any?
    ): PyObject = withContext(Dispatchers.IO) {
        val instance = instanceCache[instanceId] 
            ?: throw PythonBridgeException("Instance not found: $instanceId")
        try {
            val convertedArgs = args.map { typeConverters.convertToPython(it) }.toTypedArray()
            instance.callAttr(methodName, *convertedArgs)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to call $instanceId.$methodName: ${e.message}")
            throw PythonBridgeException("Failed to call $instanceId.$methodName", e)
        }
    }
    
    /**
     * Call a method on a Python instance and convert result to Kotlin type
     * @param instanceId Instance identifier
     * @param methodName Name of the method to call
     * @param resultType Expected Kotlin type of the result
     * @param args Arguments to pass to the method
     * @return Result converted to Kotlin type
     */
    suspend fun <T : Any> callInstanceMethodTyped(
        instanceId: String,
        methodName: String,
        resultType: KClass<T>,
        vararg args: Any?
    ): T = withContext(Dispatchers.IO) {
        val result = callInstanceMethod(instanceId, methodName, *args)
        typeConverters.convertFromPython(result, resultType)
    }
    
    /**
     * Get an attribute from a Python instance
     * @param instanceId Instance identifier
     * @param attributeName Name of the attribute
     * @return Attribute value as PyObject
     */
    suspend fun getInstanceAttribute(
        instanceId: String,
        attributeName: String
    ): PyObject = withContext(Dispatchers.IO) {
        val instance = instanceCache[instanceId] 
            ?: throw PythonBridgeException("Instance not found: $instanceId")
        try {
            instance.get(attributeName) ?: throw PythonBridgeException("Attribute $attributeName not found")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get attribute $instanceId.$attributeName: ${e.message}")
            throw PythonBridgeException("Failed to get attribute $instanceId.$attributeName", e)
        }
    }
    
    /**
     * Get an attribute from a Python instance and convert to Kotlin type
     * @param instanceId Instance identifier
     * @param attributeName Name of the attribute
     * @param resultType Expected Kotlin type of the result
     * @return Attribute value converted to Kotlin type
     */
    suspend fun <T : Any> getInstanceAttributeTyped(
        instanceId: String,
        attributeName: String,
        resultType: KClass<T>
    ): T = withContext(Dispatchers.IO) {
        val result = getInstanceAttribute(instanceId, attributeName)
        typeConverters.convertFromPython(result, resultType)
    }
    
    /**
     * Set an attribute on a Python instance
     * @param instanceId Instance identifier
     * @param attributeName Name of the attribute
     * @param value Value to set
     */
    suspend fun setInstanceAttribute(
        instanceId: String,
        attributeName: String,
        value: Any?
    ) = withContext(Dispatchers.IO) {
        val instance = instanceCache[instanceId] 
            ?: throw PythonBridgeException("Instance not found: $instanceId")
        try {
            val convertedValue = typeConverters.convertToPython(value)
            instance.put(attributeName, convertedValue)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set attribute $instanceId.$attributeName: ${e.message}")
            throw PythonBridgeException("Failed to set attribute $instanceId.$attributeName", e)
        }
    }
    
    /**
     * Get a module-level constant or variable
     * @param moduleName Name of the Python module
     * @param constantName Name of the constant
     * @return Constant value as PyObject
     */
    suspend fun getModuleConstant(
        moduleName: String,
        constantName: String
    ): PyObject = withContext(Dispatchers.IO) {
        try {
            val module = getModule(moduleName)
            module.get(constantName) ?: throw PythonBridgeException("Constant $constantName not found in module $moduleName")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get constant $moduleName.$constantName: ${e.message}")
            throw PythonBridgeException("Failed to get constant $moduleName.$constantName", e)
        }
    }
    
    /**
     * Get a module-level constant or variable and convert to Kotlin type
     * @param moduleName Name of the Python module
     * @param constantName Name of the constant
     * @param resultType Expected Kotlin type of the result
     * @return Constant value converted to Kotlin type
     */
    suspend fun <T : Any> getModuleConstantTyped(
        moduleName: String,
        constantName: String,
        resultType: KClass<T>
    ): T = withContext(Dispatchers.IO) {
        val result = getModuleConstant(moduleName, constantName)
        typeConverters.convertFromPython(result, resultType)
    }
    
    /**
     * Execute raw Python code
     * @param code Python code to execute
     * @return Result of the last expression or null
     */
    suspend fun executePythonCode(code: String): PyObject? = withContext(Dispatchers.IO) {
        try {
            val builtins = py.getBuiltins()
            val globals = builtins.callAttr("dict")
            val locals = builtins.callAttr("dict")
            
            // Make modules available in the execution context
            moduleCache.forEach { (name, module) ->
                globals.put(name, module)
            }
            
            builtins.callAttr("exec", code, globals, locals)
            
            // Try to get the result if it's stored in a variable named 'result'
            locals.get("result")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to execute Python code: ${e.message}")
            throw PythonBridgeException("Failed to execute Python code", e)
        }
    }
    
    /**
     * Check if an instance exists
     * @param instanceId Instance identifier
     * @return true if instance exists
     */
    fun hasInstance(instanceId: String): Boolean = instanceCache.containsKey(instanceId)
    
    /**
     * Remove an instance from cache
     * @param instanceId Instance identifier
     */
    fun removeInstance(instanceId: String) {
        instanceCache.remove(instanceId)
        Log.d(TAG, "Removed instance: $instanceId")
    }
    
    /**
     * Clear all caches
     */
    fun clearCaches() {
        instanceCache.clear()
        moduleCache.clear()
        Log.d(TAG, "All caches cleared")
    }
    
    /**
     * Get statistics about the bridge
     * @return JSONObject with statistics
     */
    fun getStatistics(): JSONObject {
        return JSONObject().apply {
            put("loadedModules", moduleCache.size)
            put("activeInstances", instanceCache.size)
            put("modules", JSONArray(moduleCache.keys.toList()))
            put("instances", JSONArray(instanceCache.keys.toList()))
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        clearCaches()
        INSTANCE = null
        Log.d(TAG, "UniversalPythonBridge cleanup complete")
    }
}

/**
 * Type converters for Python <-> Kotlin conversion
 */
class TypeConverters {
    private val py by lazy { Python.getInstance() }
    
    /**
     * Convert Kotlin object to Python object
     */
    fun convertToPython(value: Any?): PyObject? {
        return when (value) {
            null -> null
            is PyObject -> value
            is String -> py.builtins.callAttr("str", value)
            is Int -> py.builtins.callAttr("int", value)
            is Long -> py.builtins.callAttr("int", value)
            is Float -> py.builtins.callAttr("float", value)
            is Double -> py.builtins.callAttr("float", value)
            is Boolean -> if (value) py.builtins.get("True") else py.builtins.get("False")
            is List<*> -> {
                val pyList = py.builtins.callAttr("list")
                value.forEach { item ->
                    pyList.callAttr("append", convertToPython(item))
                }
                pyList
            }
            is Map<*, *> -> {
                val pyDict = py.builtins.callAttr("dict")
                value.forEach { (k, v) ->
                    pyDict.put(k.toString(), convertToPython(v))
                }
                pyDict
            }
            is JSONObject -> {
                val pyDict = py.builtins.callAttr("dict")
                value.keys().forEach { key ->
                    pyDict.put(key, convertToPython(value.get(key)))
                }
                pyDict
            }
            is JSONArray -> {
                val pyList = py.builtins.callAttr("list")
                for (i in 0 until value.length()) {
                    pyList.callAttr("append", convertToPython(value.get(i)))
                }
                pyList
            }
            else -> throw IllegalArgumentException("Unsupported type: ${value::class.simpleName}")
        }
    }
    
    /**
     * Convert Python object to Kotlin type
     */
    @Suppress("UNCHECKED_CAST")
    fun <T : Any> convertFromPython(pyObject: PyObject?, targetType: KClass<T>): T {
        if (pyObject == null) {
            throw IllegalArgumentException("Cannot convert null PyObject")
        }
        
        return when (targetType) {
            String::class -> pyObject.toString() as T
            Int::class -> pyObject.toInt() as T
            Long::class -> pyObject.toLong() as T
            Float::class -> pyObject.toFloat() as T
            Double::class -> pyObject.toDouble() as T
            Boolean::class -> pyObject.toBoolean() as T
            List::class -> pyObject.asList().map { convertFromPythonAny(it) } as T
            Map::class -> {
                val map = mutableMapOf<String, Any?>()
                val keys = pyObject.callAttr("keys").asList()
                keys.forEach { key ->
                    val value = pyObject.get(key.toString())
                    map[key.toString()] = convertFromPythonAny(value)
                }
                map as T
            }
            JSONObject::class -> {
                val jsonStr = pyObject.toString()
                JSONObject(jsonStr) as T
            }
            JSONArray::class -> {
                val jsonStr = pyObject.toString()
                JSONArray(jsonStr) as T
            }
            PyObject::class -> pyObject as T
            else -> throw IllegalArgumentException("Unsupported target type: ${targetType.simpleName}")
        }
    }
    
    /**
     * Convert Python object to appropriate Kotlin type (best guess)
     */
    private fun convertFromPythonAny(pyObject: PyObject?): Any? {
        if (pyObject == null) return null
        
        return try {
            when {
                isPythonNone(pyObject) -> null
                isPythonBool(pyObject) -> pyObject.toBoolean()
                isPythonInt(pyObject) -> pyObject.toLong()
                isPythonFloat(pyObject) -> pyObject.toDouble()
                isPythonString(pyObject) -> pyObject.toString()
                isPythonList(pyObject) -> pyObject.asList().map { convertFromPythonAny(it) }
                isPythonDict(pyObject) -> {
                    val map = mutableMapOf<String, Any?>()
                    val keys = pyObject.callAttr("keys").asList()
                    keys.forEach { key ->
                        val value = pyObject.get(key.toString())
                        map[key.toString()] = convertFromPythonAny(value)
                    }
                    map
                }
                else -> pyObject.toString()
            }
        } catch (e: Exception) {
            pyObject.toString()
        }
    }
    
    private fun isPythonNone(obj: PyObject): Boolean = obj.toString() == "None"
    private fun isPythonBool(obj: PyObject): Boolean = obj.toString() in listOf("True", "False")
    private fun isPythonInt(obj: PyObject): Boolean = try { obj.toInt(); true } catch (e: Exception) { false }
    private fun isPythonFloat(obj: PyObject): Boolean = try { obj.toFloat(); true } catch (e: Exception) { false }
    private fun isPythonString(obj: PyObject): Boolean = try { obj.toString(); true } catch (e: Exception) { false }
    private fun isPythonList(obj: PyObject): Boolean = try { obj.asList(); true } catch (e: Exception) { false }
    private fun isPythonDict(obj: PyObject): Boolean = try { obj.callAttr("keys"); true } catch (e: Exception) { false }
}

/**
 * Custom exception for Python bridge errors
 */
class PythonBridgeException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Extension functions for convenience
 */

// String extensions for module operations
suspend fun String.callPythonFunction(functionName: String, vararg args: Any?): PyObject =
    UniversalPythonBridge.getInstance(context = TODO("Provide context")).callModuleFunction(this, functionName, *args)

suspend fun <T : Any> String.callPythonFunctionTyped(functionName: String, resultType: KClass<T>, vararg args: Any?): T =
    UniversalPythonBridge.getInstance(context = TODO("Provide context")).callModuleFunctionTyped(this, functionName, resultType, *args)

// Inline reified versions for cleaner syntax
suspend inline fun <reified T : Any> UniversalPythonBridge.callModuleFunctionTyped(
    moduleName: String,
    functionName: String,
    vararg args: Any?
): T = callModuleFunctionTyped(moduleName, functionName, T::class, *args)

suspend inline fun <reified T : Any> UniversalPythonBridge.callInstanceMethodTyped(
    instanceId: String,
    methodName: String,
    vararg args: Any?
): T = callInstanceMethodTyped(instanceId, methodName, T::class, *args)

suspend inline fun <reified T : Any> UniversalPythonBridge.getInstanceAttributeTyped(
    instanceId: String,
    attributeName: String
): T = getInstanceAttributeTyped(instanceId, attributeName, T::class)

suspend inline fun <reified T : Any> UniversalPythonBridge.getModuleConstantTyped(
    moduleName: String,
    constantName: String
): T = getModuleConstantTyped(moduleName, constantName, T::class)

/**
 * Extension functions for convenience
 */

// String extensions for module operations
suspend fun String.callPythonFunction(functionName: String, vararg args: Any?): PyObject =
    UniversalPythonBridge.getInstance(context = TODO("Provide context")).callModuleFunction(this, functionName, *args)

suspend fun <T : Any> String.callPythonFunctionTyped(functionName: String, resultType: KClass<T>, vararg args: Any?): T =
    UniversalPythonBridge.getInstance(context = TODO("Provide context")).callModuleFunctionTyped(this, functionName, resultType, *args)

// Inline reified versions for cleaner syntax
suspend inline fun <reified T : Any> UniversalPythonBridge.callModuleFunctionTyped(
    moduleName: String,
    functionName: String,
    vararg args: Any?
): T = callModuleFunctionTyped(moduleName, functionName, T::class, *args)

suspend inline fun <reified T : Any> UniversalPythonBridge.callInstanceMethodTyped(
    instanceId: String,
    methodName: String,
    vararg args: Any?
): T = callInstanceMethodTyped(instanceId, methodName, T::class, *args)

suspend inline fun <reified T : Any> UniversalPythonBridge.getInstanceAttributeTyped(
    instanceId: String,
    attributeName: String
): T = getInstanceAttributeTyped(instanceId, attributeName, T::class)

suspend inline fun <reified T : Any> UniversalPythonBridge.getModuleConstantTyped(
    moduleName: String,
    constantName: String
): T = getModuleConstantTyped(moduleName, constantName, T::class)
```
