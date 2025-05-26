package com.universal.python.bridge.examples

import com.universal.python.bridge.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import android.content.Context
import org.json.JSONObject
import android.util.Log

/**
 * Example usage of the UniversalPythonBridge
 */
class PythonBridgeExamples(private val context: Context) {
    private val bridge = UniversalPythonBridge.getInstance(context)
    private val scope = CoroutineScope(Dispatchers.Main)
    
    /**
     * Example 1: Simple module function call
     */
    fun exampleSimpleModuleCall() {
        scope.launch {
            try {
                // Call a simple Python function
                val result = bridge.callModuleFunctionTyped<String>(
                    moduleName = "math_utils",
                    functionName = "calculate_sum",
                    resultType = String::class,
                    10, 20
                )
                Log.d("Example", "Sum result: $result")
                
                // Using inline reified version
                val result2 = bridge.callModuleFunctionTyped<Int>(
                    "math_utils",
                    "multiply",
                    5, 4
                )
                Log.d("Example", "Multiply result: $result2")
            } catch (e: PythonBridgeException) {
                Log.e("Example", "Error: ${e.message}")
            }
        }
    }
    
    /**
     * Example 2: Working with Python classes
     */
    fun exampleClassUsage() {
        scope.launch {
            try {
                // Create an instance of a Python class
                val instanceId = bridge.createInstance(
                    moduleName = "data_processor",
                    className = "DataProcessor",
                    instanceId = "processor_1",
                    "config.json"  // constructor argument
                )
                
                // Call methods on the instance
                val processed = bridge.callInstanceMethodTyped<Map<String, Any>>(
                    instanceId,
                    "process_data",
                    mapOf("input" to "test data")
                )
                Log.d("Example", "Processed data: $processed")
                
                // Get and set attributes
                val status = bridge.getInstanceAttributeTyped<String>(instanceId, "status")
                Log.d("Example", "Status: $status")
                
                bridge.setInstanceAttribute(instanceId, "verbose", true)
                
                // Clean up
                bridge.removeInstance(instanceId)
            } catch (e: PythonBridgeException) {
                Log.e("Example", "Error: ${e.message}")
            }
        }
    }
    
    /**
     * Example 3: Working with complex data types
     */
    fun exampleComplexDataTypes() {
        scope.launch {
            try {
                // Pass and receive lists
                val numbers = listOf(1, 2, 3, 4, 5)
                val doubled = bridge.callModuleFunctionTyped<List<Int>>(
                    "list_utils",
                    "double_numbers",
                    numbers
                )
                Log.d("Example", "Doubled numbers: $doubled")
                
                // Pass and receive maps/dictionaries
                val config = mapOf(
                    "mode" to "production",
                    "debug" to false,
                    "timeout" to 30
                )
                val result = bridge.callModuleFunctionTyped<JSONObject>(
                    "config_processor",
                    "validate_config",
                    config
                )
                Log.d("Example", "Config validation: $result")
            } catch (e: PythonBridgeException) {
                Log.e("Example", "Error: ${e.message}")
            }
        }
    }
    
    /**
     * Example 4: Using module constants
     */
    fun exampleModuleConstants() {
        scope.launch {
            try {
                // Get module constants
                val version = bridge.getModuleConstantTyped<String>("my_module", "VERSION")
                val maxRetries = bridge.getModuleConstantTyped<Int>("my_module", "MAX_RETRIES")
                val supportedFormats = bridge.getModuleConstantTyped<List<String>>("my_module", "SUPPORTED_FORMATS")
                
                Log.d("Example", "Module version: $version")
                Log.d("Example", "Max retries: $maxRetries")
                Log.d("Example", "Supported formats: $supportedFormats")
            } catch (e: PythonBridgeException) {
                Log.e("Example", "Error: ${e.message}")
            }
        }
    }
    
    /**
     * Example 5: Executing raw Python code
     */
    fun exampleRawPythonExecution() {
        scope.launch {
            try {
                val pythonCode = """
                    import numpy as np
                    
                    # Create a simple array
                    arr = np.array([1, 2, 3, 4, 5])
                    
                    # Calculate statistics
                    result = {
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'sum': int(np.sum(arr))
                    }
                """.trimIndent()
                
                val result = bridge.executePythonCode(pythonCode)
                Log.d("Example", "Python execution result: $result")
            } catch (e: PythonBridgeException) {
                Log.e("Example", "Error: ${e.message}")
            }
        }
    }
}

/**
 * Specialized wrapper for common Python module patterns
 */
abstract class PythonModuleWrapper(
    protected val moduleName: String,
    protected val context: Context
) {
    protected val bridge = UniversalPythonBridge.getInstance(context)
    protected var instanceId: String? = null
    
    /**
     * Initialize the module (if it requires initialization)
     */
    open suspend fun initialize(vararg args: Any?): Boolean {
        return try {
            bridge.callModuleFunction(moduleName, "initialize", *args)
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Create an instance if the module uses classes
     */
    protected suspend fun createModuleInstance(className: String, vararg args: Any?): String {
        val id = "${moduleName}_${System.currentTimeMillis()}"
        instanceId = bridge.createInstance(moduleName, className, id, *args)
        return id
    }
    
    /**
     * Clean up resources
     */
    open fun cleanup() {
        instanceId?.let { bridge.removeInstance(it) }
    }
}

/**
 * Example wrapper for the Astral module using the universal bridge
 */
class AstralModuleWrapper(context: Context) : PythonModuleWrapper("astral_symbolic_module", context) {
    
    suspend fun createModule(seed: Int? = null): Boolean {
        return try {
            val id = createModuleInstance("AstralSymbolicModule", seed)
            instanceId = id
            true
        } catch (e: Exception) {
            false
        }
    }
    
    suspend fun initializeSession(sessionName: String? = null): JSONObject? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<JSONObject>(it, "initialize_session", sessionName)
        }
    }
    
    suspend fun mapAstralGlyph(sessionId: String, seed: String = ""): JSONObject? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<JSONObject>(it, "map_astral_glyph", sessionId, seed)
        }
    }
    
    suspend fun getModuleStatus(): JSONObject? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<JSONObject>(it, "get_module_status")
        }
    }
}

/**
 * Factory for creating module wrappers
 */
object PythonModuleFactory {
    fun <T : PythonModuleWrapper> createWrapper(
        moduleType: ModuleType,
        context: Context
    ): T {
        @Suppress("UNCHECKED_CAST")
        return when (moduleType) {
            ModuleType.ASTRAL -> AstralModuleWrapper(context) as T
            // Add more module types as needed
            ModuleType.DATA_PROCESSOR -> DataProcessorWrapper(context) as T
            ModuleType.ML_MODEL -> MLModelWrapper(context) as T
            ModuleType.IMAGE_PROCESSOR -> ImageProcessorWrapper(context) as T
        }
    }
    
    enum class ModuleType {
        ASTRAL,
        DATA_PROCESSOR,
        ML_MODEL,
        IMAGE_PROCESSOR
    }
}

/**
 * Example of another module wrapper
 */
class DataProcessorWrapper(context: Context) : PythonModuleWrapper("data_processor", context) {
    
    suspend fun processCSV(filePath: String): JSONObject? {
        return bridge.callModuleFunctionTyped<JSONObject>(
            moduleName,
            "process_csv_file",
            filePath
        )
    }
    
    suspend fun analyzeData(data: List<Map<String, Any>>): Map<String, Any>? {
        return bridge.callModuleFunctionTyped<Map<String, Any>>(
            moduleName,
            "analyze_data",
            data
        )
    }
}

/**
 * Example ML model wrapper
 */
class MLModelWrapper(context: Context) : PythonModuleWrapper("ml_model", context) {
    
    suspend fun loadModel(modelPath: String): Boolean {
        return try {
            val id = createModuleInstance("MLModel", modelPath)
            instanceId = id
            true
        } catch (e: Exception) {
            false
        }
    }
    
    suspend fun predict(input: List<Float>): List<Float>? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<List<Float>>(it, "predict", input)
        }
    }
    
    suspend fun getModelInfo(): Map<String, Any>? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<Map<String, Any>>(it, "get_model_info")
        }
    }
    
    suspend fun getAccuracy(): Float? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<Float>(it, "get_accuracy")
        }
    }
    
    suspend fun getTrainingStatus(): String? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<String>(it, "get_training_status")
        }
    }
}

/**
 * Image processing module wrapper
 */
class ImageProcessorWrapper(context: Context) : PythonModuleWrapper("image_processor", context) {
    
    suspend fun loadImage(imagePath: String): Boolean {
        return try {
            val id = createModuleInstance("ImageProcessor")
            instanceId = id
            bridge.callInstanceMethod(id, "load_image", imagePath)
            true
        } catch (e: Exception) {
            false
        }
    }
    
    suspend fun applyFilter(filterName: String, params: Map<String, Any> = emptyMap()): String? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<String>(it, "apply_filter", filterName, params)
        }
    }
    
    suspend fun detectObjects(): List<Map<String, Any>>? {
        return instanceId?.let {
            bridge.callInstanceMethodTyped<List<Map<String, Any>>>(it, "detect_objects")
        }
    }
    
    suspend fun getImageDimensions(): Pair<Int, Int>? {
        return instanceId?.let {
            val result = bridge.callInstanceMethodTyped<List<Int>>(it, "get_dimensions")
            result?.let { list -> 
                if (list.size >= 2) Pair(list[0], list[1]) else null 
            }
        }
    }
}

/**
 * Utility extensions for common Python bridge operations
 */
object PythonBridgeUtils {
    
    /**
     * Execute a Python function with automatic error handling and logging
     */
    suspend fun <T> safeExecute(
        bridge: UniversalPythonBridge,
        operation: suspend () -> T,
        onError: ((Exception) -> Unit)? = null
    ): T? {
        return try {
            operation()
        } catch (e: PythonBridgeException) {
            Log.e("PythonBridge", "Bridge error: ${e.message}", e)
            onError?.invoke(e)
            null
        } catch (e: Exception) {
            Log.e("PythonBridge", "Unexpected error: ${e.message}", e)
            onError?.invoke(e)
            null
        }
    }
    
    /**
     * Batch execute multiple Python functions
     */
    suspend fun batchExecute(
        bridge: UniversalPythonBridge,
        operations: List<suspend () -> Any?>
    ): List<Any?> {
        return operations.map { operation ->
            safeExecute(bridge, operation)
        }
    }
    
    /**
     * Check if a module is available
     */
    suspend fun isModuleAvailable(
        bridge: UniversalPythonBridge,
        moduleName: String
    ): Boolean {
        return try {
            bridge.callModuleFunction(moduleName, "__version__", emptyArray())
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Get all available functions in a module
     */
    suspend fun getModuleFunctions(
        bridge: UniversalPythonBridge,
        moduleName: String
    ): List<String>? {
        return safeExecute(bridge) {
            bridge.callModuleFunctionTyped<List<String>>(
                moduleName,
                "get_available_functions"
            )
        }
    }
    
    /**
     * Convert Kotlin data to Python-compatible format
     */
    fun convertToPythonArgs(vararg args: Any?): Array<Any?> {
        return args.map { arg ->
            when (arg) {
                is Map<*, *> -> JSONObject(arg as Map<String, Any>)
                is List<*> -> arg.toTypedArray()
                else -> arg
            }
        }.toTypedArray()
    }
}

/**
 * Performance monitoring for Python bridge operations
 */
class PythonBridgeProfiler {
    private val executionTimes = mutableMapOf<String, MutableList<Long>>()
    
    suspend fun <T> profile(
        operationName: String,
        operation: suspend () -> T
    ): T {
        val startTime = System.currentTimeMillis()
        try {
            return operation()
        } finally {
            val executionTime = System.currentTimeMillis() - startTime
            executionTimes.getOrPut(operationName) { mutableListOf() }.add(executionTime)
            Log.d("PythonBridgeProfiler", "$operationName executed in ${executionTime}ms")
        }
    }
    
    fun getAverageExecutionTime(operationName: String): Double? {
        val times = executionTimes[operationName]
        return times?.let { it.sum().toDouble() / it.size }
    }
    
    fun getExecutionStats(operationName: String): Map<String, Double>? {
        val times = executionTimes[operationName]
        return times?.let {
            mapOf(
                "average" to it.sum().toDouble() / it.size,
                "min" to it.minOrNull()?.toDouble() ?: 0.0,
                "max" to it.maxOrNull()?.toDouble() ?: 0.0,
                "count" to it.size.toDouble()
            )
        }
    }
    
    fun clearStats() {
        executionTimes.clear()
    }
}

/**
 * Configuration helper for Python bridge settings
 */
data class PythonBridgeConfig(
    val defaultTimeout: Long = 30000L,
    val maxInstances: Int = 50,
    val enableProfiling: Boolean = false,
    val logLevel: LogLevel = LogLevel.INFO,
    val retryAttempts: Int = 3,
    val retryDelay: Long = 1000L
) {
    enum class LogLevel { DEBUG, INFO, WARN, ERROR, NONE }
}

/**
 * Advanced Python bridge manager with configuration and monitoring
 */
class PythonBridgeManager private constructor(
    private val context: Context,
    private val config: PythonBridgeConfig
) {
    private val bridge = UniversalPythonBridge.getInstance(context)
    private val profiler = if (config.enableProfiling) PythonBridgeProfiler() else null
    private val activeInstances = mutableSetOf<String>()
    
    companion object {
        @Volatile
        private var INSTANCE: PythonBridgeManager? = null
        
        fun getInstance(
            context: Context,
            config: PythonBridgeConfig = PythonBridgeConfig()
        ): PythonBridgeManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: PythonBridgeManager(context, config).also { INSTANCE = it }
            }
        }
    }
    
    suspend fun <T> executeWithRetry(
        operation: suspend () -> T,
        attempts: Int = config.retryAttempts
    ): T? {
        repeat(attempts) { attempt ->
            try {
                return if (profiler != null) {
                    profiler.profile("operation_attempt_$attempt") { operation() }
                } else {
                    operation()
                }
            } catch (e: Exception) {
                if (attempt == attempts - 1) {
                    Log.e("PythonBridgeManager", "Operation failed after $attempts attempts", e)
                    return null
                }
                kotlinx.coroutines.delay(config.retryDelay)
            }
        }
        return null
    }
    
    suspend fun createManagedInstance(
        moduleName: String,
        className: String,
        vararg args: Any?
    ): String? {
        if (activeInstances.size >= config.maxInstances) {
            Log.w("PythonBridgeManager", "Maximum instances reached")
            return null
        }
        
        return executeWithRetry {
            val instanceId = "${moduleName}_${className}_${System.currentTimeMillis()}"
            bridge.createInstance(moduleName, className, instanceId, *args)
            activeInstances.add(instanceId)
            instanceId
        }
    }
    
    fun removeManagedInstance(instanceId: String) {
        bridge.removeInstance(instanceId)
        activeInstances.remove(instanceId)
    }
    
    fun getProfilerStats(): Map<String, Map<String, Double>>? {
        return profiler?.let { prof ->
            activeInstances.associateWith { instanceId ->
                prof.getExecutionStats(instanceId) ?: emptyMap()
            }
        }
    }
    
    fun cleanup() {
        activeInstances.forEach { bridge.removeInstance(it) }
        activeInstances.clear()
        profiler?.clearStats()
    }
}
```
