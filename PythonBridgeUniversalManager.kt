package com.antonio.my.ai.girlfriend.free.universal.python.bridge.config

import android.content.Context
import com.universal.python.bridge.UniversalPythonBridge
import com.universal.python.bridge.PythonBridgeException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

/**
 * Configuration class for Python modules
 */
data class PythonModuleConfig(
    val moduleName: String,
    val autoInitialize: Boolean = false,
    val initializationArgs: List<Any?> = emptyList(),
    val requiredPythonPackages: List<String> = emptyList(),
    val moduleVersion: String? = null
)

/**
 * Python Bridge Manager - Handles initialization and lifecycle
 */
class PythonBridgeManager(private val context: Context) {
    private val bridge = UniversalPythonBridge.getInstance(context)
    private val moduleConfigs = mutableMapOf<String, PythonModuleConfig>()
    private val scope = CoroutineScope(Dispatchers.Main)
    
    companion object {
        private var instance: PythonBridgeManager? = null
        
        fun getInstance(context: Context): PythonBridgeManager {
            return instance ?: synchronized(this) {
                instance ?: PythonBridgeManager(context).also { instance = it }
            }
        }
    }
    
    /**
     * Initialize the Python environment
     */
    fun initialize() {
        UniversalPythonBridge.initialize(context)
        
        // Copy Python modules from assets if needed
        scope.launch {
            copyPythonModulesFromAssets()
            initializeConfiguredModules()
        }
    }
    
    /**
     * Register a module configuration
     */
    fun registerModule(config: PythonModuleConfig) {
        moduleConfigs[config.moduleName] = config
    }
    
    /**
     * Register multiple module configurations
     */
    fun registerModules(vararg configs: PythonModuleConfig) {
        configs.forEach { registerModule(it) }
    }
    
    /**
     * Initialize all registered modules that have autoInitialize = true
     */
    private suspend fun initializeConfiguredModules() = withContext(Dispatchers.IO) {
        moduleConfigs.values
            .filter { it.autoInitialize }
            .forEach { config ->
                try {
                    bridge.callModuleFunction(
                        config.moduleName,
                        "initialize",
                        *config.initializationArgs.toTypedArray()
                    )
                } catch (e: Exception) {
                    // Module might not have an initialize function
                }
            }
    }
    
    /**
     * Copy Python modules from assets to internal storage
     */
    private suspend fun copyPythonModulesFromAssets() = withContext(Dispatchers.IO) {
        try {
            val pythonDir = File(context.filesDir, "python")
            if (!pythonDir.exists()) {
                pythonDir.mkdirs()
            }
            
            // Copy all .py files from assets/python to internal storage
            context.assets.list("python")?.forEach { filename ->
                if (filename.endsWith(".py")) {
                    context.assets.open("python/$filename").use { input ->
                        File(pythonDir, filename).outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            // Assets might not exist or other IO error
        }
    }
    
    /**
     * Check if all required packages are installed
     */
    suspend fun checkRequiredPackages(): Map<String, Boolean> = withContext(Dispatchers.IO) {
        val results = mutableMapOf<String, Boolean>()
        
        moduleConfigs.values.forEach { config ->
            config.requiredPythonPackages.forEach { packageName ->
                results[packageName] = checkPackageInstalled(packageName)
            }
        }
        
        results
    }
    
    /**
     * Check if a Python package is installed
     */
    private suspend fun checkPackageInstalled(packageName: String): Boolean {
        return try {
            val code = """
                import importlib
                try:
                    importlib.import_module('$packageName')
                    result = True
                except ImportError:
                    result = False
            """.trimIndent()
            
            val result = bridge.executePythonCode(code)
            result?.toBoolean() ?: false
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Get module configuration
     */
    fun getModuleConfig(moduleName: String): PythonModuleConfig? {
        return moduleConfigs[moduleName]
    }
    
    /**
     * Get bridge statistics
     */
    fun getStatistics(): JSONObject {
        return bridge.getStatistics().apply {
            put("registeredModules", moduleConfigs.size)
            put("modules", moduleConfigs.keys.toList())
        }
    }
}

/**
 * DSL for configuring Python modules
 */
class PythonModuleConfigBuilder {
    var moduleName: String = ""
    var autoInitialize: Boolean = false
    var initializationArgs: List<Any?> = emptyList()
    var requiredPythonPackages: List<String> = emptyList()
    var moduleVersion: String? = null
    
    fun build(): PythonModuleConfig {
        require(moduleName.isNotEmpty()) { "Module name must be specified" }
        return PythonModuleConfig(
            moduleName = moduleName,
            autoInitialize = autoInitialize,
            initializationArgs = initializationArgs,
            requiredPythonPackages = requiredPythonPackages,
            moduleVersion = moduleVersion
        )
    }
}

/**
 * DSL function for creating module configurations
 */
fun pythonModule(block: PythonModuleConfigBuilder.() -> Unit): PythonModuleConfig {
    return PythonModuleConfigBuilder().apply(block).build()
}

/**
 * Batch operations helper
 */
class PythonBridgeBatchOperations(private val bridge: UniversalPythonBridge) {
    
    /**
     * Execute multiple module functions in parallel
     */
    suspend fun <T : Any> executeBatch(
        operations: List<BatchOperation<T>>
    ): List<Result<T>> = withContext(Dispatchers.IO) {
        operations.map { operation ->
            try {
                val result = when (operation) {
                    is BatchOperation.ModuleFunction -> {
                        bridge.callModuleFunctionTyped(
                            operation.moduleName,
                            operation.functionName,
                            operation.resultType,
                            *operation.args
                        )
                    }
                    is BatchOperation.InstanceMethod -> {
                        bridge.callInstanceMethodTyped(
                            operation.instanceId,
                            operation.methodName,
                            operation.resultType,
                            *operation.args
                        )
                    }
                }
                Result.success(result)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }
    
    /**
     * Batch operation types
     */
    sealed class BatchOperation<T : Any> {
        abstract val resultType: kotlin.reflect.KClass<T>
        
        data class ModuleFunction<T : Any>(
            val moduleName: String,
            val functionName: String,
            override val resultType: kotlin.reflect.KClass<T>,
            val args: Array<Any?>
        ) : BatchOperation<T>()
        
        data class InstanceMethod<T : Any>(
            val instanceId: String,
            val methodName: String,
            override val resultType: kotlin.reflect.KClass<T>,
            val args: Array<Any?>
        ) : BatchOperation<T>()
    }
}

/**
 * Error handling utilities
 */
object PythonBridgeErrorHandler {
    
    /**
     * Wrap Python calls with error handling and retry logic
     */
    suspend fun <T> withErrorHandling(
        retries: Int = 3,
        delayMs: Long = 1000,
        onError: ((Exception) -> Unit)? = null,
        block: suspend () -> T
    ): T? {
        var lastException: Exception? = null
        
        repeat(retries) { attempt ->
            try {
                return block()
            } catch (e: Exception) {
                lastException = e
                onError?.invoke(e)
                
                if (attempt < retries - 1) {
                    kotlinx.coroutines.delay(delayMs)
                }
            }
        }
        
        throw lastException ?: PythonBridgeException("Unknown error occurred")
    }
    
    /**
     * Parse Python error messages
     */
    fun parsePythonError(error: String): PythonErrorInfo {
        val tracebackRegex = """File "(.+)", line (\d+), in (.+)""".toRegex()
        val errorTypeRegex = """(\w+Error): (.+)""".toRegex()
        
        val tracebackMatch = tracebackRegex.find(error)
        val errorTypeMatch = errorTypeRegex.find(error)
        
        return PythonErrorInfo(
            file = tracebackMatch?.groupValues?.get(1),
            line = tracebackMatch?.groupValues?.get(2)?.toIntOrNull(),
            function = tracebackMatch?.groupValues?.get(3),
            errorType = errorTypeMatch?.groupValues?.get(1),
            errorMessage = errorTypeMatch?.groupValues?.get(2) ?: error
        )
    }
    
    data class PythonErrorInfo(
        val file: String?,
        val line: Int?,
        val function: String?,
        val errorType: String?,
        val errorMessage: String
    )
}

/**
 * Performance monitoring
 */
class PythonBridgePerformanceMonitor {
    private val callTimes = mutableMapOf<String, MutableList<Long>>()
    
    /**
     * Measure execution time
     */
    suspend fun <T> measureTime(
        operationName: String,
        block: suspend () -> T
    ): T {
        val startTime = System.currentTimeMillis()
        val result = block()
        val endTime = System.currentTimeMillis()
        
        val times = callTimes.getOrPut(operationName) { mutableListOf() }
        times.add(endTime - startTime)
        
        // Keep only last 100 measurements
        if (times.size > 100) {
            times.removeAt(0)
        }
        
        return result
    }
    
    /**
     * Get performance statistics
     */
    fun getStatistics(): Map<String, PerformanceStats> {
        return callTimes.mapValues { (_, times) ->
            PerformanceStats(
                callCount = times.size,
                averageTime = times.average(),
                minTime = times.minOrNull() ?: 0,
                maxTime = times.maxOrNull() ?: 0,
                totalTime = times.sum()
            )
        }
    }
    
    data class PerformanceStats(
        val callCount: Int,
        val averageTime: Double,
        val minTime: Long,
        val maxTime: Long,
        val totalTime: Long
    )
}

/**
 * Usage example in Application class
 */
class MyApplication : android.app.Application() {
    override fun onCreate() {
        super.onCreate()
        
        // Initialize Python Bridge Manager
        val bridgeManager = PythonBridgeManager.getInstance(this)
        
        // Register modules using DSL
        bridgeManager.registerModules(
            pythonModule {
                moduleName = "astral_symbolic_module"
                autoInitialize = true
                requiredPythonPackages = listOf("numpy", "json", "datetime")
            },
            pythonModule {
                moduleName = "data_processor"
                autoInitialize = false
                requiredPythonPackages = listOf("pandas", "numpy")
            },
            pythonModule {
                moduleName = "ml_model"
                autoInitialize = false
                requiredPythonPackages = listOf("tensorflow", "numpy", "scikit-learn")
            }
        )
        
        // Initialize the bridge
        bridgeManager.initialize()
    }
}
