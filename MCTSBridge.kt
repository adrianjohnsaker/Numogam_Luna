package com.antonio.my.ai.girlfriend.free.bridge

import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap

/**
 * Kotlin bridge for the Monte Carlo Tree Search (MCTS) Python module.
 * Handles communication between Kotlin/Android and Python with error handling
 * and performance optimizations.
 */
class MCTSBridge {
    companion object {
        private const val TAG = "MCTSBridge"
        private const val MODULE_NAME = "mcts_module"
        
        // Initialize Python once if not already initialized
        @Synchronized
        fun initializePython(context: android.content.Context) {
            if (!Python.isStarted()) {
                AndroidPlatform.setApp(context.applicationContext)
                Python.start(AndroidPlatform(context))
                Log.d(TAG, "Python runtime initialized")
            }
        }
    }
    
    // Cache for Python module and objects
    private var pythonModule: PyObject? = null
    private var mctsInstance: PyObject? = null
    
    // Cache for state representations (avoid repeated serialization/deserialization)
    private val stateCache = ConcurrentHashMap<String, Any>()
    
    /**
     * Initialize the MCTS bridge and load the Python module
     */
    fun initialize() {
        try {
            val py = Python.getInstance()
            pythonModule = py.getModule(MODULE_NAME)
            Log.d(TAG, "MCTS module loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MCTS module: ${e.message}", e)
            throw RuntimeException("Failed to initialize MCTS module", e)
        }
    }
    
    /**
     * Create a new MCTS instance with the specified parameters
     */
    fun createMCTS(
        explorationWeight: Float = 1.41f,
        maxRolloutDepth: Int = 50,
        simulationLimit: Int = 1000,
        timeLimitMs: Int = 1000,
        rolloutPolicy: String = "random",
        pruneTreeInterval: Int = 100,
        maxMemoryNodes: Int = 10000
    ): Boolean {
        try {
            val params = JSONObject().apply {
                put("exploration_weight", explorationWeight)
                put("max_rollout_depth", maxRolloutDepth)
                put("simulation_limit", simulationLimit)
                put("time_limit_ms", timeLimitMs)
                put("rollout_policy", rolloutPolicy)
                put("prune_tree_interval", pruneTreeInterval)
                put("max_memory_nodes", maxMemoryNodes)
            }
            
            val jsonParams = params.toString()
            mctsInstance = pythonModule?.callAttr("MCTS.from_json", jsonParams)
            
            return mctsInstance != null
        } catch (e: Exception) {
            Log.e(TAG, "Error creating MCTS instance: ${e.message}", e)
            return false
        }
    }
    
    /**
     * Get the JSON representation of the current MCTS state
     */
    fun getMCTSState(): JSONObject {
        try {
            val jsonStr = mctsInstance?.callAttr("to_json")?.toString() ?: "{}"
            return JSONObject(jsonStr)
        } catch (e: JSONException) {
            Log.e(TAG, "Error parsing MCTS state JSON: ${e.message}", e)
            return JSONObject()
        }
    }
    
    /**
     * Run MCTS search with the given parameters and callbacks
     */
    suspend fun search(
        state: Any,
        getActions: (Any) -> List<Any>,
        applyAction: (Any, Any) -> Any,
        evaluate: (Any) -> Float,
        iterations: Int? = null,
        timeLimitMs: Int? = null
    ): Any? = withContext(Dispatchers.Default) {
        try {
            // Prepare state and callback wrappers
            val stateKey = state.hashCode().toString()
            stateCache[stateKey] = state
            
            // Create Python callback wrappers
            val getActionsFn = createGetActionsFunction(getActions)
            val applyActionFn = createApplyActionFunction(applyAction)
            val evaluateFn = createEvaluateFunction(evaluate)
            
            // Call Python search method
            val action = mctsInstance?.callAttr(
                "search",
                state,
                getActionsFn,
                applyActionFn,
                evaluateFn,
                iterations,
                timeLimitMs
            )
            
            // Clean up cache for this state
            stateCache.remove(stateKey)
            
            return@withContext action?.toJava(Any::class.java)
        } catch (e: PyException) {
            Log.e(TAG, "Python error during MCTS search: ${e.message}", e)
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error during MCTS search: ${e.message}", e)
            null
        }
    }
    
    /**
     * Run asynchronous MCTS search for better UI responsiveness
     */
    suspend fun searchAsync(
        state: Any,
        getActions: (Any) -> List<Any>,
        applyAction: (Any, Any) -> Any,
        evaluate: (Any) -> Float,
        iterations: Int? = null,
        timeLimitMs: Int? = null,
        progressCallback: ((Int, Int) -> Unit)? = null
    ): Any? = withContext(Dispatchers.IO) {
        try {
            // Prepare state and callback wrappers
            val stateKey = state.hashCode().toString()
            stateCache[stateKey] = state
            
            // Create Python callback wrappers
            val getActionsFn = createGetActionsFunction(getActions)
            val applyActionFn = createApplyActionFunction(applyAction)
            val evaluateFn = createEvaluateFunction(evaluate)
            
            // Start a monitoring thread if progress callback provided
            val monitoringThread = if (progressCallback != null) {
                Thread {
                    var lastIteration = 0
                    val totalIterations = iterations ?: 1000
                    
                    while (!Thread.interrupted()) {
                        try {
                            val stats = getStats()
                            val currentIterations = stats.optInt("simulation_count", 0)
                            
                            if (currentIterations > lastIteration) {
                                progressCallback(currentIterations, totalIterations)
                                lastIteration = currentIterations
                            }
                            
                            Thread.sleep(100)
                        } catch (e: Exception) {
                            break
                        }
                    }
                }.apply { start() }
            } else null
            
            // Call Python async search method
            val action = mctsInstance?.callAttr(
                "search_async",
                state,
                getActionsFn,
                applyActionFn,
                evaluateFn,
                iterations,
                timeLimitMs
            )?.asAwait()
            
            // Stop monitoring thread
            monitoringThread?.interrupt()
            
            // Clean up cache for this state
            stateCache.remove(stateKey)
            
            return@withContext action?.toJava(Any::class.java)
        } catch (e: PyException) {
            Log.e(TAG, "Python error during async MCTS search: ${e.message}", e)
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error during async MCTS search: ${e.message}", e)
            null
        }
    }
    
    /**
     * Update the root of the MCTS tree after taking an action
     */
    fun updateRootWithAction(action: Any): Boolean {
        return try {
            val result = mctsInstance?.callAttr("update_root_with_action", action)
            result?.toBoolean() ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Error updating MCTS root: ${e.message}", e)
            false
        }
    }
    
    /**
     * Get statistics about the current search tree
     */
    fun getStats(): JSONObject {
        return try {
            val statsStr = mctsInstance?.callAttr("get_stats")?.toString() ?: "{}"
            JSONObject(statsStr)
        } catch (e: Exception) {
            Log.e(TAG, "Error getting MCTS stats: ${e.message}", e)
            JSONObject()
        }
    }
    
    /**
     * Clear search history to free memory
     */
    fun clearHistory() {
        try {
            mctsInstance?.callAttr("clear_history")
        } catch (e: Exception) {
            Log.e(TAG, "Error clearing MCTS history: ${e.message}", e)
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            mctsInstance?.callAttr("cleanup")
            stateCache.clear()
        } catch (e: Exception) {
            Log.e(TAG, "Error during MCTS cleanup: ${e.message}", e)
        }
    }
    
    /**
     * Get a simplified representation of the tree suitable for visualization
     */
    fun getTreeVisualizationData(resolution: Int = 50): JSONObject {
        return try {
            val dataStr = mctsInstance?.callAttr("get_simplified_data", resolution)?.toString() ?: "{}"
            JSONObject(dataStr)
        } catch (e: Exception) {
            Log.e(TAG, "Error getting visualization data: ${e.message}", e)
            JSONObject()
        }
    }
    
    /**
     * Create Python function wrapper for getActions callback
     */
    private fun createGetActionsFunction(getActions: (Any) -> List<Any>): PyObject {
        val py = Python.getInstance()
        
        return py.getBuiltins().callAttr("lambda", "state: $MODULE_NAME.get_actions_callback(state)")
    }
    
    /**
     * Create Python function wrapper for applyAction callback
     */
    private fun createApplyActionFunction(applyAction: (Any, Any) -> Any): PyObject {
        val py = Python.getInstance()
        
        return py.getBuiltins().callAttr(
            "lambda", 
            "state, action: $MODULE_NAME.apply_action_callback(state, action)"
        )
    }
    
    /**
     * Create Python function wrapper for evaluate callback
     */
    private fun createEvaluateFunction(evaluate: (Any) -> Float): PyObject {
        val py = Python.getInstance()
        
        return py.getBuiltins().callAttr("lambda", "state: $MODULE_NAME.evaluate_callback(state)")
    }
    
    /**
     * Register Kotlin callbacks in Python module
     */
    fun registerCallbacks() {
        try {
            // Register the Kotlin callback handlers in Python
            pythonModule?.callAttr("register_kotlin_callbacks", object : PyObject() {
                fun get_actions_callback(state: PyObject): PyObject {
                    val stateObj = state.toJava(Any::class.java)
                    val actions = try {
                        val actionsFromKotlin = stateCache[stateObj.hashCode().toString()]?.let { cachedState ->
                            (stateCache["getActionsFn"] as? ((Any) -> List<Any>))?.invoke(cachedState)
                        } ?: emptyList()
                        
                        // Convert to Python list
                        val py = Python.getInstance()
                        val pyList = py.builtins.callAttr("list")
                        actionsFromKotlin.forEach { action ->
                            pyList.callAttr("append", action)
                        }
                        pyList
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in get_actions_callback: ${e.message}", e)
                        Python.getInstance().builtins.callAttr("list")
                    }
                    return actions
                }
                
                fun apply_action_callback(state: PyObject, action: PyObject): PyObject {
                    val stateObj = state.toJava(Any::class.java)
                    val actionObj = action.toJava(Any::class.java)
                    
                    return try {
                        val nextState = stateCache[stateObj.hashCode().toString()]?.let { cachedState ->
                            (stateCache["applyActionFn"] as? ((Any, Any) -> Any))?.invoke(cachedState, actionObj)
                        }
                        Python.getInstance().builtins.callAttr("object", nextState)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in apply_action_callback: ${e.message}", e)
                        state
                    }
                }
                
                fun evaluate_callback(state: PyObject): Float {
                    val stateObj = state.toJava(Any::class.java)
                    
                    return try {
                        stateCache[stateObj.hashCode().toString()]?.let { cachedState ->
                            (stateCache["evaluateFn"] as? ((Any) -> Float))?.invoke(cachedState)
                        } ?: 0.0f
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in evaluate_callback: ${e.message}", e)
                        0.0f
                    }
                }
            })
            
            Log.d(TAG, "Kotlin callbacks registered successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error registering callbacks: ${e.message}", e)
            throw RuntimeException("Failed to register callbacks", e)
        }
    }
    
    /**
     * Store callback functions in cache for later use
     */
    fun setCallbackFunctions(
        getActionsFn: (Any) -> List<Any>,
        applyActionFn: (Any, Any) -> Any,
        evaluateFn: (Any) -> Float
    ) {
        stateCache["getActionsFn"] = getActionsFn
        stateCache["applyActionFn"] = applyActionFn
        stateCache["evaluateFn"] = evaluateFn
    }
}

/**
 * Data class for MCTS configuration parameters
 */
data class MCTSConfig(
    val explorationWeight: Float = 1.41f,
    val maxRolloutDepth: Int = 50,
    val simulationLimit: Int = 1000,
    val timeLimitMs: Int = 1000,
    val rolloutPolicy: String = "random",
    val pruneTreeInterval: Int = 100,
    val maxMemoryNodes: Int = 10000
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("exploration_weight", explorationWeight)
        put("max_rollout_depth", maxRolloutDepth)
        put("simulation_limit", simulationLimit)
        put("time_limit_ms", timeLimitMs)
        put("rollout_policy", rolloutPolicy)
        put("prune_tree_interval", pruneTreeInterval)
        put("max_memory_nodes", maxMemoryNodes)
    }
}

/**
 * Factory for creating MCTS bridges
 */
object MCTSFactory {
    /**
     * Create and initialize an MCTS bridge with the given configuration
     */
    suspend fun createMCTS(
        context: android.content.Context,
        config: MCTSConfig = MCTSConfig()
    ): MCTSBridge = withContext(Dispatchers.Default) {
        MCTSBridge.initializePython(context)
        
        val bridge = MCTSBridge()
        bridge.initialize()
        bridge.createMCTS(
            explorationWeight = config.explorationWeight,
            maxRolloutDepth = config.maxRolloutDepth,
            simulationLimit = config.simulationLimit,
            timeLimitMs = config.timeLimitMs,
            rolloutPolicy = config.rolloutPolicy,
            pruneTreeInterval = config.pruneTreeInterval,
            maxMemoryNodes = config.maxMemoryNodes
        )
        bridge.registerCallbacks()
        
        return@withContext bridge
    }
}
