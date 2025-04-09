package com.antonio.my.ai.girlfriend.free.sensitivityconfig

import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import kotlinx.serialization.descriptors.*
import java.io.File
import kotlin.math.max
import kotlin.math.min

/**
 * Data class representing sensitivity configuration parameters
 */
@Serializable
data class SensitivityConfig(
    val baseSensitivity: Float = 1.0f,
    val learningRate: Float = 0.05f,
    val maxAdjustment: Float = 5.0f,
    val minAdjustment: Float = 0.1f,
    val decayFactor: Float = 0.98f,
    val smoothingWindow: Int = 5
)

/**
 * Data class for feedback items
 */
@Serializable
data class FeedbackItem(
    val value: Float,
    val zone: String = "normal",
    val userId: String? = null,
    val context: Map<String, JsonElement>? = null
)

/**
 * Data class representing the state of the FeedbackSensitivityAdjuster
 */
@Serializable
data class AdjusterState(
    val currentSensitivity: Float,
    val feedbackHistory: List<Map<String, JsonElement>> = emptyList(),
    val adjustmentHistory: List<Float> = emptyList(),
    val lastUpdated: Double = 0.0
)

/**
 * Response data class for analysis results
 */
@Serializable
data class AnalysisResult(
    val status: String,
    val count: Int? = null,
    val mean: Float? = null,
    val min: Float? = null,
    val max: Float? = null,
    val recentTrend: String? = null
)

/**
 * Main bridge class for interacting with the Python FeedbackSensitivityAdjuster
 */
class FeedbackSensitivityBridge {
    private val json = Json { 
        ignoreUnknownKeys = true
        prettyPrint = true
        isLenient = true
        coerceInputValues = true
    }
    
    private var pythonInterpreter: PythonInterpreter? = null
    private var lastJsonState: String = ""
    
    // Cache to minimize serialization overhead
    private var configCache: SensitivityConfig? = null
    private var stateCache: AdjusterState? = null
    
    /**
     * Initialize the bridge and connect to Python interpreter
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (pythonInterpreter == null) {
                pythonInterpreter = PythonInterpreter()
                
                // Import the Python module
                pythonInterpreter?.exec("""
                    import sys
                    sys.path.append('.')
                    from FeedbackSensitivityAdjuster import FeedbackSensitivityAdjuster, SensitivityConfig
                    
                    # Create initial instance
                    adjuster = FeedbackSensitivityAdjuster()
                """.trimIndent())
            }
            
            // Get initial state
            refreshState()
            true
        } catch (e: Exception) {
            println("Failed to initialize Python bridge: ${e.message}")
            e.printStackTrace()
            false
        }
    }
    
    /**
     * Create a new adjuster with specific configuration
     */
    suspend fun createAdjuster(
        config: SensitivityConfig,
        zoneWeights: Map<String, Float>? = null,
        memoryCapacity: Int = 100
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            // Convert config to JSON for Python
            val configJson = json.encodeToString(config)
            
            // Convert zone weights to JSON if provided
            val zoneWeightsJson = zoneWeights?.let { 
                json.encodeToString(it)
            } ?: "None"
            
            // Create new instance in Python
            pythonInterpreter?.exec("""
                config_dict = json.loads('$configJson')
                config = SensitivityConfig(
                    base_sensitivity=config_dict.get('baseSensitivity', 1.0),
                    learning_rate=config_dict.get('learningRate', 0.05),
                    max_adjustment=config_dict.get('maxAdjustment', 5.0),
                    min_adjustment=config_dict.get('minAdjustment', 0.1),
                    decay_factor=config_dict.get('decayFactor', 0.98),
                    smoothing_window=config_dict.get('smoothingWindow', 5)
                )
                
                zone_weights = json.loads('$zoneWeightsJson') if '$zoneWeightsJson' != 'None' else None
                
                adjuster = FeedbackSensitivityAdjuster(
                    config=config,
                    zone_weights=zone_weights,
                    memory_capacity=$memoryCapacity
                )
            """.trimIndent())
            
            refreshState()
            true
        } catch (e: Exception) {
            println("Failed to create adjuster: ${e.message}")
            false
        }
    }
    
    /**
     * Refresh the local state from Python
     */
    private suspend fun refreshState(): String = withContext(Dispatchers.IO) {
        try {
            val stateJson = pythonInterpreter?.eval("""adjuster.to_json()""")?.toString() ?: "{}"
            lastJsonState = stateJson
            
            // Update caches
            try {
                val stateObj = json.decodeFromString<JsonObject>(stateJson)
                
                // Update config cache
                stateObj["config"]?.let {
                    configCache = json.decodeFromJsonElement(it)
                }
                
                // Update state cache
                stateObj["state_data"]?.let {
                    stateCache = json.decodeFromJsonElement(it)
                }
            } catch (e: Exception) {
                println("Warning: Failed to decode JSON state - ${e.message}")
            }
            
            stateJson
        } catch (e: Exception) {
            println("Failed to refresh state: ${e.message}")
            "{}"
        }
    }
    
    /**
     * Add feedback and get the new sensitivity
     */
    suspend fun addFeedback(
        value: Float,
        zone: String = "normal",
        userId: String? = null,
        context: Map<String, Any>? = null
    ): Float = withContext(Dispatchers.IO) {
        try {
            // Convert context to JSON if provided
            val contextJson = context?.let {
                val jsonMap = mutableMapOf<String, JsonElement>()
                for ((k, v) in it) {
                    when (v) {
                        is Int -> jsonMap[k] = JsonPrimitive(v)
                        is Float -> jsonMap[k] = JsonPrimitive(v)
                        is Double -> jsonMap[k] = JsonPrimitive(v)
                        is Boolean -> jsonMap[k] = JsonPrimitive(v)
                        is String -> jsonMap[k] = JsonPrimitive(v)
                        else -> jsonMap[k] = JsonPrimitive(v.toString())
                    }
                }
                json.encodeToString(jsonMap)
            } ?: "None"
            
            // Safe string conversion for Python
            val userIdStr = userId?.let { "'$it'" } ?: "None"
            
            // Call Python method
            val result = pythonInterpreter?.eval("""
                adjuster.add_feedback(
                    value=$value, 
                    zone='$zone', 
                    user_id=$userIdStr, 
                    context=json.loads('$contextJson') if '$contextJson' != 'None' else None
                )
            """)?.toString()?.toFloatOrNull() ?: 0.0f
            
            // Refresh state after modification
            refreshState()
            
            result
        } catch (e: Exception) {
            println("Failed to add feedback: ${e.message}")
            0.0f
        }
    }
    
    /**
     * Process a batch of feedback items asynchronously
     */
    suspend fun processFeedbackBatch(
        feedbacks: List<FeedbackItem>
    ): Map<String, Any> = withContext(Dispatchers.IO) {
        try {
            // Convert feedbacks to JSON for Python
            val feedbacksJson = json.encodeToString(feedbacks)
            
            // Call Python async method and await result
            val resultJson = pythonInterpreter?.eval("""
                import asyncio
                
                # Parse feedbacks from JSON
                feedbacks = json.loads('$feedbacksJson')
                
                # Create an event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run async method and get result
                result = loop.run_until_complete(adjuster.process_feedback_batch(feedbacks))
                
                # Return JSON result
                json.dumps(result)
            """)?.toString() ?: "{}"
            
            // Refresh state after modification
            refreshState()
            
            // Parse result
            @Suppress("UNCHECKED_CAST")
            json.decodeFromString<Map<String, JsonElement>>(resultJson).mapValues { (_, value) ->
                when (value) {
                    is JsonPrimitive -> {
                        when {
                            value.isString -> value.content
                            value.booleanOrNull != null -> value.boolean
                            value.intOrNull != null -> value.int
                            value.floatOrNull != null -> value.float
                            value.doubleOrNull != null -> value.double
                            else -> value.toString()
                        }
                    }
                    else -> json.decodeFromJsonElement<Any>(value)
                }
            }
        } catch (e: Exception) {
            println("Failed to process feedback batch: ${e.message}")
            mapOf("status" to "error", "error_message" to e.message)
        }
    }
    
    /**
     * Get the current sensitivity value
     */
    suspend fun getCurrentSensitivity(): Float = withContext(Dispatchers.IO) {
        // Use cached value if available
        stateCache?.currentSensitivity?.let { return@withContext it }
        
        try {
            pythonInterpreter?.eval("adjuster.get_current_sensitivity()")?.toString()?.toFloatOrNull() ?: 1.0f
        } catch (e: Exception) {
            println("Failed to get sensitivity: ${e.message}")
            1.0f
        }
    }
    
    /**
     * Get the smoothed sensitivity value
     */
    suspend fun getSmoothedSensitivity(): Float = withContext(Dispatchers.IO) {
        try {
            pythonInterpreter?.eval("adjuster.get_smoothed_sensitivity()")?.toString()?.toFloatOrNull() ?: 1.0f
        } catch (e: Exception) {
            println("Failed to get smoothed sensitivity: ${e.message}")
            1.0f
        }
    }
    
    /**
     * Reset sensitivity to base value
     */
    suspend fun resetSensitivity(): Boolean = withContext(Dispatchers.IO) {
        try {
            pythonInterpreter?.exec("adjuster.reset_sensitivity()")
            refreshState()
            true
        } catch (e: Exception) {
            println("Failed to reset sensitivity: ${e.
