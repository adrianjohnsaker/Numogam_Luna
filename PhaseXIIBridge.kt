package com.amelia.ai.phasexii

import android.content.Context
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * PhaseXIIBridge is the Kotlin interface to the Python modules for Phase XII of Amelia AI.
 * 
 * This class communicates with the Python bridge and provides type-safe Kotlin methods
 * for interacting with the three Python modules:
 * - Circadian Narrative Cycles
 * - Ritual Interaction Patterns
 * - Temporal Flow Maps
 */
class PhaseXIIBridge private constructor(private val context: Context) {
    private val TAG = "PhaseXIIBridge"
    
    // Python bridge
    private var pythonBridge: PyObject? = null
    
    // Module instance IDs
    private var circadianInstanceId: String? = null
    private var ritualInstanceId: String? = null
    private var temporalInstanceId: String? = null
    
    // Initialization status
    private var isInitialized = false
    private var initError: String? = null
    
    companion object {
        @Volatile
        private var instance: PhaseXIIBridge? = null
        
        fun getInstance(context: Context): PhaseXIIBridge {
            return instance ?: synchronized(this) {
                instance ?: PhaseXIIBridge(context.applicationContext).also { instance = it }
            }
        }
    }
    
    /**
     * Initialize the Python interpreter and bridge.
     * 
     * @param modulesPath Path to the Python modules
     * @return True if initialization was successful, false otherwise
     */
    suspend fun initialize(modulesPath: String): Boolean = withContext(Dispatchers.IO) {
        try {
            if (isInitialized) {
                return@withContext true
            }
            
            // Start Python if not already started
            if (!Python.isStarted()) {
                if (!Python.start(AndroidPlatform(context))) {
                    initError = "Failed to start Python"
                    Log.e(TAG, initError!!)
                    return@withContext false
                }
            }
            
            val py = Python.getInstance()
            
            // Add modules path to Python path
            val sys = py.getModule("sys")
            val pathList = sys.get("path").asList()
            if (modulesPath !in pathList) {
                sys.callAttr("path", "insert", 0, modulesPath)
            }
            
            // Import the bridge module
            try {
                pythonBridge = py.getModule("python_bridge").get("bridge")
            } catch (e: PyException) {
                initError = "Failed to import python_bridge: ${e.message}"
                Log.e(TAG, initError!!, e)
                return@withContext false
            }
            
            // Create instances of all three modules
            try {
                // Create Circadian Narrative Cycles instance
                val circadianResult = callBridgeMethod("create_circadian_instance")
                if (!circadianResult.getBoolean("success")) {
                    throw Exception("Failed to create CircadianNarrativeCycles instance: ${
                        circadianResult.getJSONObject("error").getString("message")
                    }")
                }
                circadianInstanceId = circadianResult.getJSONObject("result").toString()
                
                // Create Ritual Interaction Patterns instance
                val ritualResult = callBridgeMethod("create_ritual_instance_manager")
                if (!ritualResult.getBoolean("success")) {
                    throw Exception("Failed to create RitualInteractionPatterns instance: ${
                        ritualResult.getJSONObject("error").getString("message")
                    }")
                }
                ritualInstanceId = ritualResult.getJSONObject("result").toString()
                
                // Create Temporal Flow Maps instance
                val temporalResult = callBridgeMethod("create_temporal_flow_maps")
                if (!temporalResult.getBoolean("success")) {
                    throw Exception("Failed to create TemporalFlowMaps instance: ${
                        temporalResult.getJSONObject("error").getString("message")
                    }")
                }
                temporalInstanceId = temporalResult.getJSONObject("result").toString()
                
                isInitialized = true
                Log.d(TAG, "PhaseXIIBridge successfully initialized")
                return@withContext true
                
            } catch (e: Exception) {
                initError = "Error initializing bridge instances: ${e.message}"
                Log.e(TAG, initError!!, e)
                return@withContext false
            }
            
        } catch (e: Exception) {
            initError = "Error initializing Python: ${e.message}"
            Log.e(TAG, initError!!, e)
            return@withContext false
        }
    }
    
    /**
     * Check if the bridge is initialized
     */
    fun isInitialized(): Boolean = isInitialized
    
    /**
     * Get the initialization error, if any
     */
    fun getInitError(): String? = initError
    
    /**
     * Call a method on the Python bridge
     * 
     * @param methodName Name of the bridge method to call
     * @param params Parameters for the method
     * @return JSONObject with the result
     */
    private suspend fun callBridgeMethod(
        methodName: String,
        params: Map<String, Any?> = emptyMap()
    ): JSONObject = withContext(Dispatchers.IO) {
        try {
            if (pythonBridge == null) {
                throw Exception("Python bridge not initialized")
            }
            
            // Convert params to JSON
            val paramsJson = JSONObject(params).toString()
            
            // Call the bridge method
            val resultJson = pythonBridge!!.callAttr(
                "call_method",
                methodName,
                paramsJson
            ).toString()
            
            return@withContext JSONObject(resultJson)
        } catch (e: Exception) {
            // Create error result
            val errorResult = JSONObject()
            errorResult.put("success", false)
            
            val error = JSONObject()
            error.put("message", e.message ?: "Unknown error")
            error.put("type", e.javaClass.simpleName)
            
            errorResult.put("error", error)
            return@withContext errorResult
        }
    }
    
    //----------------------------------------
    // Circadian Narrative Cycles Methods
    //----------------------------------------
    
    /**
     * Get the narrative tone for the current time or a specific time.
     * 
     * @param dateTime Optional specific time to get tone for
     * @return JSONObject containing tone information or error
     */
    suspend fun getCurrentNarrativeTone(dateTime: LocalDateTime? = null): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to circadianInstanceId
        )
        
        if (dateTime != null) {
            params["datetime_str"] = dateTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        }
        
        return callBridgeMethod("get_current_narrative_tone", params)
    }
    
    /**
     * Transform a narrative text based on the current circadian tone.
     * 
     * @param narrativeText Original text to transform
     * @param dateTime Optional specific time to use for transformation
     * @param transformationStrength How strongly to apply the transformation (0.0-1.0)
     * @return JSONObject with transformed text or error
     */
    suspend fun transformNarrative(
        narrativeText: String,
        dateTime: LocalDateTime? = null,
        transformationStrength: Float = 1.0f
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to circadianInstanceId,
            "narrative_text" to narrativeText,
            "transformation_strength" to transformationStrength
        )
        
        if (dateTime != null) {
            params["datetime_str"] = dateTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        }
        
        return callBridgeMethod("transform_narrative", params)
    }
    
    /**
     * Generate a tone-specific version of a base prompt for AI generation.
     * 
     * @param basePrompt The original prompt text
     * @param dateTime Optional specific time to use for prompt generation
     * @return JSONObject with modified prompt or error
     */
    suspend fun generateToneSpecificPrompt(
        basePrompt: String,
        dateTime: LocalDateTime? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to circadianInstanceId,
            "base_prompt" to basePrompt
        )
        
        if (dateTime != null) {
            params["datetime_str"] = dateTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        }
        
        return callBridgeMethod("generate_tone_specific_prompt", params)
    }
    
    /**
     * Get the full schedule of tones throughout a 24-hour cycle.
     * 
     * @return JSONObject with tone schedule or error
     */
    suspend fun getToneSchedule(): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to circadianInstanceId
        )
        
        return callBridgeMethod("get_tone_schedule", params)
    }
    
    //----------------------------------------
    // Ritual Interaction Patterns Methods
    //----------------------------------------
    
    /**
     * Identify appropriate ritual opportunities based on current context.
     * 
     * @param context Context information
     * @param userHistory Optional user's ritual history
     * @return JSONObject with ritual suggestion or error
     */
    suspend fun identifyRitualOpportunity(
        context: JSONObject,
        userHistory: JSONObject? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to ritualInstanceId,
            "context_json" to context.toString()
        )
        
        if (userHistory != null) {
            params["user_history_json"] = userHistory.toString()
        }
        
        return callBridgeMethod("identify_ritual_opportunity", params)
    }
    
    /**
     * Create a new ritual instance for tracking.
     * 
     * @param ritualType Type of ritual from templates
     * @param userId Identifier for the user
     * @param concepts List of concept IDs involved
     * @param context Additional contextual information
     * @return JSONObject with ritual ID or error
     */
    suspend fun createRitual(
        ritualType: String,
        userId: String,
        concepts: List<String>,
        context: JSONObject? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val conceptsArray = JSONArray()
        concepts.forEach { conceptsArray.put(it) }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to ritualInstanceId,
            "ritual_type" to ritualType,
            "user_id" to userId,
            "concepts_json" to conceptsArray.toString()
        )
        
        if (context != null) {
            params["context_json"] = context.toString()
        }
        
        return callBridgeMethod("create_ritual", params)
    }
    
    /**
     * Advance a ritual to the next stage based on user interaction.
     * 
     * @param ritualId The ritual instance ID
     * @param interactionData Data from the user interaction
     * @return JSONObject with updated ritual status or error
     */
    suspend fun advanceRitualStage(
        ritualId: String,
        interactionData: JSONObject? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to ritualInstanceId,
            "ritual_id" to ritualId
        )
        
        if (interactionData != null) {
            params["interaction_data_json"] = interactionData.toString()
        }
        
        return callBridgeMethod("advance_ritual_stage", params)
    }
    
    /**
     * Get the current status of a ritual.
     * 
     * @param ritualId The ritual instance ID
     * @return JSONObject with ritual status or error
     */
    suspend fun getRitualStatus(ritualId: String): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to ritualInstanceId,
            "ritual_id" to ritualId
        )
        
        return callBridgeMethod("get_ritual_status", params)
    }
    
    /**
     * Get information about available ritual templates.
     * 
     * @return JSONObject with ritual templates or error
     */
    suspend fun getAvailableRituals(): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to ritualInstanceId
        )
        
        return callBridgeMethod("get_available_rituals", params)
    }
    
    /**
     * Get a user's ritual history.
     * 
     * @param userId The user identifier
     * @return JSONObject with ritual history or error
     */
    suspend fun getUserRitualHistory(userId: String): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to ritualInstanceId,
            "user_id" to userId
        )
        
        return callBridgeMethod("get_user_ritual_history", params)
    }
    
    /**
     * Generate a comprehensive report for a ritual.
     * 
     * @param ritualId The ritual instance ID
     * @return JSONObject with detailed report or error
     */
    suspend fun generateRitualReport(ritualId: String): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to ritualInstanceId,
            "ritual_id" to ritualId
        )
        
        return callBridgeMethod("generate_ritual_report", params)
    }
    
    //----------------------------------------
    // Temporal Flow Maps Methods
    //----------------------------------------
    
    /**
     * Create a new symbol in the mythology.
     * 
     * @param name Name of the symbol
     * @param description Description of the symbol
     * @param attributes Optional attributes dictionary
     * @return JSONObject with symbol ID or error
     */
    suspend fun createSymbol(
        name: String,
        description: String,
        attributes: JSONObject? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "name" to name,
            "description" to description
        )
        
        if (attributes != null) {
            params["attributes_json"] = attributes.toString()
        }
        
        return callBridgeMethod("create_symbol", params)
    }
    
    /**
     * Create a new concept in the mythology.
     * 
     * @param name Name of the concept
     * @param description Description of the concept
     * @param relatedSymbols Optional list of symbol IDs related to this concept
     * @return JSONObject with concept ID or error
     */
    suspend fun createConcept(
        name: String,
        description: String,
        relatedSymbols: List<String>? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "name" to name,
            "description" to description
        )
        
        if (relatedSymbols != null) {
            val symbolsArray = JSONArray()
            relatedSymbols.forEach { symbolsArray.put(it) }
            params["related_symbols_json"] = symbolsArray.toString()
        }
        
        return callBridgeMethod("create_concept", params)
    }
    
    /**
     * Create a new narrative thread in the mythology.
     * 
     * @param name Name of the narrative
     * @param description Description of the narrative
     * @param relatedConcepts Optional list of concept IDs in this narrative
     * @param relatedSymbols Optional list of symbol IDs in this narrative
     * @return JSONObject with narrative ID or error
     */
    suspend fun createNarrative(
        name: String,
        description: String,
        relatedConcepts: List<String>? = null,
        relatedSymbols: List<String>? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "name" to name,
            "description" to description
        )
        
        if (relatedConcepts != null) {
            val conceptsArray = JSONArray()
            relatedConcepts.forEach { conceptsArray.put(it) }
            params["related_concepts_json"] = conceptsArray.toString()
        }
        
        if (relatedSymbols != null) {
            val symbolsArray = JSONArray()
            relatedSymbols.forEach { symbolsArray.put(it) }
            params["related_symbols_json"] = symbolsArray.toString()
        }
        
        return callBridgeMethod("create_narrative", params)
    }
    
    /**
     * Add an event to a narrative timeline.
     * 
     * @param narrativeId ID of the narrative
     * @param title Event title
     * @param description Event description
     * @param involvedConcepts Optional list of concept IDs involved
     * @param involvedSymbols Optional list of symbol IDs involved
     * @param customTimestamp Optional custom timestamp for the event
     * @return JSONObject with result or error
     */
    suspend fun addNarrativeEvent(
        narrativeId: String,
        title: String,
        description: String,
        involvedConcepts: List<String>? = null,
        involvedSymbols: List<String>? = null,
        customTimestamp: String? = null
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "narrative_id" to narrativeId,
            "title" to title,
            "description" to description
        )
        
        if (involvedConcepts != null) {
            val conceptsArray = JSONArray()
            involvedConcepts.forEach { conceptsArray.put(it) }
            params["involved_concepts_json"] = conceptsArray.toString()
        }
        
        if (involvedSymbols != null) {
            val symbolsArray = JSONArray()
            involvedSymbols.forEach { symbolsArray.put(it) }
            params["involved_symbols_json"] = symbolsArray.toString()
        }
        
        if (customTimestamp != null) {
            params["custom_timestamp"] = customTimestamp
        }
        
        return callBridgeMethod("add_narrative_event", params)
    }
    
    /**
     * Generate a temporal map visualization data for the specified entities.
     * 
     * @param entityIds List of entity IDs to include
     * @param startTime Optional ISO timestamp for start time
     * @param endTime Optional ISO timestamp for end time
     * @param timeScale Time scale for the visualization
     * @param includeRelated Whether to include related entities
     * @param maxRelatedDepth Maximum depth for related entities
     * @return JSONObject with visualization data or error
     */
    suspend fun generateTemporalMap(
        entityIds: List<String>,
        startTime: String? = null,
        endTime: String? = null,
        timeScale: String? = null,
        includeRelated: Boolean = true,
        maxRelatedDepth: Int = 1
    ): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val entityIdsArray = JSONArray()
        entityIds.forEach { entityIdsArray.put(it) }
        
        val params = mutableMapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "entity_ids_json" to entityIdsArray.toString(),
            "include_related" to includeRelated,
            "max_related_depth" to maxRelatedDepth
        )
        
        if (startTime != null) params["start_time"] = startTime
        if (endTime != null) params["end_time"] = endTime
        if (timeScale != null) params["time_scale"] = timeScale
        
        return callBridgeMethod("generate_temporal_map", params)
    }
    
    /**
     * Generate a specialized timeline showing the evolution of a concept.
     * 
     * @param conceptId ID of the concept
     * @return JSONObject with timeline data or error
     */
    suspend fun generateConceptEvolutionTimeline(conceptId: String): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "concept_id" to conceptId
        )
        
        return callBridgeMethod("generate_concept_evolution_timeline", params)
    }
    
    /**
     * Generate a specialized timeline showing how a symbol has been used over time.
     * 
     * @param symbolId ID of the symbol
     * @return JSONObject with timeline data or error
     */
    suspend fun generateSymbolUsageTimeline(symbolId: String): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "symbol_id" to symbolId
        )
        
        return callBridgeMethod("generate_symbol_usage_timeline", params)
    }
    
    /**
     * Generate a specialized map showing the flow of a narrative over time.
     * 
     * @param narrativeId ID of the narrative
     * @return JSONObject with flow map data or error
     */
    suspend fun generateNarrativeFlowMap(narrativeId: String): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "narrative_id" to narrativeId
        )
        
        return callBridgeMethod("generate_narrative_flow_map", params)
    }
    
    /**
     * Export all temporal flow map data for backup or analysis.
     * 
     * @param formatType Export format (currently only 'json' supported)
     * @return JSONObject with export data or error
     */
    suspend fun exportTemporalData(formatType: String = "json"): JSONObject {
        if (!isInitialized) {
            return createErrorResult("Bridge not initialized")
        }
        
        val params = mapOf<String, Any?>(
            "instance_id" to temporalInstanceId,
            "format_type" to formatType
        )
        
        return callBridgeMethod("export_temporal_data", params)
    }
    
    /**
     * Create a standard error result JSONObject
     */
    private fun createErrorResult(message: String): JSONObject {
        val result = JSONObject()
        result.put("success", false)
        
        val error = JSONObject()
        error.put("message", message)
        error.put("type", "BridgeError")
        
        result.put("error", error)
        return result
    }
}
