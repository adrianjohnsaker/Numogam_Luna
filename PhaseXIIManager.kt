package com.amelia.ai.phasexii

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.time.LocalDateTime
import java.util.concurrent.ConcurrentHashMap

/**
 * PhaseXIIManager is the high-level manager for Phase XII functionality.
 * 
 * This class handles:
 * - Asynchronous operations with the bridge
 * - UI callbacks and LiveData for observing results
 * - Caching of frequently used data
 * - Synchronization of operations
 * - Error handling and recovery
 */
class PhaseXIIManager private constructor(private val context: Context) {
    private val TAG = "PhaseXIIManager"
    
    // Bridge instance
    private val bridge = PhaseXIIBridge.getInstance(context)
    
    // Coroutine scope for async operations
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    // Initialization state
    private val _initializationState = MutableLiveData<InitializationState>()
    val initializationState: LiveData<InitializationState> = _initializationState
    
    // Cache
    private val toneCache = ConcurrentHashMap<String, Pair<Long, JSONObject>>()
    private val ritualTemplateCache = MutableLiveData<JSONObject>()
    
    // Cache expiration time in milliseconds (5 minutes)
    private val CACHE_EXPIRATION = 5 * 60 * 1000L
    
    companion object {
        @Volatile
        private var instance: PhaseXIIManager? = null
        
        fun getInstance(context: Context): PhaseXIIManager {
            return instance ?: synchronized(this) {
                instance ?: PhaseXIIManager(context.applicationContext).also { instance = it }
            }
        }
    }
    
    init {
        _initializationState.value = InitializationState.NotInitialized
    }
    
    /**
     * Initialize the manager and underlying Python modules
     * 
     * @param pythonModulesPath Path to the Python modules
     */
    fun initialize(pythonModulesPath: String) {
        if (_initializationState.value == InitializationState.Initializing ||
            _initializationState.value == InitializationState.Initialized) {
            return
        }
        
        _initializationState.value = InitializationState.Initializing
        
        scope.launch {
            try {
                val result = bridge.initialize(pythonModulesPath)
                if (result) {
                    _initializationState.value = InitializationState.Initialized
                    Log.d(TAG, "PhaseXIIManager initialized successfully")
                    
                    // Preload common data
                    preloadCommonData()
                } else {
                    val error = bridge.getInitError() ?: "Unknown error"
                    _initializationState.value = InitializationState.Error(error)
                    Log.e(TAG, "Failed to initialize PhaseXIIManager: $error")
                }
            } catch (e: Exception) {
                _initializationState.value = InitializationState.Error(e.message ?: "Unknown error")
                Log.e(TAG, "Error initializing PhaseXIIManager", e)
            }
        }
    }
    
    /**
     * Preload commonly used data to improve responsiveness
     */
    private suspend fun preloadCommonData() {
        try {
            // Preload tone schedule
            getToneSchedule()
            
            // Preload available rituals
            getAvailableRituals()
        } catch (e: Exception) {
            Log.e(TAG, "Error preloading common data", e)
        }
    }
    
    //----------------------------------------
    // Circadian Narrative Cycles Methods
    //----------------------------------------
    
    /**
     * Get the current narrative tone based on time of day
     * 
     * @param dateTime Optional specific time to get tone for
     * @param useCache Whether to use cached data if available
     * @param callback Callback for the result
     */
    fun getCurrentNarrativeTone(
        dateTime: LocalDateTime? = null,
        useCache: Boolean = true,
        callback: (Result<JSONObject>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                // Check cache first if enabled
                if (useCache && dateTime == null) {
                    val cacheKey = "current_tone"
                    val cachedTone = toneCache[cacheKey]
                    if (cachedTone != null && (System.currentTimeMillis() - cachedTone.first) < CACHE_EXPIRATION) {
                        callback(Result.success(cachedTone.second))
                        return@launch
                    }
                }
                
                // Get fresh data
                val result = bridge.getCurrentNarrativeTone(dateTime)
                if (result.getBoolean("success")) {
                    val toneData = result.getJSONObject("result")
                    
                    // Cache the result if it's for current time
                    if (dateTime == null) {
                        toneCache["current_tone"] = Pair(System.currentTimeMillis(), toneData)
                    }
                    
                    callback(Result.success(toneData))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting current narrative tone", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Transform a narrative text based on the current circadian tone
     * 
     * @param narrativeText Original text to transform
     * @param dateTime Optional specific time to use for transformation
     * @param transformationStrength How strongly to apply the transformation (0.0-1.0)
     * @param callback Callback for the result
     */
    fun transformNarrative(
        narrativeText: String,
        dateTime: LocalDateTime? = null,
        transformationStrength: Float = 1.0f,
        callback: (Result<String>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.transformNarrative(narrativeText, dateTime, transformationStrength)
                if (result.getBoolean("success")) {
                    val transformedText = result.getJSONObject("result").toString()
                    callback(Result.success(transformedText))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error transforming narrative", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Generate a tone-specific version of a base prompt
     * 
     * @param basePrompt Original prompt to modify
     * @param dateTime Optional specific time to use
     * @param callback Callback for the result
     */
    fun generateToneSpecificPrompt(
        basePrompt: String,
        dateTime: LocalDateTime? = null,
        callback: (Result<String>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.generateToneSpecificPrompt(basePrompt, dateTime)
                if (result.getBoolean("success")) {
                    val enhancedPrompt = result.getJSONObject("result").toString()
                    callback(Result.success(enhancedPrompt))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error generating tone-specific prompt", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Get the full schedule of tones throughout a 24-hour cycle
     * 
     * @return LiveData with the tone schedule or null if an error occurs
     */
    fun getToneSchedule(): LiveData<JSONObject?> {
        val liveData = MutableLiveData<JSONObject?>()
        
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.getToneSchedule()
                if (result.getBoolean("success")) {
                    val schedule = result.getJSONObject("result")
                    liveData.value = schedule
                } else {
                    val errorObj = result.getJSONObject("error")
                    Log.e(TAG, "Error getting tone schedule: ${errorObj.getString("message")}")
                    liveData.value = null
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting tone schedule", e)
                liveData.value = null
            }
        }
        
        return liveData
    }
    
    //----------------------------------------
    // Ritual Interaction Patterns Methods
    //----------------------------------------
    
    /**
     * Identify appropriate ritual opportunities
     * 
     * @param context Context information
     * @param userHistory Optional user's ritual history
     * @param callback Callback for the result
     */
    fun identifyRitualOpportunity(
        context: JSONObject,
        userHistory: JSONObject? = null,
        callback: (Result<JSONObject?>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                // Add current time if not present
                if (!context.has("current_time")) {
                    context.put("current_time", LocalDateTime.now().toString())
                }
                
                val result = bridge.identifyRitualOpportunity(context, userHistory)
                if (result.getBoolean("success")) {
                    // The result might be null if no suitable ritual was found
                    val opportunityData = if (result.isNull("result")) null else result.getJSONObject("result")
                    callback(Result.success(opportunityData))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error identifying ritual opportunity", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Create a new ritual instance
     * 
     * @param ritualType Type of ritual from templates
     * @param userId Identifier for the user
     * @param concepts List of concept IDs involved
     * @param context Additional contextual information
     * @param callback Callback for the result
     */
    fun createRitual(
        ritualType: String,
        userId: String,
        concepts: List<String>,
        context: JSONObject? = null,
        callback: (Result<String>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.createRitual(ritualType, userId, concepts, context)
                if (result.getBoolean("success")) {
                    val ritualId = result.getJSONObject("result").toString()
                    callback(Result.success(ritualId))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error creating ritual", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Advance a ritual to the next stage
     * 
     * @param ritualId The ritual instance ID
     * @param interactionData Data from the user interaction
     * @param callback Callback for the result
     */
    fun advanceRitualStage(
        ritualId: String,
        interactionData: JSONObject? = null,
        callback: (Result<JSONObject>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.advanceRitualStage(ritualId, interactionData)
                if (result.getBoolean("success")) {
                    val statusData = result.getJSONObject("result")
                    callback(Result.success(statusData))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error advancing ritual stage", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Get the current status of a ritual
     * 
     * @param ritualId The ritual instance ID
     * @param callback Callback for the result
     */
    fun getRitualStatus(
        ritualId: String,
        callback: (Result<JSONObject>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.getRitualStatus(ritualId)
                if (result.getBoolean("success")) {
                    val statusData = result.getJSONObject("result")
                    callback(Result.success(statusData))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting ritual status", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Get information about available ritual templates
     * 
     * @return LiveData with the ritual templates
     */
    fun getAvailableRituals(): LiveData<JSONObject> {
        // Check if we already have cached data
        if (ritualTemplateCache.value != null) {
            return ritualTemplateCache
        }
        
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.getAvailableRituals()
                if (result.getBoolean("success")) {
                    val templatesData = result.getJSONObject("result")
                    ritualTemplateCache.value = templatesData
                } else {
                    val errorObj = result.getJSONObject("error")
                    Log.e(TAG, "Error getting available rituals: ${errorObj.getString("message")}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting available rituals", e)
            }
        }
        
        return ritualTemplateCache
    }
    
    /**
     * Get a user's ritual history
     * 
     * @param userId The user identifier
     * @param callback Callback for the result
     */
    fun getUserRitualHistory(
        userId: String,
        callback: (Result<JSONObject>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.getUserRitualHistory(userId)
                if (result.getBoolean("success")) {
                    val historyData = result.getJSONObject("result")
                    callback(Result.success(historyData))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting user ritual history", e)
                callback(Result.failure(e))
            }
        }
    }
    
    //----------------------------------------
    // Temporal Flow Maps Methods
    //----------------------------------------
    
    /**
     * Create a new symbol in the mythology
     * 
     * @param name Name of the symbol
     * @param description Description of the symbol
     * @param attributes Optional attributes dictionary
     * @param callback Callback for the result
     */
    fun createSymbol(
        name: String,
        description: String,
        attributes: JSONObject? = null,
        callback: (Result<String>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.createSymbol(name, description, attributes)
                if (result.getBoolean("success")) {
                    val symbolId = result.getJSONObject("result").toString()
                    callback(Result.success(symbolId))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error creating symbol", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Create a new concept in the mythology
     * 
     * @param name Name of the concept
     * @param description Description of the concept
     * @param relatedSymbols Optional list of symbol IDs related to this concept
     * @param callback Callback for the result
     */
    fun createConcept(
        name: String,
        description: String,
        relatedSymbols: List<String>? = null,
        callback: (Result<String>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.createConcept(name, description, relatedSymbols)
                if (result.getBoolean("success")) {
                    val conceptId = result.getJSONObject("result").toString()
                    callback(Result.success(conceptId))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error creating concept", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Generate a temporal map visualization
     * 
     * @param entityIds List of entity IDs to include
     * @param startTime Optional ISO timestamp for start time
     * @param endTime Optional ISO timestamp for end time
     * @param timeScale Time scale for the visualization
     * @param includeRelated Whether to include related entities
     * @param maxRelatedDepth Maximum depth for related entities
     * @param callback Callback for the result
     */
    fun generateTemporalMap(
        entityIds: List<String>,
        startTime: String? = null,
        endTime: String? = null,
        timeScale: String? = null,
        includeRelated: Boolean = true,
        maxRelatedDepth: Int = 1,
        callback: (Result<JSONObject>) -> Unit
    ) {
        scope.launch {
            try {
                ensureInitialized()
                
                val result = bridge.generateTemporalMap(
                    entityIds,
                    startTime,
                    endTime,
                    timeScale,
                    includeRelated,
                    maxRelatedDepth
                )
                
                if (result.getBoolean("success")) {
                    val mapData = result.getJSONObject("result")
                    callback(Result.success(mapData))
                } else {
                    val errorObj = result.getJSONObject("error")
                    throw Exception(errorObj.getString("message"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error generating temporal map", e)
                callback(Result.failure(e))
            }
        }
    }
    
    /**
     * Ensure that the bridge is initialized before performing operations
     * 
     * @throws Exception if initialization failed
     */
    private suspend fun ensureInitialized() {
        if (_initializationState.value !is InitializationState.Initialized) {
            // Wait for initialization to complete or fail
            var attempts = 0
            while (_initializationState.value is InitializationState.Initializing && attempts < 10) {
                attempts++
                withContext(Dispatchers.IO) {
                    Thread.sleep(500) // Wait 500ms and check again
                }
            }
            
            // Check if initialization completed successfully
            if (_initializationState.value !is InitializationState.Initialized) {
                val error = if (_initializationState.value is InitializationState.Error) {
                    (_initializationState.value as InitializationState.Error).message
                } else {
                    "Bridge not initialized"
                }
                throw Exception(error)
            }
        }
    }
    
    /**
     * Initialization state sealed class
     */
    sealed class InitializationState {
        object NotInitialized : InitializationState()
        object Initializing : InitializationState()
        object Initialized : InitializationState()
        data class Error(val message: String) : InitializationState()
    }
}
