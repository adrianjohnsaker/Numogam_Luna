```kotlin
package com.antonio.my.ai.girlfriend.free.adaptive.systemarchitect.bridge

import android.content.Context
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.adaptive.systemarchitect.model.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.util.*

/**
 * Bridge class that connects the Android app to the Python NumogramEvolutionarySystem.
 * Handles communication, data conversion, and error handling between platforms.
 */
class NumogramSystemBridge {
    companion object {
        private const val TAG = "NumogramSystemBridge"
        private const val NUMOGRAM_SYSTEM_MODULE = "numogram_evolutionary_system"
    }

    private var numogramSystem: PyObject? = null
    private var initialized = false

    /**
     * Initialize the Python environment and the NumogramEvolutionarySystem.
     * Must be called before using any other methods.
     *
     * @param context Android application context
     * @param configPath Optional path to configuration file
     * @return Result containing success status and error message if applicable
     */
    suspend fun initialize(context: Context, configPath: String? = null): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            // Initialize Python
            val py = Python.getInstance()
            
            // Import modules
            val numogramModule = py.getModule(NUMOGRAM_SYSTEM_MODULE)
            
            // Create a new enhanced numogram system
            numogramSystem = numogramModule.callAttr(
                "EnhancedNumogramEvolutionarySystem", 
                configPath
            )
            initialized = true
            
            Result.success(true)
        } catch (e: PyException) {
            Log.e(TAG, "Failed to initialize Python environment: ${e.message}")
            Result.failure(Exception("Failed to initialize: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Java exception during initialization: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Create a new session for interaction with the numogram system.
     *
     * @param userId Unique identifier for the user
     * @param sessionName Optional name for the session
     * @return Result containing the session ID or error
     */
    suspend fun createSession(
        userId: String,
        sessionName: String? = null
    ): Result<String> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val sessionData = numogramSystem?.callAttr(
                "initialize_session",
                userId,
                sessionName
            ) ?: throw Exception("Failed to create session")
            
            val sessionId = sessionData.callAttr("get", "id").toString()
            
            Result.success(sessionId)
        } catch (e: PyException) {
            Log.e(TAG, "Python error creating session: ${e.message}")
            Result.failure(Exception("Failed to create session: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error creating session: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * End an active session.
     *
     * @param sessionId ID of the session to end
     * @return Result containing success status or error
     */
    suspend fun endSession(
        sessionId: String
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val result = numogramSystem?.callAttr(
                "end_session",
                sessionId
            ) ?: throw Exception("Failed to end session")
            
            // Check if error occurred
            if (result.contains("error")) {
                return@withContext Result.failure(Exception(result.callAttr("get", "error").toString()))
            }
            
            Result.success(true)
        } catch (e: PyException) {
            Log.e(TAG, "Python error ending session: ${e.message}")
            Result.failure(Exception("Failed to end session: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error ending session: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Process text input through the numogram system.
     *
     * @param sessionId ID of the session
     * @param text Text input to process
     * @param contextData Optional context data
     * @return Result containing the processing results or error
     */
    suspend fun processText(
        sessionId: String,
        text: String,
        contextData: Map<String, Any>? = null
    ): Result<NumogramProcessingResult> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            // Convert context data to Python dict if provided
            val py = Python.getInstance()
            val contextDict = if (contextData != null) toPyDict(py, contextData) else null
            
            val resultPyObj = numogramSystem?.callAttr(
                "process",
                sessionId,
                text,
                contextDict
            ) ?: throw Exception("Failed to process text")
            
            // Check if error occurred
            if (resultPyObj.contains("error")) {
                return@withContext Result.failure(Exception(resultPyObj.callAttr("get", "error").toString()))
            }
            
            val resultJson = resultPyObj.toString()
            val processingResult = parseNumogramProcessingResult(JSONObject(resultJson))
            
            Result.success(processingResult)
        } catch (e: PyException) {
            Log.e(TAG, "Python error processing text: ${e.message}")
            Result.failure(Exception("Failed to process text: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error processing text: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Set the integration mode for the numogram system.
     *
     * @param sessionId ID of the session
     * @param primaryMode Primary integration mode
     * @param modeWeights Optional weights for different modes
     * @return Result containing success status or error
     */
    suspend fun setIntegrationMode(
        sessionId: String,
        primaryMode: String,
        modeWeights: Map<String, Double>? = null
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            // Convert mode weights to Python dict if provided
            val py = Python.getInstance()
            val weightsDict = if (modeWeights != null) toPyDict(py, modeWeights) else null
            
            val result = numogramSystem?.callAttr(
                "set_integration_mode",
                sessionId,
                primaryMode,
                weightsDict
            ) ?: throw Exception("Failed to set integration mode")
            
            // Check if error occurred
            if (result.contains("error")) {
                return@withContext Result.failure(Exception(result.callAttr("get", "error").toString()))
            }
            
            Result.success(true)
        } catch (e: PyException) {
            Log.e(TAG, "Python error setting integration mode: ${e.message}")
            Result.failure(Exception("Failed to set integration mode: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error setting integration mode: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Get information about a specific numogram zone.
     *
     * @param zone Zone identifier (1-9)
     * @return Result containing the zone information or error
     */
    suspend fun getZoneInfo(
        zone: String
    ): Result<NumogramZoneInfo> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val zoneInfoPyObj = numogramSystem?.callAttr(
                "get_zone_info",
                zone
            ) ?: throw Exception("Failed to get zone info")
            
            val zoneInfoJson = zoneInfoPyObj.toString()
            val zoneInfo = parseNumogramZoneInfo(JSONObject(zoneInfoJson))
            
            Result.success(zoneInfo)
        } catch (e: PyException) {
            Log.e(TAG, "Python error getting zone info: ${e.message}")
            Result.failure(Exception("Failed to get zone info: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error getting zone info: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Get detailed system status.
     *
     * @return Result containing the system status or error
     */
    suspend fun getSystemStatus(): Result<NumogramSystemStatus> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val statusPyObj = numogramSystem?.callAttr(
                "get_system_status"
            ) ?: throw Exception("Failed to get system status")
            
            val statusJson = statusPyObj.toString()
            val systemStatus = parseNumogramSystemStatus(JSONObject(statusJson))
            
            Result.success(systemStatus)
        } catch (e: PyException) {
            Log.e(TAG, "Python error getting system status: ${e.message}")
            Result.failure(Exception("Failed to get system status: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error getting system status: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Get visualization data for the system.
     *
     * @return Result containing visualization data or error
     */
    suspend fun visualizeSystem(): Result<NumogramVisualization> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val vizPyObj = numogramSystem?.callAttr(
                "visualize_system"
            ) ?: throw Exception("Failed to get visualization data")
            
            val vizJson = vizPyObj.toString()
            val visualization = parseNumogramVisualization(JSONObject(vizJson))
            
            Result.success(visualization)
        } catch (e: PyException) {
            Log.e(TAG, "Python error getting visualization data: ${e.message}")
            Result.failure(Exception("Failed to get visualization data: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error getting visualization data: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Get zone trajectory information for a user.
     *
     * @param userId User ID
     * @param lookback Number of past transitions to consider
     * @return Result containing trajectory data or error
     */
    suspend fun getZoneTrajectory(
        userId: String,
        lookback: Int = 10
    ): Result<NumogramTrajectory> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val trajectoryPyObj = numogramSystem?.callAttr(
                "get_zone_trajectory",
                userId,
                lookback
            ) ?: throw Exception("Failed to get zone trajectory")
            
            val trajectoryJson = trajectoryPyObj.toString()
            val trajectory = parseNumogramTrajectory(JSONObject(trajectoryJson))
            
            Result.success(trajectory)
        } catch (e: PyException) {
            Log.e(TAG, "Python error getting zone trajectory: ${e.message}")
            Result.failure(Exception("Failed to get zone trajectory: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error getting zone trajectory: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Export session data as JSON.
     *
     * @param sessionId ID of the session to export
     * @param format Export format (currently only "json" is supported)
     * @return Result containing the exported data or error
     */
    suspend fun exportSessionData(
        sessionId: String,
        format: String = "json"
    ): Result<String> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val exportDataPyObj = numogramSystem?.callAttr(
                "export_session_data",
                sessionId,
                format
            ) ?: throw Exception("Failed to export session data")
            
            // Check if result is a JSON error message
            val exportData = exportDataPyObj.toString()
            val jsonData = JSONObject(exportData)
            if (jsonData.has("error")) {
                return@withContext Result.failure(Exception(jsonData.getString("error")))
            }
            
            Result.success(exportData)
        } catch (e: PyException) {
            Log.e(TAG, "Python error exporting session data: ${e.message}")
            Result.failure(Exception("Failed to export session data: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting session data: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Save current system state to a directory.
     *
     * @param directory Directory path for saving state
     * @return Result containing success status or error
     */
    suspend fun saveSystemState(
        directory: String
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val resultPyObj = numogramSystem?.callAttr(
                "save_system_state",
                directory
            ) ?: throw Exception("Failed to save system state")
            
            // Check if status is success
            val statusStr = resultPyObj.callAttr("get", "status").toString()
            val success = statusStr == "success"
            
            if (!success) {
                val errorMsg = resultPyObj.callAttr("get", "message").toString()
                return@withContext Result.failure(Exception(errorMsg))
            }
            
            Result.success(true)
        } catch (e: PyException) {
            Log.e(TAG, "Python error saving system state: ${e.message}")
            Result.failure(Exception("Failed to save system state: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error saving system state: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Load system state from a directory.
     *
     * @param directory Directory path containing saved state
     * @return Result containing success status or error
     */
    suspend fun loadSystemState(
        directory: String
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        checkInitialized()
        
        try {
            val resultPyObj = numogramSystem?.callAttr(
                "load_system_state",
                directory
            ) ?: throw Exception("Failed to load system state")
            
            // Check if status is success
            val statusStr = resultPyObj.callAttr("get", "status").toString()
            val success = statusStr == "success"
            
            if (!success) {
                val errorMsg = resultPyObj.callAttr("get", "message").toString()
                return@withContext Result.failure(Exception(errorMsg))
            }
            
            Result.success(true)
        } catch (e: PyException) {
            Log.e(TAG, "Python error loading system state: ${e.message}")
            Result.failure(Exception("Failed to load system state: ${e.message}"))
        } catch (e: Exception) {
            Log.e(TAG, "Error loading system state: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Check if the bridge is initialized and throw an exception if not.
     * 
     * @throws IllegalStateException if not initialized
     */
    private fun checkInitialized() {
        if (!initialized || numogramSystem == null) {
            throw IllegalStateException("NumogramSystemBridge not initialized. Call initialize() first.")
        }
    }

    /**
     * Convert a Kotlin Map to a Python dict.
     * 
     * @param py Python instance
     * @param map Map to convert
     * @return Python dictionary object
     */
    private fun toPyDict(py: Python, map: Map<String, Any>): PyObject {
        val dict = py.builtins.callAttr("dict")
        
        for ((key, value) in map) {
            when (value) {
                is String -> dict.callAttr("__setitem__", key, value)
                is Int -> dict.callAttr("__setitem__", key, value)
                is Double -> dict.callAttr("__setitem__", key, value)
                is Boolean -> dict.callAttr("__setitem__", key, value)
                is List<*> -> {
                    val pyList = py.builtins.callAttr("list")
                    (value as List<*>).forEach { item ->
                        when (item) {
                            is String -> pyList.callAttr("append", item)
                            is Int -> pyList.callAttr("append", item)
                            is Double -> pyList.callAttr("append", item)
                            is Boolean -> pyList.callAttr("append", item)
                            else -> pyList.callAttr("append", item.toString())
                        }
                    }
                    dict.callAttr("__setitem__", key, pyList)
                }
                is Map<*, *> -> {
                    @Suppress("UNCHECKED_CAST")
                    val pySubDict = toPyDict(py, value as Map<String, Any>)
                    dict.callAttr("__setitem__", key, pySubDict)
                }
                else -> dict.callAttr("__setitem__", key, value.toString())
            }
        }
        
        return dict
    }

    // Parsing helper methods for JSON to Kotlin model objects

    private fun parseNumogramProcessingResult(json: JSONObject): NumogramProcessingResult {
        // Parse symbolic patterns
        val symbolicPatterns = mutableListOf<SymbolicPattern>()
        if (json.has("symbolic_patterns")) {
            val patternsArray = json.getJSONArray("symbolic_patterns")
            for (i in 0 until patternsArray.length()) {
                val patternObj = patternsArray.getJSONObject(i)
                symbolicPatterns.add(
                    SymbolicPattern(
                        id = patternObj.getString("id"),
                        coreSymbols = parseStringList(patternObj.getJSONArray("core_symbols")),
                        relatedSymbols = parseStringList(patternObj.getJSONArray("related_symbols")),
                        numogramZone = patternObj.getString("numogram_zone"),
                        intensity = patternObj.getDouble("intensity"),
                        digitalSignature = if (patternObj.has("digital_signature")) 
                            parseIntList(patternObj.getJSONArray("digital_signature")) else listOf()
                    )
                )
            }
        }
        
        // Parse emotional state
        val emotionalState = if (json.has("emotional_state")) {
            val emotionObj = json.getJSONObject("emotional_state")
            EmotionalState(
                id = emotionObj.getString("id"),
                primaryEmotion = emotionObj.getString("primary_emotion"),
                emotionalSpectrum = parseDoubleMap(emotionObj.getJSONObject("emotional_spectrum")),
                intensity = emotionObj.getDouble("intensity"),
                numogramZone = emotionObj.getString("numogram_zone"),
                digitalRatios = if (emotionObj.has("digital_ratios")) 
                    parseDoubleList(emotionObj.getJSONArray("digital_ratios")) else listOf()
            )
        } else null
        
        // Parse numogram transition
        val numogramTransition = if (json.has("numogram_transition")) {
            val transObj = json.getJSONObject("numogram_transition")
            NumogramTransition(
                currentZone = transObj.getString("current_zone"),
                nextZone = transObj.getString("next_zone"),
                feedback = transObj.getDouble("feedback"),
                timestamp = transObj.getString("timestamp")
            )
        } else null
        
        // Parse system predictions
        val systemPredictions = if (json.has("system_predictions")) {
            parseStringMap(json.getJSONObject("system_predictions"))
        } else mapOf()
        
        // Parse integration result
        val integrationResult = if (json.has("integration_result")) {
            val intObj = json.getJSONObject("integration_result")
            IntegrationResult(
                finalZone = intObj.getString("final_zone"),
                finalConfidence = intObj.getDouble("final_confidence"),
                actualZone = intObj.getString("actual_zone"),
                primaryMode = intObj.getString("primary_mode"),
                zoneVotes = parseDoubleMap(intObj.getJSONObject("zone_votes"))
            )
        } else null
        
        return NumogramProcessingResult(
            sessionId = json.getString("session_id"),
            interactionId = json.getString("interaction_id"),
            textInput = json.getString("text_input"),
            symbolicPatterns = symbolicPatterns,
            emotionalState = emotionalState,
            systemPredictions = systemPredictions,
            integrationResult = integrationResult,
            numogramTransition = numogramTransition
        )
    }

    private fun parseNumogramZoneInfo(json: JSONObject): NumogramZoneInfo {
        // Parse tensor representation
        val tensorRepresentation = if (json.has("tensor_representation")) {
            val tensorObj = json.getJSONObject("tensor_representation")
            TensorRepresentation(
                position = if (tensorObj.has("position")) 
                    parseDoubleList(tensorObj.getJSONArray("position")) else listOf(),
                energy = if (tensorObj.has("energy")) tensorObj.getDouble("energy") else null,
                stability = if (tensorObj.has("stability")) tensorObj.getDouble("stability") else null,
                flux = if (tensorObj.has("flux")) tensorObj.getDouble("flux") else null,
                dimension = if (tensorObj.has("dimension")) tensorObj.getInt("dimension") else 3
            )
        } else null
        
        // Parse active hyperedges
        val activeHyperedges = mutableListOf<Hyperedge>()
        if (json.has("active_hyperedges")) {
            val edgesArray = json.getJSONArray("active_hyperedges")
            for (i in 0 until edgesArray.length()) {
                val edgeObj = edgesArray.getJSONObject(i)
                activeHyperedges.add(
                    Hyperedge(
                        id = edgeObj.getString("id"),
                        name = edgeObj.getString("name"),
                        zones = parseStringList(edgeObj.getJSONArray("zones")),
                        strength = edgeObj.getDouble("strength")
                    )
                )
            }
        }
        
        return NumogramZoneInfo(
            zone = json.getString("zone"),
            zoneData = if (json.has("zone_data")) parseJSONObject(json.getJSONObject("zone_data")) else mapOf(),
            tensorRepresentation = tensorRepresentation,
            symbolicAssociations = if (json.has("symbolic_associations")) 
                parseStringList(json.getJSONArray("symbolic_associations")) else listOf(),
            emotionalAssociations = if (json.has("emotional_associations")) 
                parseDoubleMap(json.getJSONObject("emotional_associations")) else mapOf(),
            activeHyperedges = activeHyperedges
        )
    }

    private fun parseNumogramSystemStatus(json: JSONObject): NumogramSystemStatus {
        // Parse component status
        val componentStatus = if (json.has("components")) {
            parseStringMap(json.getJSONObject("components"))
        } else mapOf()
        
        // Parse neural stats
        val neuralStats = if (json.has("neural_stats")) {
            parseJSONObject(json.getJSONObject("neural_stats"))
        } else mapOf()
        
        // Parse attention metrics
        val attentionMetrics = if (json.has("attention_metrics")) {
            parseJSONObject(json.getJSONObject("attention_metrics"))
        } else mapOf()
        
        return NumogramSystemStatus(
            systemId = json.getString("system_id"),
            version = json.getString("version"),
            createdAt = json.getString("created_at"),
            currentTime = json.getString("current_time"),
            activeSessions = json.getInt("active_sessions"),
            totalSessions = json.getInt("total_sessions"),
            numogramUsers = json.getInt("numogram_users"),
            emotionalMemories = if (json.has("emotional_memories")) json.getInt("emotional_memories") else 0,
            components = componentStatus,
            primaryMode = json.getString("primary_mode"),
            modeWeights = if (json.has("mode_weights")) 
                parseDoubleMap(json.getJSONObject("mode_weights")) else mapOf(),
            tensorDimension = json.getInt("tensor_dimension"),
            attentionModelType = json.getString("attention_model_type"),
            neuralStats = neuralStats,
            attentionMetrics = attentionMetrics
        )
    }

    private fun parseNumogramVisualization(json: JSONObject): NumogramVisualization {
        // Parse tensor space
        val tensorSpace = if (json.has("tensor_space")) {
            parseJSONObject(json.getJSONObject("tensor_space"))
        } else mapOf()
        
        // Parse emotional landscape
        val emotionalLandscape = if (json.has("emotional_landscape")) {
            parseJSONObject(json.getJSONObject("emotional_landscape"))
        } else mapOf()
        
        // Parse attention system
        val attentionSystem = if (json.has("attention_system")) {
            parseJSONObject(json.getJSONObject("attention_system"))
        } else mapOf()
        
        // Parse tesseract
        val tesseract = if (json.has("tesseract")) {
            parseJSONObject(json.getJSONObject("tesseract"))
        } else mapOf()
        
        // Parse system config
        val systemConfig = if (json.has("system_config")) {
            parseJSONObject(json.getJSONObject("system_config"))
        } else mapOf()
        
        return NumogramVisualization(
            tensorSpace = tensorSpace,
            emotionalLandscape = emotionalLandscape,
            neuralEvolution = json.getBoolean("neural_evolution"),
            attentionSystem = attentionSystem,
            tesseract = tesseract,
            systemConfig = systemConfig
        )
    }

    private fun parseNumogramTrajectory(json: JSONObject): NumogramTrajectory {
        // Parse signature path
        val signaturePath = if (json.has("signature_path")) {
            val pathObj = json.getJSONObject("signature_path")
            SignaturePath(
                name = pathObj.getString("name"),
                zoneSequence = parseStringList(pathObj.getJSONArray("zone_sequence")),
                emotionSequence = parseStringList(pathObj.getJSONArray("emotion_sequence")),
                zoneNames = parseStringList(pathObj.getJSONArray("zone_names")),
                description = pathObj.getString("description"),
                confidence = pathObj.getDouble("confidence")
            )
        } else null
        
        // Parse temporal pattern
        val temporalPattern = if (json.has("temporal_pattern") && !json.isNull("temporal_pattern")) {
            parseJSONObject(json.getJSONObject("temporal_pattern"))
        } else null
        
        // Parse active hyperedges
        val activeHyperedges = mutableListOf<Hyperedge>()
        if (json.has("active_hyperedges")) {
            val edgesArray = json.getJSONArray("active_hyperedges")
            for (i in 0 until edgesArray.length()) {
                val edgeObj = edgesArray.getJSONObject(i)
                activeHyperedges.add(
                    Hyperedge(
                        id = edgeObj.getString("id"),
                        name = edgeObj.getString("name"),
                        zones = parseStringList(edgeObj.getJSONArray("zones")),
                        strength = edgeObj.getDouble("strength")
                    )
                )
            }
        }
        
        // Parse predicted trajectory
        val predictedTrajectory = mutableListOf<ZonePrediction>()
        if (json.has("predicted_trajectory")) {
            val predArray = json.getJSONArray("predicted_trajectory")
            for (i in 0 until predArray.length()) {
                val predObj = predArray.getJSONObject(i)
                predictedTrajectory.add(
                    ZonePrediction(
                        zone = predObj.getString("zone"),
                        confidence = predObj.getDouble("confidence"),
                        source = predObj.getString("source"),
                        emotion = if (predObj.has("emotion")) predObj.getString("emotion") else null
                    )
                )
            }
        }
        
        return NumogramTrajectory(
            userId = json.getString("user_id"),
            currentZone = json.getString("current_zone"),
            zoneSequence = parseStringList(json.getJSONArray("zone_sequence")),
            timestamps = parseStringList(json.getJSONArray("timestamps")),
            signaturePath = signaturePath,
            temporalPattern = temporalPattern,
            activeHyperedges = activeHyperedges,
            predictedTrajectory = predictedTrajectory
        )
    }

    // Helper methods for parsing JSON elements

    private fun parseStringList(jsonArray: JSONArray): List<String> {
        val result = mutableListOf<String>()
        for (i in 0 until jsonArray.length()) {
            result.add(jsonArray.getString(i))
        }
        return result
    }

    private fun parseIntList(jsonArray: JSONArray): List<Int> {
        val result = mutableListOf<Int>()
        for (i in 0 until jsonArray.length()) {
            result.add(jsonArray.getInt(i))
        }
        return result
    }

    private fun parseDoubleList(jsonArray: JSONArray): List<Double> {
        val result = mutableListOf<Double>()
        for (i in 0 until jsonArray.length()) {
            result.add(jsonArray.getDouble(i))
        }
        return result
    }

    private fun parseStringMap(jsonObject: JSONObject): Map<String, String> {
        val result = mutableMapOf<String, String>()
        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            result[key] = jsonObject.getString(key)
        }
        return result
    }

    private fun parseDoubleMap(jsonObject: JSONObject): Map<String, Double> {
        val result = mutableMapOf<String, Double>()
        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            result[key] = jsonObject.getDouble(key)
        }
        return result
    }

    private fun parseJSONObject(jsonObject: JSONObject): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val value = jsonObject.get(key)
            when (value) {
                is JSONObject -> result[key] = parseJSONObject(value)
                is JSONArray -> result[key] = parseJSONArray(value)
                else -> {
                    if (value is Int || value is Long) {
                        result[key] = jsonObject.getLong(key)
                    } else if (value is Double || value is Float) {
                        result[key] = jsonObject.getDouble(key)
                    } else if (value is Boolean) {
                        result[key] = jsonObject.getBoolean(key)
                    } else {
                        result[key] = value.toString()
                    }
                }
            }
        }
        return result
    }

    private fun parseJSONArray(jsonArray: JSONArray): List<Any> {
        val result = mutableListOf<Any>()
        for (i in 0 until jsonArray.length()) {
            val item = jsonArray.get(i)
            when (item) {
                is JSONObject -> result.add(parseJSONObject(item))
                is JSONArray -> result.add(parseJSONArray(item))
                is Int, is Long -> result.add(jsonArray.getLong(i))
                is Double, is Float -> result.add(jsonArray.getDouble(i))
                is Boolean -> result.add(jsonArray.getBoolean(i))
                else -> result.add(item.toString())
            }
        }
        return result
    }
}
