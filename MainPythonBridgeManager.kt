package com.antonio.my.ai.girlfriend.free.python

import android.content.Context
import java.util.concurrent.ConcurrentHashMap

/**
 * Main Python bridge manager that coordinates all specialized Python bridge implementations
 * in the AI Girlfriend app.
 */
class PythonBridgeManager private constructor(private val appContext: Context) {
    
    private val bridges = ConcurrentHashMap<String, Any>()
    
    companion object {
        @Volatile
        private var instance: PythonBridgeManager? = null
        
        fun getInstance(context: Context): PythonBridgeManager {
            return instance ?: synchronized(this) {
                instance ?: PythonBridgeManager(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }
    
    init {
        // Initialize core bridges at startup
        getMetacognitiveBridge()
    }
    
    /**
     * Get the current AI response style
     *
     * @return ResponseStyle object with appropriate parameters
     */
    fun getResponseStyle(): MetacognitiveBoundaryBridge.ResponseStyle {
        return getMetacognitiveBridge().getResponseStyle()
    }
    
    /**
     * Save the current state of all Python modules
     *
     * @return State data as a JSON string for persistent storage
     */
    fun saveState(): String {
        return getMetacognitiveBridge().exportState()
    }
    
    /**
     * Restore a previously saved state
     *
     * @param stateJson The state data in JSON format
     * @return Success status
     */
    fun restoreState(stateJson: String): Boolean {
        return getMetacognitiveBridge().importState(stateJson)
    }
    
    /**
     * Clean up all Python resources
     * Call this when the application is being shut down
     */
    fun cleanup() {
        bridges.values.forEach { bridge ->
            if (bridge is MetacognitiveBoundaryBridge) {
                bridge.cleanup()
            }
        }
        bridges.clear()
    }
    
    /**
     * Get or create the Metacognitive boundary bridge
     *
     * @return Metacognitive boundary bridge instance
     */
    fun getMetacognitiveBridge(): MetacognitiveBoundaryBridge {
        val key = "metacognitive_boundary"
        return bridges.getOrPut(key) {
            MetacognitiveBoundaryBridge(appContext)
        } as MetacognitiveBoundaryBridge
    }
    
    /**
     * Process a user message through all relevant Python modules
     *
     * @param userInput The user's message text
     * @param userName The user's name
     * @param botName The AI's name
     * @return Processing result as a JSON string
     */
    fun processMessage(userInput: String, userName: String, botName: String): String {
        val context = mapOf(
            "user_name" to userName,
            "bot_name" to botName,
            "app_version" to "1.2", // Match from AndroidManifest
            "timestamp" to System.currentTimeMillis()
        )
        
        return getMetacognitiveBridge().processUserMessage(userInput, context)
    }
    
    /**
     * Record AI response and any user feedback
     *
     * @param aiResponse The response given to the user
     * @param userFeedback Optional feedback score from the user
     * @return Success status
     */
    fun recordResponse(aiResponse: String, userFeedback: Float? = null): Boolean {
        return getMetacognitiveBridge().recordResponse(aiResponse, userFeedback)
    }
    
    /**
     * Get or create the Memory Manager bridge
     *
     * @return Memory Manager bridge instance
     */
    fun getMemoryBridge(): MemoryManagerBridge {
        val key = "memory_manager"
        return bridges.getOrPut(key) {
            MemoryManagerBridge(appContext)
        } as MemoryManagerBridge
    }
    
    /**
     * Get or create the Emotion Engine bridge
     *
     * @return Emotion Engine bridge instance
     */
    fun getEmotionBridge(): EmotionEngineBridge {
        val key = "emotion_engine"
        return bridges.getOrPut(key) {
            EmotionEngineBridge(appContext)
        } as EmotionEngineBridge
    }
    
    /**
     * Get or create the Personality Model bridge
     *
     * @return Personality Model bridge instance
     */
    fun getPersonalityBridge(): PersonalityModelBridge {
        val key = "personality_model"
        return bridges.getOrPut(key) {
            PersonalityModelBridge(appContext)
        } as PersonalityModelBridge
    }
    
    /**
     * Update the AI's personality settings
     *
     * @param personalitySettings Map of personality parameters to adjust
     * @return Success status
     */
    fun updatePersonality(personalitySettings: Map<String, Any>): Boolean {
        return getPersonalityBridge().updateSettings(personalitySettings)
    }
    
    /**
     * Retrieve memory entries related to a specific topic
     *
     * @param topic The topic to search for
     * @param limit Maximum number of entries to return
     * @return List of memory entries as a JSON string
     */
    fun searchMemory(topic: String, limit: Int = 10): String {
        return getMemoryBridge().searchByTopic(topic, limit)
    }
    
    /**
     * Add a new memory entry from the conversation
     *
     * @param content The content to remember
     * @param importance The importance score (0.0-1.0)
     * @return Success status
     */
    fun addMemory(content: String, importance: Float): Boolean {
        return getMemoryBridge().addEntry(content, importance)
    }
    
    /**
     * Get the current emotional state of the AI
     *
     * @return Emotional state as a JSON string
     */
    fun getEmotionalState(): String {
        return getEmotionBridge().getCurrentState()
    }
    
    /**
     * Initialize Python modules with user profile data
     *
     * @param userProfile User profile data as a JSON string
     * @return Success status
     */
    fun initializeWithUserProfile(userProfile: String): Boolean {
        val metacognitive = getMetacognitiveBridge().initializeWithProfile(userProfile)
        val memory = getMemoryBridge().initializeWithProfile(userProfile)
        val emotion = getEmotionBridge().initializeWithProfile(userProfile)
        val personality = getPersonalityBridge().initializeWithProfile(userProfile)
        
        return metacognitive && memory && emotion && personality
    }
}
