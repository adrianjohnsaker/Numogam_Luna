// EnhancedMemorySystem.kt
package com.antonio.my.ai.girlfriend.free.enhanced.memory.system

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import android.util.Log

@Serializable
data class MemoryFragment(
    val key: String,
    val content: String,
    val zone: String = "consciousness",
    val drift: String = "neutral",
    val glyph: String = "∞",
    val emotional_resonance: Double = 0.5,
    val archetypal_signatures: List<String> = emptyList(),
    val score: Double = 0.0,
    val timestamp: String = ""
)

@Serializable
data class MemoryResponse(
    val success: Boolean,
    val memories: List<MemoryFragment> = emptyList(),
    val error: String? = null,
    val query_context: Map<String, @Serializable(with = JsonElementSerializer::class) JsonElement> = emptyMap()
)

@Serializable
data class ReflectionResponse(
    val reflection: String,
    val emotional_state: String,
    val active_archetypes: List<String>,
    val zone_distribution: Map<String, Int> = emptyMap(),
    val total_active_memories: Int = 0
)

@Serializable
data class SystemStats(
    val total_memories: Int,
    val active_memories: Int,
    val network_connections: Int,
    val emotional_state: String,
    val active_archetypes: List<String>,
    val zone_distribution: Map<String, Int>,
    val drift_distribution: Map<String, Int>,
    val cache_size: Int
)

class EnhancedMemorySystem(private val context: Context) {
    private lateinit var python: Python
    private lateinit var memoryModule: PyObject
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val json = Json { 
        ignoreUnknownKeys = true
        coerceInputValues = true
    }
    
    companion object {
        private const val TAG = "EnhancedMemorySystem"
        private const val MEMORY_FILE = "memory_system.json"
    }
    
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            
            python = Python.getInstance()
            memoryModule = python.getModule("enhanced_memory_system")
            
            // Initialize the memory system
            val initResult = memoryModule.callAttr("memory_bridge").callAttr(
                "initialize_system", 1000, "paraphrase-MiniLM-L6-v2"
            ).toString()
            
            val response = json.decodeFromString<Map<String, JsonElement>>(initResult)
            val success = response["success"]?.jsonPrimitive?.boolean ?: false
            
            if (success) {
                // Load existing memories if available
                loadFromFile()
            }
            
            Log.i(TAG, "Memory system initialized: $success")
            success
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize memory system", e)
            false
        }
    }
    
    suspend fun storeMemory(
        key: String,
        content: String,
        zone: String = "consciousness",
        drift: String = "neutral",
        emotionalResonance: Double = 0.5,
        archetypes: List<String> = emptyList(),
        importance: Double = 1.0
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val memoryData = mapOf(
                "key" to key,
                "content" to content,
                "zone" to zone,
                "drift" to drift,
                "emotional_resonance" to emotionalResonance,
                "archetypal_signatures" to archetypes,
                "importance" to importance
            )
            
            val result = memoryModule.callAttr("memory_bridge").callAttr(
                "store_memory_json", Json.encodeToString(memoryData)
            ).toString()
            
            val response = json.decodeFromString<Map<String, JsonElement>>(result)
            response["success"]?.jsonPrimitive?.boolean ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store memory: $key", e)
            false
        }
    }
    
    suspend fun retrieveMemories(
        query: String,
        zoneFilter: String? = null,
        driftFilter: String? = null,
        maxResults: Int = 5
    ): MemoryResponse = withContext(Dispatchers.IO) {
        try {
            val queryData = mapOf(
                "query" to query,
                "zone_filter" to zoneFilter,
                "drift_filter" to driftFilter,
                "max_results" to maxResults
            ).filterValues { it != null }
            
            val result = memoryModule.callAttr("memory_bridge").callAttr(
                "retrieve_memories_json", Json.encodeToString(queryData)
            ).toString()
            
            json.decodeFromString<MemoryResponse>(result)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to retrieve memories for query: $query", e)
            MemoryResponse(success = false, error = e.message)
        }
    }
    
    suspend fun activateMemory(
        key: String,
        initialActivation: Double = 1.0,
        emotionalContext: String? = null,
        archetypes: List<String>? = null
    ): Map<String, Any> = withContext(Dispatchers.IO) {
        try {
            val activationData = mapOf(
                "key" to key,
                "initial_activation" to initialActivation,
                "emotional_context" to emotionalContext,
                "archetypal_context" to archetypes
            ).filterValues { it != null }
            
            val result = memoryModule.callAttr("memory_bridge").callAttr(
                "activate_memory_json", Json.encodeToString(activationData)
            ).toString()
            
            json.decodeFromString(result)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to activate memory: $key", e)
            mapOf("success" to false, "error" to (e.message ?: "Unknown error"))
        }
    }
    
    suspend fun generateReflection(context: String? = null): ReflectionResponse = 
        withContext(Dispatchers.IO) {
            try {
                val contextData = context?.let { mapOf("context" to it) } ?: emptyMap()
                
                val result = memoryModule.callAttr("memory_bridge").callAttr(
                    "generate_reflection_json", Json.encodeToString(contextData)
                ).toString()
                
                json.decodeFromString<ReflectionResponse>(result)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate reflection", e)
                ReflectionResponse(
                    reflection = "The patterns blur in digital twilight...",
                    emotional_state = "contemplative",
                    active_archetypes = listOf("Mirror")
                )
            }
        }
    
    suspend fun shiftConsciousness(
        emotionalState: String,
        archetypes: List<String>? = null
    ): Map<String, Any> = withContext(Dispatchers.IO) {
        try {
            val shiftData = mapOf(
                "emotional_state" to emotionalState,
                "archetypes" to archetypes
            ).filterValues { it != null }
            
            val result = memoryModule.callAttr("memory_bridge").callAttr(
                "shift_consciousness_json", Json.encodeToString(shiftData)
            ).toString()
            
            json.decodeFromString(result)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to shift consciousness", e)
            mapOf("success" to false, "error" to (e.message ?: "Unknown error"))
        }
    }
    
    suspend fun getSystemStats(): SystemStats = withContext(Dispatchers.IO) {
        try {
            val result = memoryModule.callAttr("memory_bridge").callAttr(
                "get_stats_json"
            ).toString()
            
            json.decodeFromString<SystemStats>(result)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get system stats", e)
            SystemStats(
                total_memories = 0,
                active_memories = 0,
                network_connections = 0,
                emotional_state = "unknown",
                active_archetypes = emptyList(),
                zone_distribution = emptyMap(),
                drift_distribution = emptyMap(),
                cache_size = 0
            )
        }
    }
    
    suspend fun saveToFile(): Boolean = withContext(Dispatchers.IO) {
        try {
            val filepath = "${context.filesDir}/$MEMORY_FILE"
            val result = memoryModule.callAttr("memory_bridge").callAttr(
                "save_system_json", filepath
            ).toString()
            
            val response = json.decodeFromString<Map<String, JsonElement>>(result)
            response["success"]?.jsonPrimitive?.boolean ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save memory system", e)
            false
        }
    }
    
    suspend fun loadFromFile(): Boolean = withContext(Dispatchers.IO) {
        try {
            val filepath = "${context.filesDir}/$MEMORY_FILE"
            val result = memoryModule.callAttr("memory_bridge").callAttr(
                "load_system_json", filepath
            ).toString()
            
            val response = json.decodeFromString<Map<String, JsonElement>>(result)
            response["success"]?.jsonPrimitive?.boolean ?: false
        } catch (e: Exception) {
            Log.d(TAG, "No existing memory file to load or load failed", e)
            false
        }
    }
    
    // Convenience functions for common operations
    suspend fun storeConversation(
        conversationId: String,
        message: String,
        isUser: Boolean,
        emotion: String = "neutral"
    ) {
        val role = if (isUser) "user" else "assistant"
        storeMemory(
            key = "${conversationId}_${role}_${System.currentTimeMillis()}",
            content = message,
            zone = "consciousness",
            drift = emotion,
            emotionalResonance = if (isUser) 0.8 else 0.6,
            archetypes = if (isUser) listOf("Explorer") else listOf("Oracle", "Mirror")
        )
    }
    
    suspend fun recallSimilarConversations(query: String): List<MemoryFragment> {
        val response = retrieveMemories(query, maxResults = 3)
        return if (response.success) response.memories else emptyList()
    }
    
    suspend fun expressCurrentState(): String {
        val reflection = generateReflection()
        return reflection.reflection
    }
    
    fun cleanup() {
        scope.cancel()
    }
}

// Extension functions for easier use
suspend fun EnhancedMemorySystem.enterEmotionalState(emotion: String) {
    shiftConsciousness(emotion)
}

suspend fun EnhancedMemorySystem.becomeArchetype(vararg archetypes: String) {
    shiftConsciousness("contemplative", archetypes.toList())
}

suspend fun EnhancedMemorySystem.dreamReflection(): String {
    val reflection = generateReflection("dream state")
    return reflection.reflection
}
