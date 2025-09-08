package com.antionio.my.ai.girlfriend.free

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * Repository for managing Amelia's memory system through Chaquopy bridge
 */
class MainActivityRepository(private val context: Context) {
    
    companion object {
        private const val TAG = "AmeliaMemoryRepo"
        private const val PREFS_NAME = "amelia_memory_prefs"
        private const val KEY_MEMORIES_IMPORTED = "memories_imported"
        private const val KEY_CURRENT_SESSION = "current_session_id"
        private const val KEY_USER_ID = "user_id"
        private const val ASSETS_MEMORY_FOLDER = "memories"
        private const val DEFAULT_USER_ID = "default_user"
    }
    
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val python: Python by lazy { Python.getInstance() }
    private val memoryModule: PyObject by lazy { python.getModule("memory_module") }
    private var memory: PyObject? = null
    
    // Data classes for results
    data class ConversationResult(
        val success: Boolean,
        val sessionId: String? = null,
        val title: String? = null,
        val summary: String? = null,
        val topics: List<String> = emptyList(),
        val messageCount: Int = 0,
        val error: String? = null
    )
    
    data class ImportResult(
        val totalFiles: Int,
        val successfulFiles: Int,
        val failedFiles: Int,
        val totalConversations: Int,
        val importedConversations: List<ConversationSummary>
    )
    
    data class ConversationSummary(
        val filename: String,
        val sessionId: String,
        val title: String,
        val summary: String
    )
    
    data class MemoryStats(
        val totalConversations: Int,
        val totalMessages: Int,
        val uniqueTopics: Int,
        val indexedKeywords: Int,
        val storageSizeMb: Float,
        val archivedConversations: Int,
        val hasActiveSession: Boolean
    )
    
    data class SearchResult(
        val sessionId: String,
        val startedAt: String,
        val summary: String?,
        val matchingMessages: List<Message>
    )
    
    data class Message(
        val role: String,
        val content: String,
        val timestamp: String
    )
    
    /**
     * Initialize the memory module
     */
    suspend fun initializeMemory(storagePath: String = "amelia_memory"): Boolean = withContext(Dispatchers.IO) {
        try {
            memory = memoryModule.callAttr("create_memory_module", storagePath)
            
            // Import memories from assets if not already done
            if (!prefs.getBoolean(KEY_MEMORIES_IMPORTED, false)) {
                importMemoriesFromAssets()
            }
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize memory module", e)
            false
        }
    }
    
    /**
     * Get or create the current user ID
     */
    fun getUserId(): String {
        return prefs.getString(KEY_USER_ID, DEFAULT_USER_ID) ?: DEFAULT_USER_ID
    }
    
    /**
     * Set the user ID
     */
    fun setUserId(userId: String) {
        prefs.edit().putString(KEY_USER_ID, userId).apply()
    }
    
    /**
     * Start a new conversation
     */
    suspend fun startConversation(userId: String = getUserId()): String? = withContext(Dispatchers.IO) {
        try {
            val sessionId = memoryModule.callAttr("start_new_conversation", memory, userId).toString()
            prefs.edit().putString(KEY_CURRENT_SESSION, sessionId).apply()
            sessionId
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start conversation", e)
            null
        }
    }
    
    /**
     * Add a user message to the current conversation
     */
    suspend fun addUserMessage(content: String): Boolean = withContext(Dispatchers.IO) {
        try {
            memoryModule.callAttr("add_user_message", memory, content).toBoolean()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add user message", e)
            false
        }
    }
    
    /**
     * Add an Amelia message to the current conversation
     */
    suspend fun addAmeliaMessage(content: String): Boolean = withContext(Dispatchers.IO) {
        try {
            memoryModule.callAttr("add_amelia_message", memory, content).toBoolean()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add Amelia message", e)
            false
        }
    }
    
    /**
     * Get current conversation context
     */
    suspend fun getCurrentContext(limit: Int = 10): List<Message> = withContext(Dispatchers.IO) {
        try {
            val contextList = memoryModule.callAttr("get_current_context", memory, limit).asList()
            contextList.map { msg ->
                val msgMap = msg.asMap()
                Message(
                    role = msgMap["role"].toString(),
                    content = msgMap["content"].toString(),
                    timestamp = msgMap["timestamp"].toString()
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get conversation context", e)
            emptyList()
        }
    }
    
    /**
     * End current conversation with optional summary and topics
     */
    suspend fun endConversation(summary: String? = null, topics: List<String>? = null): Boolean = withContext(Dispatchers.IO) {
        try {
            memoryModule.callAttr("end_current_conversation", memory, summary, topics)
            prefs.edit().remove(KEY_CURRENT_SESSION).apply()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to end conversation", e)
            false
        }
    }
    
    /**
     * Import memories from assets folder
     */
    suspend fun importMemoriesFromAssets(): ImportResult = withContext(Dispatchers.IO) {
        try {
            val assetFiles = mutableListOf<List<Any>>()
            val assetManager = context.assets
            
            // List all files in memories folder
            val fileList = assetManager.list(ASSETS_MEMORY_FOLDER) ?: arrayOf()
            
            for (filename in fileList) {
                if (filename.endsWith(".txt")) {
                    try {
                        val inputStream = assetManager.open("$ASSETS_MEMORY_FOLDER/$filename")
                        val content = inputStream.bufferedReader().use { it.readText() }
                        assetFiles.add(listOf(filename, content))
                    } catch (e: IOException) {
                        Log.e(TAG, "Failed to read asset file: $filename", e)
                    }
                }
            }
            
            // Import all files
            val result = memoryModule.callAttr("import_all_assets", memory, assetFiles, getUserId())
            
            // Mark as imported
            prefs.edit().putBoolean(KEY_MEMORIES_IMPORTED, true).apply()
            
            // Parse result
            parseImportResult(result)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to import memories from assets", e)
            ImportResult(0, 0, 0, 0, emptyList())
        }
    }
    
    /**
     * Paste a single conversation from clipboard
     */
    suspend fun pasteSingleConversation(
        pastedText: String, 
        title: String? = null
    ): ConversationResult = withContext(Dispatchers.IO) {
        try {
            val result = memoryModule.callAttr(
                "paste_single_conversation", 
                memory, 
                pastedText, 
                getUserId(), 
                title
            )
            parseConversationResult(result)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to paste conversation", e)
            ConversationResult(false, error = e.message)
        }
    }
    
    /**
     * Quick paste - returns simple success message
     */
    suspend fun quickPaste(text: String): String = withContext(Dispatchers.IO) {
        try {
            memoryModule.callAttr("quick_paste", memory, text).toString()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to quick paste", e)
            "✗ Failed to import conversation: ${e.message}"
        }
    }
    
    /**
     * Search memories by keyword
     */
    suspend fun searchMemories(query: String): List<SearchResult> = withContext(Dispatchers.IO) {
        try {
            val results = memoryModule.callAttr("search_memories", memory, query).asList()
            results.map { result ->
                val resultMap = result.asMap()
                val messages = resultMap["matching_messages"]?.asList()?.map { msg ->
                    val msgMap = msg.asMap()
                    Message(
                        role = msgMap["role"].toString(),
                        content = msgMap["content"].toString(),
                        timestamp = msgMap["timestamp"].toString()
                    )
                } ?: emptyList()
                
                SearchResult(
                    sessionId = resultMap["session_id"].toString(),
                    startedAt = resultMap["started_at"].toString(),
                    summary = resultMap["summary"]?.toString(),
                    matchingMessages = messages
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to search memories", e)
            emptyList()
        }
    }
    
    /**
     * Get memory statistics
     */
    suspend fun getMemoryStats(): MemoryStats? = withContext(Dispatchers.IO) {
        try {
            val stats = memoryModule.callAttr("get_memory_stats", memory).asMap()
            MemoryStats(
                totalConversations = stats["total_conversations"]?.toInt() ?: 0,
                totalMessages = stats["total_messages"]?.toInt() ?: 0,
                uniqueTopics = stats["unique_topics"]?.toInt() ?: 0,
                indexedKeywords = stats["indexed_keywords"]?.toInt() ?: 0,
                storageSizeMb = stats["storage_size_mb"]?.toFloat() ?: 0f,
                archivedConversations = stats["archived_conversations"]?.toInt() ?: 0,
                hasActiveSession = stats["active_session"]?.toBoolean() ?: false
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get memory stats", e)
            null
        }
    }
    
    /**
     * Prune old memories
     */
    suspend fun pruneMemories(dryRun: Boolean = true): Map<String, Any> = withContext(Dispatchers.IO) {
        try {
            val result = memoryModule.callAttr("prune_old_memories", memory, dryRun)
            result.asMap().mapValues { it.value.toString() }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to prune memories", e)
            emptyMap()
        }
    }
    
    /**
     * Get recent conversations
     */
    suspend fun getRecentConversations(limit: Int = 10): List<Map<String, Any>> = withContext(Dispatchers.IO) {
        try {
            val recentConvs = memory?.callAttr("get_recent_conversations", limit)?.asList()
            recentConvs?.map { conv ->
                conv.asMap().mapValues { it.value.toString() }
            } ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get recent conversations", e)
            emptyList()
        }
    }
    
    /**
     * Search by topic
     */
    suspend fun searchByTopic(topic: String): List<Map<String, Any>> = withContext(Dispatchers.IO) {
        try {
            val results = memory?.callAttr("search_by_topic", topic)?.asList()
            results?.map { result ->
                result.asMap().mapValues { it.value.toString() }
            } ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to search by topic", e)
            emptyList()
        }
    }
    
    /**
     * Export memory to a specified path
     */
    suspend fun exportMemory(exportPath: String): Boolean = withContext(Dispatchers.IO) {
        try {
            memory?.callAttr("export_memory", exportPath)?.toBoolean() ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export memory", e)
            false
        }
    }
    
    // Helper functions to parse Python results
    
    private fun parseConversationResult(result: PyObject): ConversationResult {
        val resultMap = result.asMap()
        val success = resultMap["success"]?.toBoolean() ?: false
        
        return if (success) {
            ConversationResult(
                success = true,
                sessionId = resultMap["session_id"]?.toString(),
                title = resultMap["title"]?.toString(),
                summary = resultMap["summary"]?.toString(),
                topics = resultMap["topics"]?.asList()?.map { it.toString() } ?: emptyList(),
                messageCount = resultMap["message_count"]?.toInt() ?: 0
            )
        } else {
            ConversationResult(
                success = false,
                error = resultMap["error"]?.toString()
            )
        }
    }
    
    private fun parseImportResult(result: PyObject): ImportResult {
        val resultMap = result.asMap()
        val conversations = resultMap["imported_conversations"]?.asList()?.map { conv ->
            val convMap = conv.asMap()
            ConversationSummary(
                filename = convMap["filename"].toString(),
                sessionId = convMap["session_id"].toString(),
                title = convMap["title"].toString(),
                summary = convMap["summary"].toString()
            )
        } ?: emptyList()
        
        return ImportResult(
            totalFiles = resultMap["total_files"]?.toInt() ?: 0,
            successfulFiles = resultMap["successful_files"]?.toInt() ?: 0,
            failedFiles = resultMap["failed_files"]?.toInt() ?: 0,
            totalConversations = resultMap["total_conversations"]?.toInt() ?: 0,
            importedConversations = conversations
        )
    }
}
