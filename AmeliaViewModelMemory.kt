package com.antonio.my.ai.girlfriend.free.amelia.ai.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.amelia.memory.MainActivityRepository
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ViewModel for Amelia AI with integrated memory system
 */
class AmeliaViewModel(
    private val repository: MainActivityRepository
) : ViewModel() {
    
    // Python objects
    private val python: Python by lazy { Python.getInstance() }
    private val integrationModule: PyObject by lazy { python.getModule("amelia_memory_integration") }
    private var memoryIntegration: PyObject? = null
    
    // UI State
    private val _messages = MutableLiveData<List<ChatMessage>>()
    val messages: LiveData<List<ChatMessage>> = _messages
    
    private val _isLoading = MutableLiveData(false)
    val isLoading: LiveData<Boolean> = _isLoading
    
    private val _memoryStatus = MutableLiveData<MemoryStatus>()
    val memoryStatus: LiveData<MemoryStatus> = _memoryStatus
    
    private val _currentSessionId = MutableLiveData<String?>()
    val currentSessionId: LiveData<String?> = _currentSessionId
    
    // Data classes
    data class ChatMessage(
        val id: String,
        val role: String, // "user" or "amelia"
        val content: String,
        val timestamp: Long = System.currentTimeMillis(),
        val hasMemoryContext: Boolean = false
    )
    
    data class MemoryStatus(
        val isInitialized: Boolean,
        val totalConversations: Int,
        val currentSessionActive: Boolean,
        val lastError: String? = null
    )
    
    init {
        initializeMemorySystem()
    }
    
    /**
     * Initialize the memory system and integration
     */
    private fun initializeMemorySystem() {
        viewModelScope.launch {
            _isLoading.value = true
            
            val initialized = repository.initializeMemory()
            if (initialized) {
                // Create memory integration
                withContext(Dispatchers.IO) {
                    try {
                        val memory = repository.getMemoryModule() // You'll need to add this getter
                        memoryIntegration = integrationModule.callAttr("create_integration", memory)
                        
                        // Start a new conversation session
                        val sessionId = repository.startConversation()
                        _currentSessionId.postValue(sessionId)
                        
                        updateMemoryStatus()
                    } catch (e: Exception) {
                        _memoryStatus.postValue(
                            MemoryStatus(false, 0, false, e.message)
                        )
                    }
                }
            }
            
            _isLoading.value = false
        }
    }
    
    /**
     * Send a message and get Amelia's response with memory integration
     */
    fun sendMessage(userMessage: String) {
        if (userMessage.isBlank()) return
        
        viewModelScope.launch {
            _isLoading.value = true
            
            // Add user message to UI
            val userMsg = ChatMessage(
                id = generateMessageId(),
                role = "user",
                content = userMessage
            )
            addMessageToUI(userMsg)
            
            // Get Amelia's base response (your existing AI logic)
            val ameliaBaseResponse = getAmeliaBaseResponse(userMessage)
            
            // Enhance with memory
            val enhancedResponse = withContext(Dispatchers.IO) {
                try {
                    memoryIntegration?.let { integration ->
                        integrationModule.callAttr(
                            "process_with_memory",
                            integration,
                            userMessage,
                            ameliaBaseResponse,
                            true // save to memory
                        ).toString()
                    } ?: ameliaBaseResponse
                } catch (e: Exception) {
                    // Fallback to base response if memory fails
                    ameliaBaseResponse
                }
            }
            
            // Check if memory was used
            val hasMemoryContext = enhancedResponse != ameliaBaseResponse
            
            // Add Amelia's response to UI
            val ameliaMsg = ChatMessage(
                id = generateMessageId(),
                role = "amelia",
                content = enhancedResponse,
                hasMemoryContext = hasMemoryContext
            )
            addMessageToUI(ameliaMsg)
            
            _isLoading.value = false
            updateMemoryStatus()
        }
    }
    
    /**
     * Load conversation context from memory
     */
    fun loadConversationContext() {
        viewModelScope.launch {
            val context = repository.getCurrentContext()
            val messages = context.map { msg ->
                ChatMessage(
                    id = generateMessageId(),
                    role = msg.role,
                    content = msg.content,
                    hasMemoryContext = true
                )
            }
            _messages.value = messages
        }
    }
    
    /**
     * Search memories for a specific query
     */
    fun searchMemories(query: String) {
        viewModelScope.launch {
            _isLoading.value = true
            
            val results = repository.searchMemories(query)
            
            // Display search results as a special message
            if (results.isNotEmpty()) {
                val searchSummary = buildSearchResultsSummary(results)
                val searchMsg = ChatMessage(
                    id = generateMessageId(),
                    role = "system",
                    content = searchSummary,
                    hasMemoryContext = true
                )
                addMessageToUI(searchMsg)
            }
            
            _isLoading.value = false
        }
    }
    
    /**
     * End current conversation with summary
     */
    fun endConversation(summary: String? = null) {
        viewModelScope.launch {
            // Auto-generate summary if not provided
            val conversationSummary = summary ?: generateConversationSummary()
            
            // Extract topics from conversation
            val topics = extractTopicsFromConversation()
            
            repository.endConversation(conversationSummary, topics)
            _currentSessionId.value = null
            updateMemoryStatus()
        }
    }
    
    /**
     * Start a new conversation
     */
    fun startNewConversation() {
        viewModelScope.launch {
            val sessionId = repository.startConversation()
            _currentSessionId.value = sessionId
            _messages.value = emptyList()
            updateMemoryStatus()
        }
    }
    
    /**
     * Import conversations from clipboard
     */
    fun importFromClipboard(clipboardText: String) {
        viewModelScope.launch {
            _isLoading.value = true
            
            val result = repository.pasteSingleConversation(clipboardText)
            
            val importMsg = if (result.success) {
                "✓ Imported: ${result.title}\n" +
                "Summary: ${result.summary}\n" +
                "Messages: ${result.messageCount}"
            } else {
                "✗ Import failed: ${result.error}"
            }
            
            val systemMsg = ChatMessage(
                id = generateMessageId(),
                role = "system",
                content = importMsg
            )
            addMessageToUI(systemMsg)
            
            _isLoading.value = false
            updateMemoryStatus()
        }
    }
    
    // Helper functions
    
    private fun getAmeliaBaseResponse(userMessage: String): String {
        // This is where you'd call your existing Amelia AI logic
        // For now, returning a placeholder
        return "This is Amelia's base response to: $userMessage"
    }
    
    private fun addMessageToUI(message: ChatMessage) {
        val current = _messages.value ?: emptyList()
        _messages.value = current + message
    }
    
    private fun generateMessageId(): String {
        return "msg_${System.currentTimeMillis()}_${(0..999).random()}"
    }
    
    private fun updateMemoryStatus() {
        viewModelScope.launch {
            repository.getMemoryStats()?.let { stats ->
                _memoryStatus.value = MemoryStatus(
                    isInitialized = true,
                    totalConversations = stats.totalConversations,
                    currentSessionActive = stats.hasActiveSession
                )
            }
        }
    }
    
    private fun buildSearchResultsSummary(results: List<MainActivityRepository.SearchResult>): String {
        val sb = StringBuilder("🔍 Found ${results.size} relevant conversations:\n\n")
        
        results.take(3).forEach { result ->
            sb.append("📅 ${result.startedAt.substringBefore('T')}\n")
            result.summary?.let { sb.append("📝 $it\n") }
            if (result.matchingMessages.isNotEmpty()) {
                sb.append("💬 Sample: \"${result.matchingMessages.first().content.take(100)}...\"\n")
            }
            sb.append("\n")
        }
        
        return sb.toString()
    }
    
    private suspend fun generateConversationSummary(): String {
        // Simple summary based on message count and content
        val messages = _messages.value ?: return "Brief conversation"
        
        val userMessageCount = messages.count { it.role == "user" }
        val topics = extractTopicsFromConversation()
        
        return "Discussion with $userMessageCount exchanges" +
               if (topics.isNotEmpty()) " covering: ${topics.joinToString(", ")}" else ""
    }
    
    private fun extractTopicsFromConversation(): List<String> {
        val messages = _messages.value ?: return emptyList()
        val allText = messages.joinToString(" ") { it.content }
        
        // Simple topic extraction - look for capitalized words
        val topics = mutableSetOf<String>()
        val capitalizedWords = Regex("[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*").findAll(allText)
        
        capitalizedWords.forEach { match ->
            val word = match.value
            if (word.length > 4 && !word.startsWith("The") && !word.startsWith("This")) {
                topics.add(word.toLowerCase())
            }
        }
        
        return topics.take(5).toList()
    }
}
