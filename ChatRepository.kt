// app/src/main/java/com/amelia/chat/ChatRepository.kt
package com.amelia.chat

import com.amelia.bridge.AmeliaPythonBridge
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ChatRepository(
    private val io: CoroutineDispatcher = Dispatchers.IO
) {
    suspend fun ask(text: String): String = withContext(io) {
        AmeliaPythonBridge.processText(text)                    // no headers
    }
    suspend fun askWithHeaders(text: String, headers: Map<String, String>): String = withContext(io) {
        AmeliaPythonBridge.process(text, headers)               // optional runtime controls
    }
}
