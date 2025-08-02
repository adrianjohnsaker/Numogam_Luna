package chat.simplex.bridge

import chat.simplex.common.views.chat.ComposeMessage
import chat.simplex.common.views.chat.OnMessageCallback
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch

object ClaudeBridge {
    var sendMessageCallback: OnMessageCallback? = null

    fun handleClaudeReply(text: String) {
        val message = ComposeMessage(text)
        MainScope().launch {
            sendMessageCallback?.onMessage(message)
        }
    }
}
