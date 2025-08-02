package chat.simplex.bridge

import com.chaquo.python.Python

object ClaudeCaller {
    fun sendToClaude(userText: String) {
        val py = Python.getInstance()
        val module = py.getModule("claude_chat")
        // Call process_and_reply asynchronously if needed
        module.callAttr("process_and_reply", userText)
    }
}
