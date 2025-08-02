package chat.simplex.bridge

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class ClaudeBridgeReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        val message = intent?.getStringExtra("claude_reply") ?: return
        ClaudeBridge.handleClaudeReply(message)
    }
}
