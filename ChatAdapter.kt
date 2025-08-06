// File: app/src/main/java/com/amelia/consciousness/ChatAdapter.kt
package com.amelia.consciousness

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class ChatAdapter : RecyclerView.Adapter<ChatAdapter.MessageViewHolder>() {

  private val messages = mutableListOf<Message>()

  fun addMessage(text: String, isUser: Boolean) {
    messages += Message(text, isUser)
    notifyItemInserted(messages.size - 1)
  }

  override fun getItemViewType(position: Int): Int {
    return if (messages[position].isUser) VIEW_TYPE_USER else VIEW_TYPE_BOT
  }

  override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
    val layoutId = when (viewType) {
      VIEW_TYPE_USER -> R.layout.item_message_user
      else           -> R.layout.item_message_bot
    }
    val view = LayoutInflater.from(parent.context)
      .inflate(layoutId, parent, false)
    return MessageViewHolder(view)
  }

  override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
    holder.bind(messages[position].text)
  }

  override fun getItemCount(): Int = messages.size

  inner class MessageViewHolder(view: View) : RecyclerView.ViewHolder(view) {
    private val messageText: TextView = view.findViewById(R.id.messageText)
    fun bind(text: String) {
      messageText.text = text
    }
  }

  companion object {
    private const val VIEW_TYPE_USER = 1
    private const val VIEW_TYPE_BOT  = 2
  }
}
