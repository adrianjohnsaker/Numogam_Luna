package com.amelia.ai

import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.amelia.ai.adapter.ChatMessageAdapter
import com.amelia.ai.databinding.ActivityChatBinding
import com.amelia.ai.viewmodel.AmeliaViewModel
import com.amelia.memory.MainActivityRepository
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class ChatActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityChatBinding
    private lateinit var viewModel: AmeliaViewModel
    private lateinit var messageAdapter: ChatMessageAdapter
    private lateinit var bottomSheetBehavior: BottomSheetBehavior<View>
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChatBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Initialize repository and ViewModel
        val repository = MainActivityRepository(this)
        val viewModelFactory = AmeliaViewModelFactory(repository)
        viewModel = ViewModelProvider(this, viewModelFactory)[AmeliaViewModel::class.java]
        
        setupUI()
        observeViewModel()
    }
    
    private fun setupUI() {
        // Setup toolbar
        setSupportActionBar(binding.toolbar)
        
        // Setup RecyclerView
        messageAdapter = ChatMessageAdapter()
        binding.messagesRecyclerView.apply {
            adapter = messageAdapter
            layoutManager = LinearLayoutManager(this@ChatActivity)
        }
        
        // Setup bottom sheet
        bottomSheetBehavior = BottomSheetBehavior.from(binding.memoryBottomSheet.root)
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_HIDDEN
        
        // Setup click listeners
        setupClickListeners()
        
        // Setup input
        setupMessageInput()
    }
    
    private fun setupClickListeners() {
        // Send button
        binding.sendButton.setOnClickListener {
            sendMessage()
        }
        
        // Memory actions button
        binding.memoryActionsButton.setOnClickListener {
            toggleBottomSheet()
        }
        
        // Search button in search card
        binding.searchButton.setOnClickListener {
            performMemorySearch()
        }
        
        // Bottom sheet actions
        binding.memoryBottomSheet.actionSearchMemories.setOnClickListener {
            showSearchCard()
            bottomSheetBehavior.state = BottomSheetBehavior.STATE_HIDDEN
        }
        
        binding.memoryBottomSheet.actionImportClipboard.setOnClickListener {
            importFromClipboard()
            bottomSheetBehavior.state = BottomSheetBehavior.STATE_HIDDEN
        }
        
        binding.memoryBottomSheet.actionEndConversation.setOnClickListener {
            showEndConversationDialog()
        }
        
        binding.memoryBottomSheet.actionViewStats.setOnClickListener {
            showMemoryStats()
        }
    }
    
    private fun setupMessageInput() {
        binding.messageInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                sendMessage()
                true
            } else {
                false
            }
        }
    }
    
    private fun observeViewModel() {
        // Observe messages
        viewModel.messages.observe(this) { messages ->
            messageAdapter.submitList(messages)
            if (messages.isNotEmpty()) {
                binding.messagesRecyclerView.smoothScrollToPosition(messages.size - 1)
            }
        }
        
        // Observe loading state
        viewModel.isLoading.observe(this) { isLoading ->
            binding.loadingIndicator.visibility = if (isLoading) View.VISIBLE else View.GONE
            binding.sendButton.isEnabled = !isLoading
        }
        
        // Observe memory status
        viewModel.memoryStatus.observe(this) { status ->
            updateMemoryStatusUI(status)
        }
        
        // Observe current session
        viewModel.currentSessionId.observe(this) { sessionId ->
            // Update UI based on session state
            binding.memoryBottomSheet.actionEndConversation.isEnabled = sessionId != null
        }
    }
    
    private fun sendMessage() {
        val message = binding.messageInput.text?.toString()?.trim()
        if (!message.isNullOrEmpty()) {
            viewModel.sendMessage(message)
            binding.messageInput.text?.clear()
        }
    }
    
    private fun toggleBottomSheet() {
        if (bottomSheetBehavior.state == BottomSheetBehavior.STATE_HIDDEN) {
            bottomSheetBehavior.state = BottomSheetBehavior.STATE_EXPANDED
        } else {
            bottomSheetBehavior.state = BottomSheetBehavior.STATE_HIDDEN
        }
    }
    
    private fun showSearchCard() {
        binding.searchCard.visibility = View.VISIBLE
        binding.searchInput.requestFocus()
    }
    
    private fun performMemorySearch() {
        val query = binding.searchInput.text?.toString()?.trim()
        if (!query.isNullOrEmpty()) {
            viewModel.searchMemories(query)
            binding.searchCard.visibility = View.GONE
            binding.searchInput.text?.clear()
        }
    }
    
    private fun importFromClipboard() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clipData = clipboard.primaryClip
        
        if (clipData != null && clipData.itemCount > 0) {
            val text = clipData.getItemAt(0).text?.toString()
            if (!text.isNullOrEmpty()) {
                viewModel.importFromClipboard(text)
            } else {
                Toast.makeText(this, "Clipboard is empty", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(this, "No text in clipboard", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun showEndConversationDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("End Conversation")
            .setMessage("Would you like to save a summary of this conversation?")
            .setPositiveButton("Save with Summary") { _, _ ->
                // Let the ViewModel auto-generate summary
                viewModel.endConversation()
                bottomSheetBehavior.state = BottomSheetBehavior.STATE_HIDDEN
                Toast.makeText(this, "Conversation saved to memory", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Save without Summary") { _, _ ->
                viewModel.endConversation("")
                bottomSheetBehavior.state = BottomSheetBehavior.STATE_HIDDEN
            }
            .setNeutralButton("Cancel", null)
            .show()
    }
    
    private fun showMemoryStats() {
        // This could open a detailed stats dialog or activity
        // For now, just showing a toast with basic info
        viewModel.memoryStatus.value?.let { status ->
            val message = """
                Memory Statistics:
                ${status.totalConversations} conversations stored
                Session active: ${status.currentSessionActive}
            """.trimIndent()
            
            MaterialAlertDialogBuilder(this)
                .setTitle("Memory Statistics")
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show()
        }
    }
    
    private fun updateMemoryStatusUI(status: AmeliaViewModel.MemoryStatus) {
        // Update toolbar memory indicator
        binding.memoryStatusText.text = "${status.totalConversations} memories"
        
        // Update bottom sheet stats
        binding.memoryBottomSheet.apply {
            statsConversations.text = "${status.totalConversations}\nConversations"
            // You'd need to add more stats from the repository
        }
        
        // Update memory icon color based on initialization
        val iconTint = if (status.isInitialized) {
            getColor(R.color.memory_active)
        } else {
            getColor(R.color.memory_inactive)
        }
        binding.memoryStatusIcon.setColorFilter(iconTint)
    }
    
    // ViewModel Factory
    class AmeliaViewModelFactory(
        private val repository: MainActivityRepository
    ) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(AmeliaViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return AmeliaViewModel(repository) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}
