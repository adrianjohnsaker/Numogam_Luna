package com.antonio.my.ai.girlfriend.free.ui

import android.os.Bundle
import android.view.inputmethod.EditorInfo
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.antonio.my.ai.girlfriend.free.bridge.PipelineBridge
import com.antonio.my.ai.girlfriend.free.databinding.ActivityPipelineBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*

/**
 * MainActivityPipeline
 * ---------------------
 * A dedicated UI to exercise pipeline.py.
 * Users can type input, see Amelia’s output, and scroll through a history list.
 */
class MainActivityPipeline : AppCompatActivity() {

    private lateinit var binding: ActivityPipelineBinding
    private lateinit var bridge: PipelineBridge
    private lateinit var adapter: PipelineHistoryAdapter
    private val history = mutableListOf<PipelineRecord>()
    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPipelineBinding.inflate(layoutInflater)
        setContentView(binding.root)

        bridge = PipelineBridge.getInstance(this)

        adapter = PipelineHistoryAdapter(history)
        binding.recyclerViewHistory.layoutManager = LinearLayoutManager(this)
        binding.recyclerViewHistory.adapter = adapter

        binding.inputField.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                sendToPipeline()
                true
            } else false
        }

        binding.sendButton.setOnClickListener { sendToPipeline() }
    }

    private fun sendToPipeline() {
        val text = binding.inputField.text.toString().trim()
        if (text.isEmpty()) {
            Toast.makeText(this, getString(R.string.pipeline_empty_input), Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            binding.progressBar.show()
            val result = withContext(Dispatchers.IO) {
                bridge.processText(text)
            }
            binding.progressBar.hide()
            if (result != null) {
                displayResult(text, result)
            } else {
                Toast.makeText(this@MainActivityPipeline, getString(R.string.pipeline_error), Toast.LENGTH_LONG).show()
            }
            binding.inputField.text?.clear()
        }
    }

    private fun displayResult(input: String, result: JSONObject) {
        val content = result.optJSONArray("choices")
            ?.optJSONObject(0)
            ?.optJSONObject("message")
            ?.optString("content", "") ?: result.toString()

        val record = PipelineRecord(
            timestamp = sdf.format(Date()),
            input = input,
            output = content
        )
        history.add(0, record)
        adapter.notifyItemInserted(0)
        binding.recyclerViewHistory.scrollToPosition(0)
    }
}

// Data class for records
data class PipelineRecord(
    val timestamp: String,
    val input: String,
    val output: String
)
