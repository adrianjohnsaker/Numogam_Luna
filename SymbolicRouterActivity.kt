package com.antonio.my.ai.girlfriend.free

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.antonio.my.ai.girlfriend.free.pythonbridge.PythonBridgeService
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Activity for interacting with the Symbolic Router Python module
 */
class SymbolicRouterActivity : AppCompatActivity() {
    private val TAG = "SymbolicRouterActivity"
    
    private lateinit var inputEditText: EditText
    private lateinit var processButton: Button
    private lateinit var resultTextView: TextView
    
    private var pythonBridgeService: PythonBridgeService? = null
    private var isBound = false
    
    // Service connection object
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, service: IBinder) {
            val binder = service as PythonBridgeService.LocalBinder
            pythonBridgeService = binder.getService()
            isBound = true
            
            // Check router status once connected
            checkRouterStatus()
        }
        
        override fun onServiceDisconnected(className: ComponentName) {
            pythonBridgeService = null
            isBound = false
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_symbolic_router)
        
        // Initialize UI elements
        inputEditText = findViewById(R.id.inputEditText)
        processButton = findViewById(R.id.processButton)
        resultTextView = findViewById(R.id.resultTextView)
        
        // Set up button click listener
        processButton.setOnClickListener {
            val query = inputEditText.text.toString().trim()
            if (query.isNotEmpty()) {
                processQuery(query)
            } else {
                Toast.makeText(this, "Please enter a query", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Bind to Python bridge service
        Intent(this, PythonBridgeService::class.java).also { intent ->
            bindService(intent, connection, Context.BIND_AUTO_CREATE)
        }
    }
    
    private fun checkRouterStatus() {
        if (!isBound) return
        
        lifecycleScope.launch {
            try {
                val status = pythonBridgeService?.getRouterStatus()
                Log.d(TAG, "Router status: $status")
                
                status?.let {
                    if (it.optString("status") == "ready") {
                        val routes = it.optJSONArray("routes")
                        val hint = "Available routes: " + (0 until routes.length())
                            .map { i -> routes.getString(i) }
                            .joinToString(", ")
                        
                        runOnUiThread {
                            inputEditText.hint = hint
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error checking router status", e)
                showError("Failed to check router status: ${e.message}")
            }
        }
    }
    
    private fun processQuery(query: String) {
        if (!isBound) {
            showError("Service not bound")
            return
        }
        
        // Show loading state
        runOnUiThread {
            resultTextView.text = "Processing..."
            processButton.isEnabled = false
        }
        
        lifecycleScope.launch {
            try {
                val result = pythonBridgeService?.processQuery(query)
                Log.d(TAG, "Query result: $result")
                
                runOnUiThread {
                    result?.let {
                        displayResult(it)
                    } ?: showError("No result returned")
                    processButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing query", e)
                showError("Error: ${e.message}")
                runOnUiThread {
                    processButton.isEnabled = true
                }
            }
        }
    }
    
    private fun displayResult(result: JSONObject) {
        val status = result.optString("status")
        
        when (status) {
            "success" -> {
                // Handle successful result
                val formattedResult = formatSuccessResult(result)
                resultTextView.text = formattedResult
            }
            "error" -> {
                // Handle error result
                val errorMessage = result.optString("error_message", "Unknown error")
                showError(errorMessage)
            }
            "unhandled" -> {
                // Handle unhandled query
                val message = result.optString("message", "Query not handled")
                resultTextView.text = "Unhandled: $message\n\nTry using one of the available routes."
            }
            else -> {
                // Unknown status
                resultTextView.text = "Unknown status: $status\n\n$result"
            }
        }
    }
    
    private fun formatSuccessResult(result: JSONObject): String {
        val sb = StringBuilder()
        
        // For poetic drift results
        if (result.has("glyph")) {
            sb.append("Glyph: ${result.getString("glyph")}\n\n")
            
            if (result.has("interpretation")) {
                val interpretation = result.getJSONObject("interpretation")
                sb.append("Essence: ${interpretation.optString("essence", "unknown")}\n")
                
                val patterns = interpretation.optJSONArray("patterns")
                if (patterns != null && patterns.length() > 0) {
                    sb.append("Patterns: ")
                    for (i in 0 until patterns.length()) {
                        if (i > 0) sb.append(", ")
                        sb.append(patterns.getString(i))
                    }
                }
                
                val resonance = interpretation.optDouble("resonance", 0.0)
                sb.append("\nResonance: ${(resonance * 100).toInt()}%")
            }
        } else {
            // Generic formatting for other result types
            result.keys().forEach { key ->
                if (key != "status") {
                    val value = result.get(key)
                    sb.append("$key: $value\n")
                }
            }
        }
        
        return sb.toString()
    }
    
    private fun showError(message: String) {
        Log.e(TAG, "Error: $message")
        runOnUiThread {
            resultTextView.text = "Error: $message"
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Unbind from service
        if (isBound) {
            unbindService(connection)
            isBound = false
        }
    }
}
