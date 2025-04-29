package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PythonHookActivity : AppCompatActivity() {
    private val TAG = "PythonHookActivity"
    private lateinit var moduleInterceptor: AmeliaModuleInterceptor
    
    private lateinit var inputEditText: EditText
    private lateinit var resultTextView: TextView
    private lateinit var executeButton: Button
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_python_hook)
        
        // Initialize views
        inputEditText = findViewById(R.id.inputEditText)
        resultTextView = findViewById(R.id.resultTextView)
        executeButton = findViewById(R.id.executeButton)
        
        // Initialize Python module interceptor
        moduleInterceptor = AmeliaModuleInterceptor.getInstance(this)
        
        // Set up execute button
        executeButton.setOnClickListener {
            val query = inputEditText.text.toString()
            if (query.isNotEmpty()) {
                processQuery(query)
            }
        }
        
        // Demonstrate Python module initialization
        testPythonIntegration()
    }
    
    private fun processQuery(query: String) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = moduleInterceptor.interceptQuery(query)
                
                withContext(Dispatchers.Main) {
                    if (result != null) {
                        val formattedResult = """
                            Module: ${result.moduleName}
                            Success: ${result.success}
                            Data: ${result.data}
                            ${if (result.error != null) "Error: ${result.error}" else ""}
                            Metadata: ${result.metadata}
                        """.trimIndent()
                        
                        resultTextView.text = formattedResult
                    } else {
                        resultTextView.text = "No module matched your query.\nTry one of these formats:\n" +
                                "- access Zone 1 user 'user123'\n" +
                                "- process input 'echo: hello world'\n" +
                                "- combine archetypes 'hero' 'shadow'\n" +
                                "- drift: your poetic input here\n" +
                                "- temporal query memories from 'yesterday'"
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing query: ${e.message}")
                withContext(Dispatchers.Main) {
                    resultTextView.text = "Error: ${e.message}"
                }
            }
        }
    }
    
    private fun testPythonIntegration() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val bridge = ChaquopyBridge.getInstance(this@PythonHookActivity)
                val testModule = bridge.getModule("test_module")
                val result = testModule?.callAttr("test_function", "Hello from Android!")
                
                withContext(Dispatchers.Main) {
                    Log.d(TAG, "Python test result: $result")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Python test failed: ${e.message}")
            }
        }
    }
}
