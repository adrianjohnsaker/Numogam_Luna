package com.example.theoryfiction.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.widget.CheckBox
import android.widget.ProgressBar
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.theoryfiction.R
import com.example.theoryfiction.bridge.LanguageHyperformationBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * LanguageHyperformationFragment - A fragment for accessing language hyperformation capabilities
 * 
 * This fragment provides a simplified interface to the language hyperformation modules:
 * - Transglyphic Linguistic Kernel
 * - Poetic Syntax Rewriter
 * - Mood-Tuned Language Modulator
 */
class LanguageHyperformationFragment : Fragment() {
    private lateinit var bridge: LanguageHyperformationBridge
    
    // UI Elements
    private lateinit var inputText: EditText
    private lateinit var outputText: TextView
    private lateinit var toneSpinner: Spinner
    private lateinit var actionButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var metamoodCheckbox: CheckBox
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_language_hyperformation, container, false)
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        // Initialize bridge
        bridge = LanguageHyperformationBridge(requireContext())
        
        // Initialize UI elements
        inputText = view.findViewById(R.id.editTextInput)
        outputText = view.findViewById(R.id.textViewOutput)
        toneSpinner = view.findViewById(R.id.spinnerTone)
        actionButton = view.findViewById(R.id.buttonProcess)
        progressBar = view.findViewById(R.id.progressBar)
        metamoodCheckbox = view.findViewById(R.id.checkboxMetamood)
        
        // Set up tone spinner
        ArrayAdapter.createFromResource(
            requireContext(),
            R.array.tones_array,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            toneSpinner.adapter = adapter
        }
        
        // Set up action button
        actionButton.setOnClickListener {
            processText()
        }
        
        // Initialize bridge in background
        initializeBridge()
    }
    
    private fun initializeBridge() {
        lifecycleScope.launch {
            progressBar.visibility = View.VISIBLE
            actionButton.isEnabled = false
            
            val initialized = withContext(Dispatchers.IO) {
                try {
                    bridge.initialize()
                } catch (e: Exception) {
                    false
                }
            }
            
            if (initialized) {
                outputText.text = "Language hyperformation modules initialized successfully."
            } else {
                outputText.text = "Failed to initialize language modules. Please try again."
            }
            
            progressBar.visibility = View.GONE
            actionButton.isEnabled = true
        }
    }
    
    private fun processText() {
        val text = inputText.text.toString()
        if (text.isBlank()) {
            outputText.text = "Please enter some text to process."
            return
        }
        
        val tone = toneSpinner.selectedItem.toString()
        val useMetamood = metamoodCheckbox.isChecked
        
        actionButton.isEnabled = false
        progressBar.visibility = View.VISIBLE
        outputText.text = "Processing..."
        
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                try {
                    // Determine which processing to do based on tone
                    when {
                        tone.startsWith("poetic") || tone.startsWith("recursive") -> {
                            // Use poetic syntax rewriter
                            val technique = if (tone.startsWith("poetic")) "mirroring" else "recursion"
                            bridge.rewritePhrase(
                                phrase = text,
                                syntaxType = "fluid",
                                technique = technique
                            ).fold(
                                onSuccess = { it.transformed },
                                onFailure = { "Error: ${it.message}" }
                            )
                        }
                        tone.startsWith("mood") -> {
                            // Use mood tuned modulator
                            val mood = tone.substringAfter(":").trim()
                            bridge.modulateText(
                                text = text,
                                mood = mood,
                                metamoodFactor = if (useMetamood) 0.5f else null
                            ).fold(
                                onSuccess = { it.modulated },
                                onFailure = { "Error: ${it.message}" }
                            )
                        }
                        else -> {
                            // Use transglyphic kernel by default
                            val glyphs = text.split(" ").take(3)
                            bridge.generateTransglyphicPhrase(
                                glyphs = glyphs,
                                tone = "poetic"
                            ).fold(
                                onSuccess = { it.transformedPhrase },
                                onFailure = { "Error: ${it.message}" }
                            )
                        }
                    }
                } catch (e: Exception) {
                    "Error: ${e.message}"
                }
            }
            
            outputText.text = result
            actionButton.isEnabled = true
            progressBar.visibility = View.GONE
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clean up resources
        bridge.cleanup()
    }
    
    companion object {
        /**
         * Use this factory method to create a new instance of this fragment.
         */
        @JvmStatic
        fun newInstance() = LanguageHyperformationFragment()
    }
}
