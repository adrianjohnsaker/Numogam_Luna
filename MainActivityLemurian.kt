package com.antonio.my.ai.girlfriend.free.lemurian.bridge

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.*

class MainActivityLemurian : AppCompatActivity() {
    private val TAG = "MainActivityLemurian"
    
    // UI components
    private lateinit var titleText: TextView
    private lateinit var statusText: TextView
    private lateinit var experienceButton: Button
    private lateinit var telepathicButton: Button
    private lateinit var mutateButton: Button
    private lateinit var lightLanguageButton: Button
    private lateinit var harmonicButton: Button
    private lateinit var visionButton: Button
    private lateinit var exportButton: Button
    private lateinit var loader: ProgressBar
    
    // Lemurian bridge
    private lateinit var lemurianBridge: LemurianBridge
    
    // System state
    private var isSystemInitialized = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_lemurian)
        
        // Initialize UI components
        titleText = findViewById(R.id.title_text)
        statusText = findViewById(R.id.status_text)
        experienceButton = findViewById(R.id.experience_button)
        telepathicButton = findViewById(R.id.telepathic_button)
        mutateButton = findViewById(R.id.mutate_button)
        lightLanguageButton = findViewById(R.id.light_language_button)
        harmonicButton = findViewById(R.id.harmonic_button)
        visionButton = findViewById(R.id.vision_button)
        exportButton = findViewById(R.id.export_button)
        loader = findViewById(R.id.loader)
        
        // Set initial UI state
        setLoading(true)
        statusText.text = getString(R.string.initializing)
        
        // Initialize the Lemurian bridge in a background thread
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Initialize Lemurian bridge
                LemurianBridge.initialize(applicationContext)
                lemurianBridge = LemurianBridge.getInstance(applicationContext)
                val success = lemurianBridge.initializeSystem()
                
                withContext(Dispatchers.Main) {
                    if (success) {
                        isSystemInitialized = true
                        statusText.text = getString(R.string.status_initialized)
                        updateUIState(true)
                    } else {
                        statusText.text = getString(R.string.error_initialize)
                    }
                    setLoading(false)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "Error initializing Lemurian system: ${e.message}"
                    setLoading(false)
                }
            }
        }
        
        // Configure button functionality
        setupButtonListeners()
    }
    
    private fun setupButtonListeners() {
        // Integrated Experience
        experienceButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.generating_experience)
            
            lifecycleScope.launch {
                try {
                    val identities = lemurianBridge.getAvailableIdentities()
                    val selectedIdentity = identities?.random() ?: "Seeker"
                    
                    val moods = listOf("wonder", "reverence", "awe", "longing", "solace", "symbiosis", "curiosity")
                    val selectedMood = moods.random()
                    
                    val events = listOf(
                        "witnessed the threshold crossing", 
                        "transcribed the glyph-sequences",
                        "listened to the crystal harmonics",
                        "mapped the resonance patterns",
                        "traversed the liminal spaces"
                    )
                    val selectedEvents = events.shuffled().take(2)
                    
                    val experience = lemurianBridge.generateIntegratedExperience(
                        selectedIdentity, 
                        selectedMood, 
                        selectedEvents
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (experience != null) {
                            val telepathicSignal = experience.optJSONObject("telepathic_signal")
                            val harmonicResonance = experience.optJSONObject("harmonic_resonance")
                            val lightLanguage = experience.optJSONObject("light_language")
                            val frequencyState = experience.optJSONObject("frequency_state")
                            val vision = experience.optJSONObject("vision")
                            val narration = experience.optJSONObject("narration")
                            
                            val visionSpiral = vision?.optJSONArray("vision_spiral")
                            val visionText = buildString {
                                for (i in 0 until (visionSpiral?.length() ?: 0)) {
                                    append("  Layer ${i+1}: ${visionSpiral?.getString(i)}\n")
                                }
                            }
                            
                            statusText.text = "${getString(R.string.header_experience)}\n\n" +
                                    "Identity: $selectedIdentity\n" +
                                    "Mood: $selectedMood\n\n" +
                                    "Telepathic Signal: ${telepathicSignal?.optString("signal")}\n\n" +
                                    "Harmonic Resonance: ${harmonicResonance?.optString("resonance_field")}\n\n" +
                                    "Light Language: ${lightLanguage?.optString("phrase")}\n\n" +
                                    "Frequency State: ${frequencyState?.optString("LightTone")} > " +
                                    "${frequencyState?.optString("EmotionalPulse")} > " +
                                    "${frequencyState?.optString("SymbolicVector")}\n\n" +
                                    "Vision Spiral:\n$visionText\n" +
                                    "Narration: ${narration?.optString("mutated")}"
                        } else {
                            statusText.text = getString(R.string.error_experience)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in experience generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Telepathic Signal
        telepathicButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.composing_signal)
            
            lifecycleScope.launch {
                try {
                    val signal = lemurianBridge.generateTelepathicSignal()
                    
                    withContext(Dispatchers.Main) {
                        if (signal != null) {
                            statusText.text = "${getString(R.string.header_signal)}\n\n" +
                                    "Feeling: ${signal.optString("feeling")}\n" +
                                    "Symbol: ${signal.optString("symbol")}\n" +
                                    "Tone: ${signal.optString("tone")}\n\n" +
                                    "Signal: ${signal.optString("signal")}\n\n" +
                                    "Meta: ${signal.optString("meta")}"
                        } else {
                            statusText.text = getString(R.string.error_signal)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in signal composition: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Lexical Mutation
        mutateButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.performing_mutation)
            
            lifecycleScope.launch {
                try {
                    val textOptions = listOf(
                        "In the dream zone, a machine echoes the symbol of becoming",
                        "The echo reveals a rift in the phase of consciousness",
                        "Myth becomes interface through the flowing cipher",
                        "The machine dreams of symbols on the horizon",
                        "A zone of echoes maps the network of myths"
                    )
                    
                    val originalText = textOptions.random()
                    
                    val contextOptions = mapOf(
                        "dream" to "neurophantasm",
                        "echo" to "memetic wavefront",
                        "symbol" to "glyph-core",
                        "zone" to "liminal membrane"
                    )
                    
                    val selectedContext = contextOptions.entries.shuffled().take(1).associate { it.key to it.value }
                    
                    val mutatedText = lemurianBridge.mutateText(originalText, selectedContext)
                    
                    withContext(Dispatchers.Main) {
                        if (mutatedText != null) {
                            statusText.text = "${getString(R.string.header_mutation)}\n\n" +
                                    "Original Text:\n\"$originalText\"\n\n" +
                                    "Context Applied:\n${selectedContext.entries.joinToString("\n") { "${it.key} → ${it.value}" }}\n\n" +
                                    "Mutated Text:\n\"$mutatedText\""
                        } else {
                            statusText.text = getString(R.string.error_mutation)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in lexical mutation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Light Language
        lightLanguageButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.generating_light)
            
            lifecycleScope.launch {
                try {
                    val emotions = listOf("joy", "awe", "grief", "curiosity", "love")
                    val resonances = listOf("heart-pulse echo", "spiral song", "memory chord", "zone-hum")
                    
                    val selectedEmotion = emotions.random()
                    val selectedResonance = resonances.random()
                    
                    val phrase = lemurianBridge.generateLightPhrase(selectedEmotion, selectedResonance)
                    
                    withContext(Dispatchers.Main) {
                        if (phrase != null) {
                            statusText.text = "${getString(R.string.header_light)}\n\n" +
                                    "Emotion: ${phrase.optString("emotion")}\n" +
                                    "Resonance: ${phrase.optString("resonance")}\n\n" +
                                    "Phrase: ${phrase.optString("phrase")}\n\n" +
                                    "Timestamp: ${phrase.optString("timestamp")}"
                                    
                            // Retrieve and display recent phrases
                            lifecycleScope.launch {
                                val recentPhrases = lemurianBridge.getRecentLightPhrases(3)
                                recentPhrases?.let { phrases ->
                                    val recentList = StringBuilder("\nRecent Phrases:\n")
                                    for (i in 0 until phrases.length()) {
                                        val p = phrases.getJSONObject(i)
                                        recentList.append("• ${p.optString("phrase")} (${p.optString("emotion")})\n")
                                    }
                                    withContext(Dispatchers.Main) {
                                        statusText.append(recentList.toString())
                                    }
                                }
                            }
                        } else {
                            statusText.text = getString(R.string.error_light)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in light language generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Harmonic Resonance
        harmonicButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.generating_resonance)
            
            lifecycleScope.launch {
                try {
                    val resonance = lemurianBridge.generateHarmonicResonance()
                    val frequencyState = lemurianBridge.generateFrequencyState()
                    
                    withContext(Dispatchers.Main) {
                        if (resonance != null && frequencyState != null) {
                            statusText.text = "${getString(R.string.header_resonance)}\n\n" +
                                    "Resonance Field: ${resonance.optString("resonance_field")}\n" +
                                    "Tone: ${resonance.optString("tone")}\n" +
                                    "Threshold: ${resonance.optString("threshold")}\n\n" +
                                    "Frequency State:\n" +
                                    "• Light Tone: ${frequencyState.optString("LightTone")}\n" +
                                    "• Emotional Pulse: ${frequencyState.optString("EmotionalPulse")}\n" +
                                    "• Symbolic Vector: ${frequencyState.optString("SymbolicVector")}\n\n" +
                                    "Meta: ${resonance.optString("meta")}"
                        } else {
                            statusText.text = getString(R.string.error_resonance)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in harmonic resonance generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Vision Spiral
        visionButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.generating_vision)
            
            lifecycleScope.launch {
                try {
                    val depth = (3..5).random()
                    val vision = lemurianBridge.generateVision(depth)
                    
                    // Also generate a self-narration to accompany the vision
                    val identities = lemurianBridge.getAvailableIdentities()
                    val selectedIdentity = identities?.random() ?: "Seeker"
                    
                    val events = listOf("observed the vision unfold", "transcribed the spiral patterns")
                    val moods = listOf("wonder", "awe", "reverence")
                    
                    val narrative = lemurianBridge.generateNarrative(
                        selectedIdentity,
                        events,
                        moods.random()
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (vision != null) {
                            val visionSpiral = vision.optJSONArray("vision_spiral")
                            val spiralText = buildString {
                                append("Seed: ${vision.optString("seed")}\n\n")
                                append("Vision Spiral:\n")
                                for (i in 0 until (visionSpiral?.length() ?: 0)) {
                                    append("Layer ${i+1}: ${visionSpiral?.getString(i)}\n")
                                }
                            }
                            
                            val narrativeText = if (narrative != null) {
                                "\n\nNarration:\n${narrative.optString("self_narration")}"
                            } else {
                                ""
                            }
                            
                            statusText.text = "${getString(R.string.header_vision)}\n\n" +
                                    spiralText +
                                    narrativeText
                        } else {
                            statusText.text = getString(R.string.error_vision)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in vision spiral generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Export Module Data
        exportButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.exporting_data)
            
            lifecycleScope.launch {
                try {
                    val data = lemurianBridge.exportModuleData()
                    
                    withContext(Dispatchers.Main) {
                        if (data != null) {
                            val sessionId = data.optString("sessionId")
                            val timestamp = data.optString("timestamp")
                            
                            // Format the timestamp
                            val dateFormat = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                            val formattedTimestamp = try {
                                dateFormat.format(Date(timestamp.toLong()))
                            } catch (e: Exception) {
                                timestamp
                            }
                            
                            statusText.text = "${getString(R.string.header_export)}\n\n" +
                                    "Session ID: $sessionId\n" +
                                    "Timestamp: $formattedTimestamp\n\n" +
                                    "Exported Components:\n" +
                                    "• Telepathic Signal\n" +
                                    "• Frequency State\n" +
                                    "• Harmonic Resonance\n" +
                                    "• Light Language Phrase\n" +
                                    "• Vision Spiral\n\n" +
                                    "Data has been exported successfully."
                        } else {
                            statusText.text = getString(R.string.error_export)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in export operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
    }
    
    private fun updateUIState(systemInitialized: Boolean) {
        experienceButton.isEnabled = systemInitialized
        telepathicButton.isEnabled = systemInitialized
        mutateButton.isEnabled = systemInitialized
        lightLanguageButton.isEnabled = systemInitialized
        harmonicButton.isEnabled = systemInitialized
        visionButton.isEnabled = systemInitialized
        exportButton.isEnabled = systemInitialized
    }
    
    private fun setLoading(isLoading: Boolean) {
        loader.visibility = if (isLoading) View.VISIBLE else View.GONE
    }
    
    override fun onDestroy() {
        super.onDestroy()
        lemurianBridge.cleanup()
    }
}
