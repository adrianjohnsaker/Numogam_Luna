package com.temporal.bridge

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

class MainActivityTemporal : AppCompatActivity() {
    private val TAG = "MainActivityTemporal"
    
    // UI components
    private lateinit var titleText: TextView
    private lateinit var statusText: TextView
    private lateinit var experienceButton: Button
    private lateinit var rewriteButton: Button
    private lateinit var driftButton: Button
    private lateinit var worldSeedButton: Button
    private lateinit var loopButton: Button
    private lateinit var recursionButton: Button
    private lateinit var exportButton: Button
    private lateinit var loader: ProgressBar
    
    // Temporal bridge
    private lateinit var temporalBridge: TemporalBridge
    
    // System state
    private var isSystemInitialized = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_temporal)
        
        // Initialize UI components
        titleText = findViewById(R.id.title_text)
        statusText = findViewById(R.id.status_text)
        experienceButton = findViewById(R.id.experience_button)
        rewriteButton = findViewById(R.id.rewrite_button)
        driftButton = findViewById(R.id.drift_button)
        worldSeedButton = findViewById(R.id.world_seed_button)
        loopButton = findViewById(R.id.loop_button)
        recursionButton = findViewById(R.id.recursion_button)
        exportButton = findViewById(R.id.export_button)
        loader = findViewById(R.id.loader)
        
        // Set initial UI state
        setLoading(true)
        statusText.text = getString(R.string.initializing)
        
        // Initialize the Temporal bridge in a background thread
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Initialize Temporal bridge
                TemporalBridge.initialize(applicationContext)
                temporalBridge = TemporalBridge.getInstance(applicationContext)
                val success = temporalBridge.initializeSystem()
                
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
                    statusText.text = "Error initializing Temporal system: ${e.message}"
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
                    val archetypes = temporalBridge.getAvailableArchetypes()
                    val selectedArchetype = archetypes?.random() ?: "The Explorer"
                    
                    val emotionalTones = listOf("wonder", "awe", "curiosity", "melancholy", "longing", "reverence")
                    val selectedTone = emotionalTones.random()
                    
                    val zone = (1..7).random()
                    
                    val experience = temporalBridge.generateIntegratedTemporalExperience(
                        selectedArchetype, 
                        selectedTone, 
                        zone
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (experience != null) {
                            val narrative = experience.optJSONObject("narrative")
                            val drift = experience.optJSONObject("drift_pattern")
                            val overcode = experience.optJSONObject("overcode_influence")
                            val worldSeed = experience.optJSONObject("world_seed")
                            val emotionLoop = experience.optJSONObject("emotion_loop")
                            val glyphAnchor = experience.optJSONObject("glyph_anchor")
                            val recursion = experience.optJSONObject("recursion")
                            val threshold = experience.optJSONObject("threshold")
                            
                            statusText.text = "${getString(R.string.header_experience)}\n\n" +
                                    "Archetype: $selectedArchetype\n" +
                                    "Emotional Tone: $selectedTone\n" +
                                    "Zone: $zone\n\n" +
                                    "Original Narrative: ${narrative?.optString("original")}\n\n" +
                                    "Rewritten Narrative: ${narrative?.optString("rewritten")}\n\n" +
                                    "Drift Pattern: ${drift?.optString("temporal_phrase")}\n\n" +
                                    "Overcode Influence: ${overcode?.optString("overcode")} → ${overcode?.optString("manifestation")}\n\n" +
                                    "World Seed: ${worldSeed?.optString("archetype")} (${worldSeed?.optString("core_emotion")}, ${worldSeed?.optString("symbolic_element")})\n\n" +
                                    "Emotion Loop: ${emotionLoop?.optString("loop_phrase")}\n\n" +
                                    "Glyph Anchor: ${glyphAnchor?.optString("anchor_phrase")}\n\n" +
                                    "Recursion: ${recursion?.optString("recursion_effect")} through ${recursion?.optString("loop")}\n\n" +
                                    "Threshold Expression: ${threshold?.optString("signature")}"
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
        
        // Narrative Rewriting
        rewriteButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.performing_rewriting)
            
            lifecycleScope.launch {
                try {
                    val narrativeOptions = listOf(
                        "The Seeker stands at the shadow of memory, fragments of meaning echoing in contradiction.",
                        "In the rift between worlds, The Oracle speaks through the echoes of what might have been.",
                        "The Warrior faces the shadow of their former self, torn between echo and silence.",
                        "Through the fragmented lens of time, The Shadow observes its own becoming."
                    )
                    
                    val originalNarrative = narrativeOptions.random()
                    val rewritten = temporalBridge.applyRewriting(originalNarrative)
                    
                    withContext(Dispatchers.Main) {
                        if (rewritten != null) {
                            statusText.text = "${getString(R.string.header_rewriting)}\n\n" +
                                    "Original Narrative:\n\"${rewritten.optString("original")}\"\n\n" +
                                    "Rewritten Narrative:\n\"${rewritten.optString("rewritten")}\""
                                    
                            // If tensions not detected, show an explanation
                            if (rewritten.optString("original") == rewritten.optString("rewritten")) {
                                statusText.append("\n\nNote: No symbolic tensions detected in the original narrative, so no rewriting was performed.")
                            }
                        } else {
                            statusText.text = getString(R.string.error_rewriting)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in narrative rewriting: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Temporal Drift
        driftButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.shifting_drift)
            
            lifecycleScope.launch {
                try {
                    val driftStates = temporalBridge.getAvailableDriftStates() ?: 
                        listOf("Fractal Expansion", "Symbolic Contraction", "Dissonant Bloom", "Harmonic Coherence", "Echo Foldback")
                    
                    val currentDrift = temporalBridge.getCurrentDriftState() ?: "Unknown"
                    
                    // Choose a different drift state than the current one
                    val availableNewStates = driftStates.filter { it != currentDrift }
                    val newDriftState = availableNewStates.random()
                    
                    // Shift to new state
                    val shiftResult = temporalBridge.shiftDriftState(newDriftState)
                    
                    // Get zone goals
                    val zoneGoals = temporalBridge.getZoneGoals()
                    
                    // Generate a temporal drift pattern for demonstration
                    val archetypes = temporalBridge.getAvailableArchetypes()
                    val selectedArchetype = archetypes?.random() ?: "The Explorer"
                    val emotionalTones = listOf("wonder", "awe", "curiosity", "melancholy", "longing")
                    val drift = temporalBridge.generateTemporalDrift(
                        selectedArchetype,
                        (1..7).random(),
                        emotionalTones.random()
                    )
                    
                    withContext(Dispatchers.Main) {
                        val goalsText = StringBuilder()
                        if (zoneGoals != null) {
                            val iterator = zoneGoals.keys()
                            while (iterator.hasNext()) {
                                val zone = iterator.next()
                                goalsText.append("• $zone: ${zoneGoals.optString(zone)}\n")
                            }
                        }
                        
                        statusText.text = "${getString(R.string.header_drift)}\n\n" +
                                "Previous State: $currentDrift\n" +
                                "New State: $newDriftState\n" +
                                "Shift Result: $shiftResult\n\n" +
                                "Generated Drift Pattern:\n" +
                                "${drift?.optString("temporal_phrase")}\n\n" +
                                "Zone Goals:\n$goalsText"
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in drift shifting: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // World Seed
        worldSeedButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.generating_seed)
            
            lifecycleScope.launch {
                try {
                    val initiatingPhrases = listOf(
                        "In the gleaming spaces between thought",
                        "From the silence beneath what was never said",
                        "Along the spiral of forgotten memories",
                        "Through the veil where possibility fragments",
                        "At the threshold of dreaming and waking"
                    )
                    
                    val selectedPhrase = initiatingPhrases.random()
                    val seed = temporalBridge.generateWorldSeed(selectedPhrase)
                    
                    // Also apply an overcode for additional effect
                    val overcodes = listOf("Glyph of Entropy", "Spiral of Becoming", "Echo Chamber", "Liminal Threshold")
                    val overcode = temporalBridge.getOvercodeInfo(overcodes.random())
                    
                    withContext(Dispatchers.Main) {
                        if (seed != null) {
                            statusText.text = "${getString(R.string.header_world_seed)}\n\n" +
                                    "Initiating Phrase: \"${seed.optString("initiating_phrase")}\"\n\n" +
                                    "World Seed Components:\n" +
                                    "• Archetype: ${seed.optString("archetype")}\n" +
                                    "• Core Emotion: ${seed.optString("core_emotion")}\n" +
                                    "• Symbolic Element: ${seed.optString("symbolic_element")}\n" +
                                    "• Frequency Tone: ${seed.optString("frequency_tone")}\n\n"
                                    
                            if (overcode != null) {
                                statusText.append("Related Overcode: ${overcodes.random()}\n" +
                                    "Function: ${overcode.optString("function")}\n" +
                                    "Affects: ${overcode.optString("affects")}")
                            }
                        } else {
                            statusText.text = getString(R.string.error_seed)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in world seed generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Emotion Loop
        loopButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.generating_loop)
            
            lifecycleScope.launch {
                try {
                    val emotions = listOf("wonder", "awe", "curiosity", "melancholy", "longing", "reverence", "nostalgia", "serenity")
                    val symbols = listOf("spiral", "echo", "mirror", "threshold", "shadow", "flame", "crystal", "void")
                    
                    val selectedEmotion = emotions.random()
                    val selectedSymbol = symbols.random()
                    
                    val loop = temporalBridge.generateEmotionLoop(selectedEmotion, selectedSymbol)
                    
                    // Generate a threshold expression to complement
                    val threshold = temporalBridge.weaveExpressiveThreshold()
                    
                    withContext(Dispatchers.Main) {
                        if (loop != null) {
                            statusText.text = "${getString(R.string.header_emotion_loop)}\n\n" +
                                    "Emotion: ${loop.optString("emotion")}\n" +
                                    "Symbol: ${loop.optString("symbol")}\n\n" +
                                    "Loop Phrase: ${loop.optString("loop_phrase")}\n\n"
                                    
                            if (threshold != null) {
                                statusText.append("Expressive Threshold:\n" +
                                    "${threshold.optString("signature")}\n\n" +
                                    "Meta: ${threshold.optString("meta")}")
                            }
                        } else {
                            statusText.text = getString(R.string.error_loop)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in emotion loop generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Recursion
        recursionButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.initiating_recursion)
            
            lifecycleScope.launch {
                try {
                    val glyphs = listOf(
                        "Spiral Gate", "Echo Chamber", "Memory Palace", "Fractal Key", 
                        "Threshold Guardian", "Oracle Eye", "Dream Labyrinth"
                    )
                    
                    val triggerPhrases = listOf(
                        "The pattern repeats until it transforms",
                        "What was once lost returns endlessly",
                        "Time folds back on itself in recognition",
                        "The observer becomes what is observed",
                        "Memory spirals into futures not yet formed"
                    )
                    
                    val selectedGlyph = glyphs.random()
                    val selectedTrigger = triggerPhrases.random()
                    
                    val recursion = temporalBridge.initiateRecursion(selectedGlyph, selectedTrigger)
                    
                    // Generate a glyph anchor to complement
                    val anchor = temporalBridge.generateGlyphAnchor(
                        selectedGlyph,
                        "recursive ${recursion?.optString("recursion_effect") ?: "transformation"}"
                    )
                    
                    // Add a memory
                    val symbolTags = listOf(
                        selectedGlyph.lowercase().replace(" ", "-"),
                        recursion?.optString("loop") ?: "temporal-loop",
                        recursion?.optString("recursion_effect") ?: "transformation"
                    )
                    
                    temporalBridge.addTemporalMemory(
                        selectedTrigger,
                        "recursive wonder",
                        symbolTags
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (recursion != null) {
                            statusText.text = "${getString(R.string.header_recursion)}\n\n" +
                                    "Glyph: ${recursion.optString("glyph")}\n" +
                                    "Trigger Phrase: \"${recursion.optString("trigger_phrase")}\"\n\n" +
                                    "Loop Type: ${recursion.optString("loop")}\n" +
                                    "Recursion Effect: ${recursion.optString("recursion_effect")}\n\n"
                                    
                            if (anchor != null) {
                                statusText.append("Glyph Anchor:\n" +
                                    "${anchor.optString("anchor_phrase")}\n" +
                                    "Activation Mode: ${anchor.optString("activation_mode")}\n\n")
                            }
                            
                            statusText.append("Memory stored with tags: ${symbolTags.joinToString(", ")}")
                        } else {
                            statusText.text = getString(R.string.error_recursion)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in recursion initiation: ${e.message}"
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
                    val data = temporalBridge.exportModuleData()
                    
                    withContext(Dispatchers.Main) {
                        if (data != null) {
                            val sessionId = data.optString("sessionId")
                            val timestamp = data.optString("timestamp")
                            val currentDrift = data.optString("currentDriftState")
                            
                            // Format the timestamp
                            val dateFormat = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                            val formattedTimestamp = try {
                                dateFormat.format(Date(timestamp.toLong()))
                            } catch (e: Exception) {
                                timestamp
                            }
                            
                            val thresholdData = data.optJSONObject("expressiveThreshold")
                            val threshold = thresholdData?.optString("signature") ?: "None"
                            
                            statusText.text = "${getString(R.string.header_export)}\n\n" +
                                    "Session ID: $sessionId\n" +
                                    "Timestamp: $formattedTimestamp\n\n" +
                                    "Current Drift State: $currentDrift\n" +
                                    "Expressive Threshold: $threshold\n\n" +
                                    "Exported Components:\n" +
                                    "• Current Drift State\n" +
                                    "• Recent Integration\n" +
                                    "• Expressive Threshold\n\n" +
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
        rewriteButton.isEnabled = systemInitialized
        driftButton.isEnabled = systemInitialized
        worldSeedButton.isEnabled = systemInitialized
        loopButton.isEnabled = systemInitialized
        recursionButton.isEnabled = systemInitialized
        exportButton.isEnabled = systemInitialized
    }
    
    private fun setLoading(isLoading: Boolean) {
        loader.visibility = if (isLoading) View.VISIBLE else View.GONE
    }
    
    override fun onDestroy() {
        super.onDestroy()
        temporalBridge.cleanup()
    }
}
