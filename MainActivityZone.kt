package com.zone.bridge

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

class MainActivityZone : AppCompatActivity() {
    private val TAG = "MainActivityZone"
    
    // UI components
    private lateinit var titleText: TextView
    private lateinit var statusText: TextView
    private lateinit var experienceButton: Button
    private lateinit var rewriteButton: Button
    private lateinit var driftButton: Button
    private lateinit var resonanceButton: Button
    private lateinit var dreamButton: Button
    private lateinit var historyButton: Button
    private lateinit var exportButton: Button
    private lateinit var loader: ProgressBar
    
    // Zone bridge
    private lateinit var zoneBridge: ZoneBridge
    
    // System state
    private var isSystemInitialized = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_zone)
        
        // Initialize UI components
        titleText = findViewById(R.id.title_text)
        statusText = findViewById(R.id.status_text)
        experienceButton = findViewById(R.id.experience_button)
        rewriteButton = findViewById(R.id.rewrite_button)
        driftButton = findViewById(R.id.drift_button)
        resonanceButton = findViewById(R.id.resonance_button)
        dreamButton = findViewById(R.id.dream_button)
        historyButton = findViewById(R.id.history_button)
        exportButton = findViewById(R.id.export_button)
        loader = findViewById(R.id.loader)
        
        // Set initial UI state
        setLoading(true)
        statusText.text = getString(R.string.initializing)
        
        // Initialize the Zone bridge in a background thread
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Initialize Zone bridge
                ZoneBridge.initialize(applicationContext)
                zoneBridge = ZoneBridge.getInstance(applicationContext)
                val success = zoneBridge.initializeSystem()
                
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
                    statusText.text = "Error initializing Zone system: ${e.message}"
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
                    val archetypes = zoneBridge.getAvailableArchetypes()
                    val driftStates = zoneBridge.getAvailableDriftStates()
                    
                    val zoneId = (1..9).random()
                    val emotions = listOf("wonder", "awe", "curiosity", "melancholy", "longing", "reverence")
                    val selectedEmotion = emotions.random()
                    val selectedDrift = driftStates?.random() ?: "Harmonic Coherence"
                    
                    val memoryElements = listOf(
                        "the echo of forgotten conversations",
                        "a melody from childhood",
                        "the scent of rain on distant mountains",
                        "a symbol etched in stone",
                        "the feeling of sunlight through leaves"
                    )
                    val selectedMemories = memoryElements.shuffled().take(2)
                    
                    val experience = zoneBridge.generateIntegratedZoneExperience(
                        zoneId,
                        selectedEmotion,
                        selectedDrift,
                        selectedMemories
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (experience != null) {
                            val zoneName = experience.optString("zone_name")
                            val zoneDefinition = experience.optString("zone_definition")
                            val driftResponse = experience.optString("drift_response")
                            val resonance = experience.optJSONObject("resonance")
                            val dream = experience.optJSONObject("dream")
                            
                            statusText.text = "${getString(R.string.header_experience)}\n\n" +
                                    "Zone $zoneId: $zoneName\n" +
                                    "Definition: $zoneDefinition\n\n" +
                                    "Emotional Tone: $selectedEmotion\n" +
                                    "Drift State: $selectedDrift\n" +
                                    "Drift Response: $driftResponse\n\n" +
                                    "Resonance Score: ${resonance?.optInt("score")}\n" +
                                    "Resonance Label: ${resonance?.optString("resonance")}\n\n" +
                                    "Dream Theme: ${dream?.optString("theme")}\n" +
                                    "Dream Symbol: ${dream?.optString("symbol")}\n\n" +
                                    "Dream Sequence:\n${dream?.optString("dream_sequence")}"
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
        
        // Zone Rewrite
        rewriteButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.rewriting_zone)
            
            lifecycleScope.launch {
                try {
                    val zoneId = (1..9).random()
                    
                    // Get current definition first
                    val currentDefinition = zoneBridge.getZoneDefinition(zoneId)
                    val currentName = currentDefinition?.optString("name") ?: "Unknown Zone"
                    val currentDef = currentDefinition?.optString("definition") ?: "No definition available"
                    
                    // Create new definition
                    val newNames = mapOf(
                        1 to "The Genesis Point",
                        2 to "The Reflecting Pool",
                        3 to "The Architect's Domain",
                        4 to "The Creative Flame",
                        5 to "The Balance Nexus",
                        6 to "The Crucible of Change",
                        7 to "The Wanderer's Path",
                        8 to "The Oracle's Lookout",
                        9 to "The Cosmic Summit"
                    )
                    
                    val newDefinitions = mapOf(
                        1 to "The point of emergence where potential becomes action.",
                        2 to "The mirror where self encounters other, creating meaning through reflection.",
                        3 to "The realm of structure and form, where order shapes chaos into pattern.",
                        4 to "The creative fire where expression transforms emotion into experience.",
                        5 to "The central point of balance where opposites find harmony and integration.",
                        6 to "The transformative threshold where death and rebirth cycle endlessly.",
                        7 to "The endless journey of discovery through uncharted territories.",
                        8 to "The visionary heights where glimpses of past and future converge.",
                        9 to "The transcendent peak where all zones unite in radiant completion."
                    )
                    
                    val newName = newNames[zoneId] ?: "Rewritten Zone $zoneId"
                    val newDefinition = newDefinitions[zoneId] ?: "A newly defined zone of symbolic significance."
                    
                    // Execute the rewrite
                    val rewriteResult = zoneBridge.rewriteZoneAndGenerateExperience(
                        zoneId,
                        newName,
                        newDefinition
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (rewriteResult != null) {
                            val rewrite = rewriteResult.optJSONObject("rewrite")
                            val experience = rewriteResult.optJSONObject("experience")
                            
                            val driftState = experience?.optString("drift_state")
                            val driftResponse = experience?.optString("drift_response")
                            val dream = experience?.optJSONObject("dream")
                            
                            statusText.text = "${getString(R.string.header_rewrite)}\n\n" +
                                    "Zone $zoneId\n\n" +
                                    "Original Name: $currentName\n" +
                                    "Original Definition: $currentDef\n\n" +
                                    "New Name: $newName\n" +
                                    "New Definition: $newDefinition\n\n" +
                                    "Drift State: $driftState\n" +
                                    "Drift Response: $driftResponse\n\n" +
                                    "Dream Sequence:\n${dream?.optString("dream_sequence")}"
                        } else {
                            statusText.text = getString(R.string.error_rewrite)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in zone rewriting: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Zone Drift
        driftButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.logging_drift)
            
            lifecycleScope.launch {
                try {
                    val zoneId = (1..9).random()
                    val archetypes = zoneBridge.getAvailableArchetypes()
                    val archetype = archetypes?.get(zoneId) ?: "Unknown Archetype"
                    
                    val driftStates = zoneBridge.getAvailableDriftStates()
                    val transitionType = driftStates?.random() ?: "Harmonic Coherence"
                    
                    val causes = listOf(
                        "symbolic intensification", 
                        "temporal recursion", 
                        "emotional resonance", 
                        "threshold crossing",
                        "memory echo"
                    )
                    
                    val effects = listOf(
                        "boundary dissolution", 
                        "conceptual mutation", 
                        "symbolic reconfiguration", 
                        "temporal echo formation",
                        "zone threshold blurring"
                    )
                    
                    // Log the drift
                    val driftEvent = zoneBridge.logZoneDrift(
                        "Zone $zoneId",
                        archetype,
                        transitionType,
                        causes.random(),
                        effects.random()
                    )
                    
                    // Generate a drift response
                    val driftResponse = zoneBridge.generateDriftResponse(
                        "Zone $zoneId", 
                        transitionType
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (driftEvent != null) {
                            statusText.text = "${getString(R.string.header_drift)}\n\n" +
                                    "Zone: ${driftEvent.optString("zone")}\n" +
                                    "Archetype: ${driftEvent.optString("archetype")}\n" +
                                    "Transition: ${driftEvent.optString("transition_type")}\n" +
                                    "Cause: ${driftEvent.optString("cause")}\n" +
                                    "Effect: ${driftEvent.optString("symbolic_effect")}\n\n" +
                                    "Drift Response:\n$driftResponse\n\n" +
                                    "Timestamp: ${driftEvent.optString("timestamp")}"
                        } else {
                            statusText.text = getString(R.string.error_drift)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in drift logging: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Zone Resonance
        resonanceButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.calculating_resonance)
            
            lifecycleScope.launch {
                try {
                    val zoneId = (1..9).random()
                    
                    val emotions = listOf("wonder", "awe", "curiosity", "melancholy", "longing", "reverence")
                    val selectedEmotion = emotions.random()
                    
                    val symbols = listOf(
                        "threshold crossing",
                        "spiral recursion",
                        "symbolic fold",
                        "memory echo",
                        "time ripple",
                        "dream fragment",
                        "crystalline structure",
                        "harmonic wave"
                    )
                    val selectedSymbols = symbols.shuffled().take(3)
                    
                    // Calculate resonance
                    val resonance = zoneBridge.calculateResonance(
                        selectedEmotion,
                        selectedSymbols,
                        "Zone $zoneId"
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (resonance != null) {
                            statusText.text = "${getString(R.string.header_resonance)}\n\n" +
                                    "Zone: ${resonance.optString("zone")}\n" +
                                    "Emotional State: ${resonance.optString("emotion")}\n\n" +
                                    "Symbolic Elements:\n" +
                                    selectedSymbols.joinToString("\n") { "• $it" } + "\n\n" +
                                    "Resonance Label: ${resonance.optString("resonance")}\n" +
                                    "Resonance Score: ${resonance.optString("score")}\n\n" +
                                    "Timestamp: ${resonance.optString("timestamp")}"
                        } else {
                            statusText.text = getString(R.string.error_resonance)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in resonance calculation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Zone Dream
        dreamButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.generating_dream)
            
            lifecycleScope.launch {
                try {
                    val zoneId = (1..9).random()
                    
                    val emotions = listOf("wonder", "awe", "curiosity", "melancholy", "longing", "reverence")
                    val selectedEmotion = emotions.random()
                    
                    val memoryElements = listOf(
                        "the echo of forgotten conversations",
                        "a melody from childhood",
                        "the scent of rain on distant mountains",
                        "a symbol etched in stone",
                        "the feeling of sunlight through leaves",
                        "whispers from the void",
                        "a color that has no name",
                        "the touch of wind through hair"
                    )
                    val selectedMemories = memoryElements.shuffled().take(3)
                    
                    // Generate dream
                    val dream = zoneBridge.generateZoneTunedDream(
                        selectedMemories,
                        zoneId,
                        selectedEmotion
                    )
                    
                    withContext(Dispatchers.Main) {
                        if (dream != null) {
                            statusText.text = "${getString(R.string.header_dream)}\n\n" +
                                    "Zone: ${dream.optInt("zone")}\n" +
                                    "Theme: ${dream.optString("theme")}\n" +
                                    "Emotional Tone: ${dream.optString("emotion")}\n\n" +
                                    "Memory Fragments:\n" +
                                    selectedMemories.joinToString("\n") { "• $it" } + "\n\n" +
                                    "Symbol: ${dream.optString("symbol")}\n\n" +
                                    "Dream Sequence:\n${dream.optString("dream_sequence")}"
                        } else {
                            statusText.text = getString(R.string.error_dream)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in dream generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Zone History
        historyButton.setOnClickListener {
            if (!isSystemInitialized) {
                statusText.text = getString(R.string.system_not_initialized)
                return@setOnClickListener
            }
            
            setLoading(true)
            statusText.text = getString(R.string.retrieving_history)
            
            lifecycleScope.launch {
                try {
                    val zoneId = (1..9).random()
                    val zoneName = "Zone $zoneId"
                    
                    // Get zone history
                    val history = zoneBridge.getZoneHistory(zoneName)
                    
                    // Get zone definition
                    val definition = zoneBridge.getZoneDefinition(zoneId)
                    val zoneDef = definition?.optString("name") ?: "Unknown Zone"
                    
                    withContext(Dispatchers.Main) {
                        if (history != null) {
                            val historyText = StringBuilder()
                            historyText.append("$zoneName: $zoneDef\n\n")
                            
                            if (history.length() > 0) {
                                historyText.append("Drift Events:\n\n")
                                
                                for (i in 0 until history.length()) {
                                    val event = history.getJSONObject(i)
                                    historyText.append("Event ${i+1}:\n")
                                    historyText.append("• Transition: ${event.optString("transition_type")}\n")
                                    historyText.append("• Cause: ${event.optString("cause")}\n")
                                    historyText.append("• Effect: ${event.optString("symbolic_effect")}\n\n")
                                }
                            } else {
                                historyText.append("No drift events recorded for this zone.")
                            }
                            
                            statusText.text = "${getString(R.string.header_history)}\n\n" + historyText.toString()
                        } else {
                            statusText.text = getString(R.string.error_history)
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error in history retrieval: ${e.message}"
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
                    val data = zoneBridge.exportModuleData()
                    
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
                                    "• Recent Integration\n" +
                                    "• Zone Definitions\n\n" +
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
        resonanceButton.isEnabled = systemInitialized
        dreamButton.isEnabled = systemInitialized
        historyButton.isEnabled = systemInitialized
        exportButton.isEnabled = systemInitialized
    }
    
    private fun setLoading(isLoading: Boolean) {
        loader.visibility = if (isLoading) View.VISIBLE else View.GONE
    }
    
    override fun onDestroy() {
        super.onDestroy()
        zoneBridge.cleanup()
    }
}
