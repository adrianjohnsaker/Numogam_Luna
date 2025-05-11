// MainActivitySymbolic.kt

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

class MainActivity : AppCompatActivity() {
    // UI components
    private lateinit var statusText: TextView
    private lateinit var processButton: Button
    private lateinit var exploreButton: Button
    private lateinit var adaptButton: Button
    private lateinit var ethicalButton: Button
    private lateinit var memoryButton: Button
    private lateinit var loader: ProgressBar

    // Symbolic bridge
    private lateinit var symbolicBridge: AmeliaSymbolicBridge
    
    // Sample user data for testing
    private val userId = "user_${System.currentTimeMillis()}"
    private val memoryElements = listOf(
        "A conversation about dreams",
        "The sensation of wind", 
        "A moment of clarity"
    )
    private val emotionalTones = listOf("joy", "sadness", "curiosity", "fear", "awe", "confusion")
    private val archetypes = listOf(
        "The Mirror", "The Artist", "The Explorer", "The Mediator", 
        "The Architect", "The Transformer", "The Oracle", "The Initiator", "The Enlightened"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize UI components
        statusText = findViewById(R.id.status_text)
        processButton = findViewById(R.id.process_button)
        exploreButton = findViewById(R.id.explore_button)
        adaptButton = findViewById(R.id.adapt_button)
        ethicalButton = findViewById(R.id.ethical_button)
        memoryButton = findViewById(R.id.memory_button)
        loader = findViewById(R.id.loader)
        
        // Set initial UI state
        setLoading(true)
        statusText.text = "Initializing Amelia Symbolic Bridge..."
        
        // Initialize the symbolic bridge in a background thread
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                symbolicBridge = AmeliaSymbolicBridge(applicationContext)
                
                // Check if bridge is initialized correctly by getting component status
                val status = symbolicBridge.getComponentStatus()
                
                withContext(Dispatchers.Main) {
                    // Update UI with initialization status
                    if (status.optString("status", "") == "error") {
                        statusText.text = "Error initializing bridge: ${status.optString("error", "Unknown error")}"
                    } else {
                        statusText.text = "Amelia Symbolic Bridge initialized successfully!\n\n" +
                                "Ready to process symbolic narratives.\n\n" +
                                "Component Status:\n" +
                                formatComponentStatus(status)
                        
                        // Enable buttons
                        enableButtons(true)
                    }
                    setLoading(false)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "Error initializing bridge: ${e.message}"
                    setLoading(false)
                }
            }
        }
        
        // Set up button click listeners
        setupButtonListeners()
    }
    
    private fun setupButtonListeners() {
        // Process button - Generate a complex symbolic narrative
        processButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    // Get random values for testing
                    val emotionalTone = emotionalTones.random()
                    val currentZone = (1..9).random()
                    val archetype = archetypes.random()
                    val recentInput = "Exploring the symbolic landscape"
                    
                    val result = symbolicBridge.generateSymbolicNarrative(
                        userId = userId,
                        memoryElements = memoryElements,
                        emotionalTone = emotionalTone,
                        currentZone = currentZone,
                        archetype = archetype,
                        recentInput = recentInput
                    )
                    
                    withContext(Dispatchers.Main) {
                        displayNarrativeResult(result, "Process", emotionalTone, currentZone, archetype)
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error processing symbolic narrative: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Explore button - Generate a morphogenetic wave
        exploreButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    val result = symbolicBridge.generateMorphogeneticWave()
                    
                    withContext(Dispatchers.Main) {
                        if (result.optString("status") == "success") {
                            val wave = result.optJSONObject("data")?.optJSONObject("wave")
                            statusText.text = "MORPHOGENETIC WAVE EXPLORATION\n\n" +
                                    "Wave Type: ${wave?.optString("morphogenetic_wave")}\n" +
                                    "State Field: ${wave?.optString("state_field")}\n" +
                                    "Symbolic Signature: ${wave?.optString("symbolic_signature")}\n\n" +
                                    "Meta: ${wave?.optString("meta")}\n" +
                                    "Timestamp: ${wave?.optString("timestamp")}"
                        } else {
                            statusText.text = "Error exploring wave: ${result.optString("error")}"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error generating wave: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Adapt button - Generate a resonance pulse
        adaptButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    val result = symbolicBridge.generateResonancePulse()
                    
                    withContext(Dispatchers.Main) {
                        if (result.optString("status") == "success") {
                            val pulse = result.optJSONObject("data")?.optJSONObject("pulse")
                            statusText.text = "INTUITIVE RESONANCE PULSE\n\n" +
                                    "Tone: ${pulse?.optString("tone")}\n" +
                                    "Form: ${pulse?.optString("form")}\n" +
                                    "Field: ${pulse?.optString("field")}\n\n" +
                                    "Pulse: ${pulse?.optString("pulse")}\n\n" +
                                    "Meta: ${pulse?.optString("meta")}\n" +
                                    "Timestamp: ${pulse?.optString("timestamp")}"
                        } else {
                            statusText.text = "Error adapting pulse: ${result.optString("error")}"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error generating pulse: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Ethical button - Generate a metasigil for an ethical concept
        ethicalButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    // Generate a sigil for an ethical concept
                    val ethicalConcept = listOf("Compassion", "Autonomy", "Justice", "Integrity", "Responsibility").random()
                    val result = symbolicBridge.generateMetasigil(ethicalConcept)
                    
                    withContext(Dispatchers.Main) {
                        if (result.optString("status") == "success") {
                            val sigil = result.optJSONObject("data")?.optJSONObject("sigil")
                            statusText.text = "ETHICAL METASIGIL\n\n" +
                                    "Concept: ${sigil?.optString("name")}\n" +
                                    "Energy Source: ${sigil?.optString("energy_source")}\n" +
                                    "Structure Style: ${sigil?.optString("structure_style")}\n\n" +
                                    "Sigil Formula: ${sigil?.optString("sigil_formula")}\n\n" +
                                    "Insight: ${sigil?.optString("insight")}\n" +
                                    "Timestamp: ${sigil?.optString("timestamp")}"
                        } else {
                            statusText.text = "Error generating ethical sigil: ${result.optString("error")}"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error generating ethical sigil: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Memory button - Get component status to check memory systems
        memoryButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    val status = symbolicBridge.getComponentStatus()
                    
                    withContext(Dispatchers.Main) {
                        statusText.text = "MEMORY SYSTEMS STATUS\n\n" + formatComponentStatus(status)
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error checking memory systems: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
    }
    
    private fun displayNarrativeResult(result: JSONObject, operation: String, emotion: String, zone: Int, archetype: String) {
        if (result.optString("status") == "success") {
            val data = result.optJSONObject("data")
            val mainNarrative = data?.optString("main_narrative")
            
            val dream = data?.optJSONObject("dream")
            val wave = data?.optJSONObject("morphogenetic_wave")
            val pulse = data?.optJSONObject("resonance_pulse")
            
            statusText.text = "SYMBOLIC NARRATIVE PROCESS\n\n" +
                    "Operation: $operation\n" +
                    "Emotional Tone: $emotion\n" +
                    "Zone: $zone\n" +
                    "Archetype: $archetype\n\n" +
                    "MAIN NARRATIVE:\n$mainNarrative\n\n" +
                    "DREAM THEME: ${dream?.optString("dream_theme")}\n" +
                    "METAPHOR: ${dream?.optString("metaphor")}\n" +
                    "WAVE: ${wave?.optString("symbolic_signature")}\n" +
                    "PULSE: ${pulse?.optString("pulse")}"
        } else {
            statusText.text = "Error: ${result.optString("error", "Unknown error")}"
        }
    }
    
    private fun formatComponentStatus(statusJson: JSONObject): String {
        val dreamGenerator = statusJson.optJSONObject("dream_generator")
        val mythogenicEngine = statusJson.optJSONObject("mythogenic_engine")
        val ontologyGenerator = statusJson.optJSONObject("ontology_generator")
        val symbolicEvolution = statusJson.optJSONObject("symbolic_evolution")
        val phaseDriftEngine = statusJson.optJSONObject("phase_drift_engine")
        val archetypeMutationTracker = statusJson.optJSONObject("archetype_mutation_tracker")
        
        return "Dream Archive Size: ${dreamGenerator?.optInt("archive_size", 0)}\n" +
                "Mythogenic Archive Size: ${mythogenicEngine?.optInt("archive_size", 0)}\n" +
                "Myth Map Size: ${mythogenicEngine?.optInt("myth_map_size", 0)}\n" +
                "Ontology Log Size: ${ontologyGenerator?.optInt("log_size", 0)}\n" +
                "Symbol Count: ${symbolicEvolution?.optInt("symbol_count", 0)}\n" +
                "Phase State Count: ${phaseDriftEngine?.optInt("state_count", 0)}\n" +
                "Mutation Count: ${archetypeMutationTracker?.optInt("mutation_count", 0)}\n\n" +
                "Themes: ${statusJson.optJSONArray("themes")}\n" +
                "Motifs: ${statusJson.optJSONArray("motifs")}\n\n" +
                "Last Updated: ${statusJson.optString("timestamp")}"
    }
    
    private fun setLoading(isLoading: Boolean) {
        loader.visibility = if (isLoading) View.VISIBLE else View.GONE
    }
    
    private fun enableButtons(enabled: Boolean) {
        processButton.isEnabled = enabled
        exploreButton.isEnabled = enabled
        adaptButton.isEnabled = enabled
        ethicalButton.isEnabled = enabled
        memoryButton.isEnabled = enabled
    }
}
```
