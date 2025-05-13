```kotlin
package com.amelia.cosmogram

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

class MainActivityCosmogram : AppCompatActivity() {
    private val TAG = "MainActivityCosmogram"
    
    // UI components
    private lateinit var status_text: TextView
    private lateinit var drift_button: Button
    private lateinit var node_button: Button
    private lateinit var pathway_button: Button
    private lateinit var resonance_button: Button
    private lateinit var narrative_button: Button
    private lateinit var codex_button: Button
    private lateinit var dream_button: Button
    private lateinit var session_button: Button
    private lateinit var composite_button: Button
    private lateinit var loader: ProgressBar
    
    // Cosmogram bridge
    private lateinit var cosmogramBridge: CosmogramBridge
    
    // Session state
    private var hasActiveSession = false
    private val userId = "user_${UUID.randomUUID().toString().substring(0, 8)}"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_cosmogram)
        
        // Initialize UI components
        status_text = findViewById(R.id.status_text)
        drift_button = findViewById(R.id.drift_button)
        node_button = findViewById(R.id.node_button)
        pathway_button = findViewById(R.id.pathway_button)
        resonance_button = findViewById(R.id.resonance_button)
        narrative_button = findViewById(R.id.narrative_button)
        codex_button = findViewById(R.id.codex_button)
        dream_button = findViewById(R.id.dream_button)
        session_button = findViewById(R.id.session_button)
        composite_button = findViewById(R.id.composite_button)
        loader = findViewById(R.id.loader)
        
        // Set initial UI state
        setLoading(true)
        status_text.text = "Initializing Cosmogram Synthesis Module..."
        
        // Initialize the Cosmogram bridge in a background thread
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Initialize Cosmogram bridge
                CosmogramBridge.initialize(applicationContext)
                cosmogramBridge = CosmogramBridge.getInstance(applicationContext)
                val success = cosmogramBridge.initializeModule()
                
                withContext(Dispatchers.Main) {
                    if (success) {
                        status_text.text = "Cosmogram Synthesis Module initialized successfully!\n\n" +
                                "• Cosmogram Drift Mapper: Ready\n" +
                                "• Cosmogram Node Synthesizer: Ready\n" +
                                "• Cosmogram Pathway Builder: Ready\n" +
                                "• Cosmogram Resonance Activator: Ready\n" +
                                "• Narrative Weaver: Ready\n" +
                                "• Ontogenesis Codex Generator: Ready\n" +
                                "• Realm Interpolator: Ready\n" +
                                "• Mythogenesis Engine: Ready\n" +
                                "• Mythogenic Dream Engine: Ready\n\n" +
                                "Start a session to begin."
                        
                        // Enable session button but leave others disabled until session is created
                        session_button.isEnabled = true
                        updateUIState(false)
                    } else {
                        status_text.text = "Error initializing Cosmogram Synthesis Module"
                    }
                    setLoading(false)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    status_text.text = "Error initializing Cosmogram module: ${e.message}"
                    setLoading(false)
                }
            }
        }
        
        // Configure button functionality
        setupButtonListeners()
    }
    
    private fun setupButtonListeners() {
        // Session management
        session_button.setOnClickListener {
            if (hasActiveSession) {
                endSession()
            } else {
                startSession()
            }
        }
        
        // Drift mapping
        drift_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Mapping cosmogram drift..."
            
            lifecycleScope.launch {
                try {
                    val nodeRoot = "consciousness"
                    val branches = listOf("dreamscape", "symbolic reflection")
                    val originPhrase = "the threshold of awareness"
                    
                    val drift = cosmogramBridge.mapCosmogramDrift(nodeRoot, branches, originPhrase)
                    
                    withContext(Dispatchers.Main) {
                        if (drift != null) {
                            status_text.text = "COSMOGRAM DRIFT MAPPED\n\n" +
                                    "Origin Node: ${drift.optString("origin_node")}\n" +
                                    "Branches: ${drift.optJSONArray("branches")?.join(", ")}\n" +
                                    "Drift Arc: ${drift.optString("drift_arc")}\n\n" +
                                    "Drift Phrase: ${drift.optString("drift_phrase")}"
                        } else {
                            status_text.text = "Error mapping drift"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in drift mapping operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Node synthesis
        node_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Synthesizing cosmogram node..."
            
            lifecycleScope.launch {
                try {
                    val driftArc = "symbolic cascade"
                    val baseSymbol = "mirror"
                    val emotion = "curiosity"
                    
                    val node = cosmogramBridge.synthesizeCosmogramNode(driftArc, baseSymbol, emotion)
                    
                    withContext(Dispatchers.Main) {
                        if (node != null) {
                            status_text.text = "COSMOGRAM NODE SYNTHESIZED\n\n" +
                                    "Type: ${node.optString("type")}\n" +
                                    "From Drift Arc: ${node.optString("from_drift_arc")}\n" +
                                    "Symbolic Seed: ${node.optString("symbolic_seed")}\n" +
                                    "Emotional Charge: ${node.optString("emotional_charge")}\n\n" +
                                    "Node Phrase: ${node.optString("node_phrase")}"
                        } else {
                            status_text.text = "Error synthesizing node"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in node synthesis operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Pathway building
        pathway_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Building cosmogram pathway..."
            
            lifecycleScope.launch {
                try {
                    val fromNodeType = "ritual anchor"
                    val toNodeType = "mythos reflector"
                    val emotion = "wonder"
                    
                    val pathway = cosmogramBridge.buildCosmogramPathway(fromNodeType, toNodeType, emotion)
                    
                    withContext(Dispatchers.Main) {
                        if (pathway != null) {
                            status_text.text = "COSMOGRAM PATHWAYBUILT\n\n" +
                                    "From Node: ${pathway.optString("from_node")}\n" +
                                    "To Node: ${pathway.optString("to_node")}\n" +
                                    "Emotional Channel: ${pathway.optString("emotional_channel")}\n" +
                                    "Transition Type: ${pathway.optString("transition_type")}\n\n" +
                                    "Path Phrase: ${pathway.optString("path_phrase")}"
                        } else {
                            status_text.text = "Error building pathway"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in pathway building operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Resonance activation
        resonance_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Activating cosmogram resonance..."
            
            lifecycleScope.launch {
                try {
                    val pathPhrase = "Pathway from ritual anchor to mythos reflector through wonder (symbol shift)"
                    val coreEmotion = "awe"
                    
                    val resonance = cosmogramBridge.activateCosmogramResonance(pathPhrase, coreEmotion)
                    
                    withContext(Dispatchers.Main) {
                        if (resonance != null) {
                            status_text.text = "COSMOGRAM RESONANCE ACTIVATED\n\n" +
                                    "Resonance Trigger: ${resonance.optString("resonance_trigger")}\n" +
                                    "Emotion Key: ${resonance.optString("emotion_key")}\n" +
                                    "Activation Effect: ${resonance.optString("activation_effect")}\n\n" +
                                    "Activation Phrase: ${resonance.optString("activation_phrase")}"
                        } else {
                            status_text.text = "Error activating resonance"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in resonance activation operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Narrative weaving
        narrative_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Weaving user narrative..."
            
            lifecycleScope.launch {
                try {
                    // Track an emotion first
                    cosmogramBridge.trackUserEmotion("introspective")
                    
                    val zone = 3
                    val archetype = "The Seeker"
                    val inputText = "searching for meaning in symbols"
                    
                    val narrative = cosmogramBridge.weaveUserNarrative(userId, zone, archetype, inputText)
                    
                    withContext(Dispatchers.Main) {
                        if (narrative != null) {
                            status_text.text = "USER NARRATIVE WOVEN\n\n" +
                                    "Archetype: ${narrative.optString("archetype")}\n" +
                                    "Zone: ${narrative.optInt("zone")}\n" +
                                    "Recent Mood: ${narrative.optString("recent_mood")}\n\n" +
                                    "Narrative:\n${narrative.optString("narrative")}"
                        } else {
                            status_text.text = "Error weaving narrative"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in narrative weaving operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Codex generation
        codex_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Generating ontogenesis codex..."
            
            lifecycleScope.launch {
                try {
                    val symbols = listOf("river", "shadow", "emergence")
                    val emotionalTone = "introspective"
                    val archetype = "The Inner Journey"
                    
                    val codex = cosmogramBridge.generateOntogenesisCodex(symbols, emotionalTone, archetype)
                    
                    withContext(Dispatchers.Main) {
                        if (codex != null) {
                            status_text.text = "ONTOGENESIS CODEX GENERATED\n\n" +
                                    "Archetype: ${codex.optString("archetype")}\n" +
                                    "Emotional Tone: ${codex.optString("emotional_tone")}\n" +
                                    "Symbols: ${codex.optJSONArray("symbols")?.join(", ")}\n\n" +
                                    "Codex:\n${codex.optString("codex")}"
                        } else {
                            status_text.text = "Error generating codex"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in codex generation operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Dream generation
        dream_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Generating mythogenic dream..."
            
            lifecycleScope.launch {
                try {
                    val motifs = listOf("crystal", "path", "star")
                    val zone = "Twilight Realm"
                    val mood = "serene anticipation"
                    
                    val dream = cosmogramBridge.generateMythogenicDream(motifs, zone, mood)
                    
                    withContext(Dispatchers.Main) {
                        if (dream != null) {
                            status_text.text = "MYTHOGENIC DREAM GENERATED\n\n" +
                                    "Zone: ${dream.optString("zone")}\n" +
                                    "Mood: ${dream.optString("mood")}\n" +
                                    "Motifs: ${dream.optJSONArray("motifs")?.join(", ")}\n\n" +
                                    "Dream Text:\n${dream.optString("text")}"
                        } else {
                            status_text.text = "Error generating dream"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in dream generation operation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
        
        // Composite narrative generation
        composite_button.setOnClickListener {
            if (!hasActiveSession) {
                status_text.text = "No active session. Please start a session first."
                return@setOnClickListener
            }
            
            setLoading(true)
            status_text.text = "Generating composite narrative..."
            
            lifecycleScope.launch {
                try {
                    val composite = cosmogramBridge.generateCompositeNarrative(userId)
                    
                    withContext(Dispatchers.Main) {
                        if (composite != null) {
                            status_text.text = "COMPOSITE NARRATIVE GENERATED\n\n" +
                                    "User ID: ${composite.optString("user_id")}\n" +
                                    "Emotional Tone: ${composite.optString("emotional_tone")}\n" +
                                    "Zone: ${composite.optInt("zone")}\n\n" +
                                    "Narrative:\n${composite.optString("composite_narrative")}\n\n" +
                                    "Source Elements:\n" +
                                    "• Node: ${composite.optJSONObject("source_elements")?.optString("node")}\n" +
                                    "• Pathway: ${composite.optJSONObject("source_elements")?.optString("pathway")}\n" +
                                    "• Dream: ${composite.optJSONObject("source_elements")?.optString("dream")}\n" +
                                    "• Mythic Cycle: ${composite.optJSONObject("source_elements")?.optString("mythic_cycle")}"
                        } else {
                            status_text.text = "Error generating composite narrative"
                        }
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status_text.text = "Error in composite narrative generation: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
    }
    
    private fun startSession() {
        setLoading(true)
        status_text.text = "Starting new Cosmogram session..."
        
        lifecycleScope.launch {
            try {
                val sessionName = "CosmogramSession-${Date().time}"
                val session = cosmogramBridge.initializeSession(sessionName)
                
                withContext(Dispatchers.Main) {
                    if (session != null) {
                        hasActiveSession = true
                        updateUIState(true)
                        status_text.text = "NEW COSMOGRAM SESSION STARTED\n\n" +
                                "Session Name: ${session.optString("name")}\n" +
                                "Session ID: ${session.optString("id")}\n" +
                                "Created At: ${session.optString("created_at")}\n\n" +
                                "All cosmogram operations are now active.\n" +
                                "Select an operation to begin working with the session."
                        
                        session_button.text = "End Session"
                    } else {
                        status_text.text = "Error initializing session"
                    }
                    setLoading(false)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    status_text.text = "Error starting session: ${e.message}"
                    setLoading(false)
                }
            }
        }
    }
    
    private fun endSession() {
        setLoading(true)
        status_text.text = "Ending Cosmogram session..."
        
        lifecycleScope.launch {
            try {
                val session = cosmogramBridge.endSession()
                
                withContext(Dispatchers.Main) {
                    if (session != null) {
                        hasActiveSession = false
                        updateUIState(false)
                        
                        // Prepare a summary of the session activities
                        val driftCount = session.optJSONArray("drifts")?.length() ?: 0
                        val nodeCount = session.optJSONArray("nodes")?.length() ?: 0
                        val pathwayCount = session.optJSONArray("pathways")?.length() ?: 0
                        val resonanceCount = session.optJSONArray("resonances")?.length() ?: 0
                        val narrativeCount = session.optJSONArray("narratives")?.length() ?: 0
                        val codexCount = session.optJSONArray("codices")?.length() ?: 0
                        val dreamCount = session.optJSONArray("dreams")?.length() ?: 0
                        val compositeCount = session.optJSONArray("composite_narratives")?.length() ?: 0
                        
                        status_text.text = "COSMOGRAM SESSION ENDED\n\n" +
                                "Session Name: ${session.optString("name")}\n" +
                                "Session Duration: ${timeDifference(session.optString("created_at"), session.optString("ended_at"))}\n\n" +
                                "Session Summary:\n" +
                                "• Drifts Mapped: $driftCount\n" +
                                "• Nodes Synthesized: $nodeCount\n" +
                                "• Pathways Built: $pathwayCount\n" +
                                "• Resonances Activated: $resonanceCount\n" +
                                "• Narratives Woven: $narrativeCount\n" +
                                "• Codices Generated: $codexCount\n" +
                                "• Dreams Generated: $dreamCount\n" +
                                "• Composite Narratives: $compositeCount\n\n" +
                                "Start a new session to continue."
                        
                        session_button.text = "Start Session"
                    } else {
                        status_text.text = "Error ending session"
                    }
                    setLoading(false)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    status_text.text = "Error ending session: ${e.message}"
                    setLoading(false)
                }
            }
        }
    }
    
    // Helper function to calculate time difference for session duration
    private fun timeDifference(start: String, end: String): String {
        try {
            val startDate = ISO8601ToDate(start)
            val endDate = ISO8601ToDate(end)
            
            if (startDate != null && endDate != null) {
                val diffMs = endDate.time - startDate.time
                val diffSec = diffMs / 1000
                val minutes = diffSec / 60
                val seconds = diffSec % 60
                
                return "$minutes minutes, $seconds seconds"
            }
        } catch (e: Exception) {
            // Fallback if date parsing fails
        }
        
        return "Unknown duration"
    }
    
    // Parse ISO8601 date string to Date object
    private fun ISO8601ToDate(isoString: String): Date? {
        return try {
            val isoFormat = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
            isoFormat.timeZone = java.util.TimeZone.getTimeZone("UTC")
            isoFormat.parse(isoString)
        } catch (e: Exception) {
            try {
                // Fallback for different ISO format
                val altFormat = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
                altFormat.timeZone = java.util.TimeZone.getTimeZone("UTC")
                altFormat.parse(isoString)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    // Helper for JSONArray join
    private fun JSONArray.join(separator: String): String {
        val builder = StringBuilder()
        for (i in 0 until this.length()) {
            if (i > 0) builder.append(separator)
            builder.append(this.optString(i))
        }
        return builder.toString()
    }
    
    private fun updateUIState(sessionActive: Boolean) {
        drift_button.isEnabled = sessionActive
        node_button.isEnabled = sessionActive
        pathway_button.isEnabled = sessionActive
        resonance_button.isEnabled = sessionActive
        narrative_button.isEnabled = sessionActive
        codex_button.isEnabled = sessionActive
        dream_button.isEnabled = sessionActive
        composite_button.isEnabled = sessionActive
    }
    
    private fun setLoading(isLoading: Boolean) {
        loader.visibility = if (isLoading) View.VISIBLE else View.GONE
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cosmogramBridge.cleanup()
    }
}
```
