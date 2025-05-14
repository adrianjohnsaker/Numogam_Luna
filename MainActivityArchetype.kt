package com.antonio.my.ai.girlfriend.free.archetypal

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

class MainActivityArchetype : AppCompatActivity() {
    // UI components
    private lateinit var titleText: TextView
    private lateinit var statusText: TextView
    private lateinit var sessionButton: Button
    private lateinit var driftButton: Button
    private lateinit var nodeButton: Button
    private lateinit var loader: ProgressBar

    // Archetypal bridge instance
    private lateinit var archetypalBridge: AmeliaArchetypeBridge

    // Sample data for demonstration
    private val archetypes = listOf(
        "MIRROR", "EXPLORER", "TRANSFORMER",
        "ORACLE", "INITIATOR"
    )
    private val emotionalTones = listOf(
        "CURIOSITY", "AWE", "MELANCHOLY",
        "EUPHORIA", "MYSTERY", "DREAD", "SERENITY"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_archetype)

        // Initialize UI components
        titleText = findViewById(R.id.title_text)
        statusText = findViewById(R.id.status_text)
        sessionButton = findViewById(R.id.session_button)
        driftButton = findViewById(R.id.drift_button)
        nodeButton = findViewById(R.id.node_button)
        loader = findViewById(R.id.loader)

        // Set initial UI state
        setLoading(true)
        titleText.text = "Archetypal Consciousness Engine"
        statusText.text = "Initializing archetypal systems..."

        // Initialize the archetypal bridge on a background thread
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                AmeliaArchetypeBridge.initialize(applicationContext)
                archetypalBridge = AmeliaArchetypeBridge.getInstance(applicationContext)
                val initialized = archetypalBridge.getComponentStatus().optString("status") != "error"
                withContext(Dispatchers.Main) {
                    if (initialized) {
                        statusText.text = "Archetypal systems initialized successfully!\n\n" +
                                "Available archetypes:\n${archetypes.joinToString("\n")}\n\n" +
                                "Emotional tones:\n${emotionalTones.joinToString("\n")}"
                        enableButtons(true)
                    } else {
                        statusText.text = "Error initializing archetypal systems"
                    }
                    setLoading(false)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "Critical error: ${e.message}"
                    setLoading(false)
                }
            }
        }

        // Set up button click listeners
        setupButtonListeners()
    }

    private fun setupButtonListeners() {
        // Session button: start or end an archetypal session
        sessionButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    if (getCurrentSessionId() == null) {
                        val session = archetypalBridge.createSession("UserSession_${System.currentTimeMillis()}")
                        withContext(Dispatchers.Main) {
                            sessionButton.text = "End Session"
                            statusText.text = "Session started: ${session.optString("id")}"
                            setLoading(false)
                        }
                    } else {
                        val session = archetypalBridge.endSession()
                        withContext(Dispatchers.Main) {
                            sessionButton.text = "Start Session"
                            statusText.text = "Session ended: ${session.optString("id")}"
                            setLoading(false)
                        }
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Session error: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }

        // Drift button: generate archetypal drift
        driftButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    val archetype = archetypes.random()
                    val zone = (1..9).random()  // Using an integer zone depth
                    // For simplicity, we call generateComplexArchetype which returns the full synthesis
                    val result = archetypalBridge.generateComplexArchetype(archetype, emotionalTones.random(), zone)
                    withContext(Dispatchers.Main) {
                        displayDriftResult(result, archetype)
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Drift error: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }

        // Node button: synthesize an archetypal node (the full complex archetype)
        nodeButton.setOnClickListener {
            setLoading(true)
            lifecycleScope.launch {
                try {
                    val archetype = archetypes.random()
                    val tone = emotionalTones.random()
                    val zone = (1..9).random()
                    val result = archetypalBridge.generateComplexArchetype(archetype, tone, zone)
                    withContext(Dispatchers.Main) {
                        displayNodeResult(result)
                        setLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Node synthesis error: ${e.message}"
                        setLoading(false)
                    }
                }
            }
        }
    }

    private fun displayDriftResult(result: JSONObject?, originalArchetype: String) {
        if (result?.optString("status") == "success") {
            val data = result.optJSONObject("data")
            val sb = StringBuilder()
            sb.append("ARCHETYPAL DRIFT RESULT\n\n")
            sb.append("Original: $originalArchetype\n")
            // Show drift evolution details
            val drift = data?.optJSONObject("drift_evolution")
            if (drift != null) {
                sb.append("Drifted Form: ${drift.optString("drifted_form", "N/A")}\n")
                sb.append("Drift Value: ${drift.optDouble("drift_value", 0.0)}\n")
                sb.append("Drift Similarity: ${drift.optString("similarity", "N/A")}\n")
            }
            // Also include a snippet from mutated form
            val mut = data?.optJSONObject("mutated_form")
            if (mut != null) {
                sb.append("\n-- Mutated Form --\n")
                sb.append("Mutated: ${mut.optString("mutated", "N/A")}\n")
                sb.append("Mutation Factor: ${mut.optDouble("mutation_factor", 0.0)}\n")
            }
            sb.append("\nTimestamp: ${data?.optString("timestamp", "N/A")}")
            statusText.text = sb.toString()
        } else {
            statusText.text = "Error: ${result?.optString("error", "Unknown error")}"
        }
    }

    private fun displayNodeResult(result: JSONObject?) {
        if (result?.optString("status") == "success") {
            val data = result.optJSONObject("data")
            val sb = StringBuilder()
            sb.append("ARCHETYPAL NODE SYNTHESIS\n\n")
            sb.append("Base Archetype: ${data?.optString("base_archetype", "N/A")}\n")
            // Although emotional tone is an input parameter, you can show related data if available
            sb.append("Zone Depth: ${data?.optInt("temporal_depth", 0)}\n")
            
            sb.append("\n-- Refracted Memory --\n")
            val mem = data?.optJSONObject("refracted_memory")
            if (mem != null) {
                sb.append("Projection: ${mem.optString("projection", "N/A")}\n")
                sb.append("Details: ${mem.optString("details", "N/A")}\n")
                sb.append("Similarity: ${mem.optString("similarity", "N/A")}\n")
            }
            
            sb.append("\n-- Resonance Profile --\n")
            val resonance = data?.optJSONObject("resonance_profile")
            if (resonance != null) {
                for (key in resonance.keys()) {
                    sb.append("$key: ${resonance.optString(key, "N/A")}\n")
                }
            }
            
            sb.append("\n-- Drift Evolution --\n")
            val drift = data?.optJSONObject("drift_evolution")
            if (drift != null) {
                sb.append("Drifted Form: ${drift.optString("drifted_form", "N/A")}\n")
                sb.append("Drift Value: ${drift.optDouble("drift_value", 0.0)}\n")
                sb.append("Drift Similarity: ${drift.optString("similarity", "N/A")}\n")
            }
            
            sb.append("\n-- Mutated Form --\n")
            val mutated = data?.optJSONObject("mutated_form")
            if (mutated != null) {
                sb.append("Mutated: ${mutated.optString("mutated", "N/A")}\n")
                sb.append("Mutation Factor: ${mutated.optDouble("mutation_factor", 0.0)}\n")
                sb.append("Tone Length: ${mutated.optInt("tone_length", 0)}\n")
            }
            
            sb.append("\n-- Symbolic Anchors --\n")
            val anchors = data?.optJSONArray("symbolic_anchors")
            if (anchors != null) {
                for (i in 0 until anchors.length()) {
                    sb.append("${anchors.optString(i)}\n")
                }
            }
            
            sb.append("\nSession: ${data?.optString("session", "N/A")}")
            sb.append("\nTimestamp: ${data?.optLong("timestamp", 0L)}")
            statusText.text = sb.toString()
        } else {
            statusText.text = "Error: ${result?.optString("error", "Unknown error")}"
        }
    }

    private fun setLoading(isLoading: Boolean) {
        loader.visibility = if (isLoading) View.VISIBLE else View.GONE
    }

    private fun enableButtons(enabled: Boolean) {
        sessionButton.isEnabled = enabled
        driftButton.isEnabled = enabled
        nodeButton.isEnabled = enabled
    }

    // This helper function would ideally check a stored session ID.
    // For this example, we assume the UI does not maintain it locally.
    private fun getCurrentSessionId(): String? {
        // In a full implementation, return the active session's ID.
        // Here we simply toggle based on the session button text.
        return if (sessionButton.text.toString().contains("End")) "active_session" else null
    }
}
