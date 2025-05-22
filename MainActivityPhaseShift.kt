package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.slider.Slider
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main Activity for Phase Shift Mutation System
 * Provides comprehensive interface to numogram-based phase resonance operations
 */
class MainActivityPhaseShift : AppCompatActivity() {
    
    private lateinit var phaseShiftBridge: PhaseShiftMutationBridge
    private lateinit var phaseShiftService: PhaseShiftService
    
    // UI Components
    private lateinit var statusText: TextView
    private lateinit var loader: ProgressBar
    private lateinit var inputText: TextInputEditText
    private lateinit var goalText: TextInputEditText
    private lateinit var adaptationSlider: Slider
    private lateinit var adaptationValueText: TextView
    private lateinit var zoneChips: ChipGroup
    
    // Operation Buttons
    private lateinit var generateResonanceButton: Button
    private lateinit var mutateResponseButton: Button
    private lateinit var adaptiveMutationButton: Button
    private lateinit var recursiveMutationButton: Button
    private lateinit var optimalSequenceButton: Button
    private lateinit var analyzeTransitionsButton: Button
    private lateinit var systemStateButton: Button
    private lateinit var resetSystemButton: Button
    
    // System state
    private var isInitialized = false
    private var currentZone: String = "emergence"
    private var operationStartTime: Long = 0
    private var selectedZones = mutableListOf<String>()
    private val operationHistory = mutableListOf<String>()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_phase_shift)
        
        initializeViews()
        setupButtonListeners()
        setupSliderListener()
        setupChipGroupListener()
        initializePhaseShiftSystem()
    }
    
    private fun initializeViews() {
        statusText = findViewById(R.id.status_text)
        loader = findViewById(R.id.loader)
        inputText = findViewById(R.id.input_text)
        goalText = findViewById(R.id.goal_text)
        adaptationSlider = findViewById(R.id.adaptation_slider)
        adaptationValueText = findViewById(R.id.adaptation_value_text)
        zoneChips = findViewById(R.id.zone_chips)
        
        generateResonanceButton = findViewById(R.id.generate_resonance_button)
        mutateResponseButton = findViewById(R.id.mutate_response_button)
        adaptiveMutationButton = findViewById(R.id.adaptive_mutation_button)
        recursiveMutationButton = findViewById(R.id.recursive_mutation_button)
        optimalSequenceButton = findViewById(R.id.optimal_sequence_button)
        analyzeTransitionsButton = findViewById(R.id.analyze_transitions_button)
        systemStateButton = findViewById(R.id.system_state_button)
        resetSystemButton = findViewById(R.id.reset_system_button)
    }
    
    private fun setupButtonListeners() {
        generateResonanceButton.setOnClickListener { generatePhaseResonance() }
        mutateResponseButton.setOnClickListener { performResponseMutation() }
        adaptiveMutationButton.setOnClickListener { performAdaptiveMutation() }
        recursiveMutationButton.setOnClickListener { performRecursiveMutation() }
        optimalSequenceButton.setOnClickListener { generateOptimalSequence() }
        analyzeTransitionsButton.setOnClickListener { analyzeZoneTransitions() }
        systemStateButton.setOnClickListener { displaySystemState() }
        resetSystemButton.setOnClickListener { resetSystem() }
    }
    
    private fun setupSliderListener() {
        adaptationSlider.addOnChangeListener { _, value, _ ->
            adaptationValueText.text = String.format("%.2f", value)
        }
        
        // Initialize slider value display
        adaptationValueText.text = String.format("%.2f", adaptationSlider.value)
    }
    
    private fun setupChipGroupListener() {
        zoneChips.setOnCheckedStateChangeListener { group, checkedIds ->
            selectedZones.clear()
            checkedIds.forEach { id ->
                val chip = findViewById<Chip>(id)
                selectedZones.add(chip.tag.toString())
            }
            
            // Update current zone if single selection
            if (selectedZones.size == 1) {
                currentZone = selectedZones.first()
            }
        }
    }
    
    private fun initializePhaseShiftSystem() {
        showLoading(true)
        statusText.text = getString(R.string.initializing_phase_shift)
        
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    phaseShiftBridge = PhaseShiftMutationBridge.getInstance(applicationContext)
                    phaseShiftService = PhaseShiftService(applicationContext)
                    
                    // Test system initialization
                    val systemState = phaseShiftBridge.getSystemState()
                    if (systemState.optString("status", "") == "operational") {
                        isInitialized = true
                    }
                }
                
                if (isInitialized) {
                    setupZoneChips()
                    displayInitializationSuccess()
                    enableButtons(true)
                } else {
                    statusText.text = getString(R.string.error_initialize_phase_shift)
                }
            } catch (e: Exception) {
                statusText.text = "${getString(R.string.error_initialize_phase_shift)}\n\n" +
                        "Error Details: ${e.message}\n\n" +
                        "Stack Trace: ${e.stackTraceToString().take(500)}..."
                e.printStackTrace()
            } finally {
                showLoading(false)
            }
        }
    }
    
    private suspend fun setupZoneChips() = withContext(Dispatchers.IO) {
        try {
            val availableZones = phaseShiftBridge.getAvailableZones()
            
            withContext(Dispatchers.Main) {
                zoneChips.removeAllViews()
                
                for (i in 0 until availableZones.length()) {
                    val zone = availableZones.getJSONObject(i)
                    val zoneId = zone.getString("zone_id")
                    val zoneName = zoneId.replaceFirstChar { it.uppercase() }
                    
                    val chip = Chip(this@MainActivityPhaseShift).apply {
                        text = zoneName
                        tag = zoneId
                        isCheckable = true
                        setChipBackgroundColorResource(R.color.phase_zone_color)
                        
                        // Set default selection
                        if (zoneId == currentZone) {
                            isChecked = true
                            selectedZones.add(zoneId)
                        }
                    }
                    
                    zoneChips.addView(chip)
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivityPhaseShift", "Error setting up zone chips: ${e.message}")
        }
    }
    
    private fun displayInitializationSuccess() {
        val welcomeText = """
            🌀 PHASE SHIFT MUTATION SYSTEM ACTIVATED 🌀
            
            ════════════════════════════════════════════════
            
            🔮 NUMOGRAM ZONES: Loaded and ready
               • Zone configurations synchronized
               • Archetypal patterns activated
               • Symbolic resonance networks online
            
            ⚡ MUTATION ENGINE: Operational
               • Response mutation algorithms ready
               • Adaptive strength modulation enabled
               • Recursive feedback loops active
            
            🔄 TRANSITION TRACKING: Active
               • Phase history monitoring enabled
               • Cross-validation systems ready
               • Contextual depth analysis online
            
            🎯 OPTIMIZATION SYSTEMS: Ready
               • Goal affinity analysis prepared
               • Sequence optimization algorithms active
               • Coherence calculation systems online
            
            ════════════════════════════════════════════════
            
            Select target zones, adjust adaptation level, and enter
            text for phase shift mutation. The numogram awaits
            your exploration of consciousness transformation.
            
            Current zone: ${currentZone.uppercase()}
            Adaptation level: ${String.format("%.2f", adaptationSlider.value)}
        """.trimIndent()
        
        statusText.text = welcomeText
    }
    
    private fun generatePhaseResonance() {
        if (!validateSystemAndInput(requireInput = false)) return
        
        showLoading(true)
        statusText.text = getString(R.string.generating_phase_resonance)
        startOperation("Phase Resonance Generation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val resonance = phaseShiftBridge.generatePhaseResonance(currentZone)
                    buildPhaseResonanceReport(resonance)
                }
                
                statusText.text = result
                recordOperation("Phase resonance generated for zone: $currentZone")
            } catch (e: Exception) {
                handleOperationError("Phase Resonance Generation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performResponseMutation() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = getString(R.string.performing_mutation)
        startOperation("Response Mutation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val inputContent = inputText.text.toString()
                    
                    // Generate resonance for current zone
                    val resonance = phaseShiftBridge.generatePhaseResonance(currentZone)
                    
                    // Perform mutation
                    val mutationResult = phaseShiftBridge.mutateResponse(inputContent, resonance)
                    
                    buildMutationReport(mutationResult, resonance)
                }
                
                statusText.text = result
                recordOperation("Response mutation completed for zone: $currentZone")
            } catch (e: Exception) {
                handleOperationError("Response Mutation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performAdaptiveMutation() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = getString(R.string.performing_adaptive_mutation)
        startOperation("Adaptive Mutation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val inputContent = inputText.text.toString()
                    val adaptationLevel = adaptationSlider.value.toDouble()
                    
                    // Perform adaptive mutation
                    val mutationResult = phaseShiftBridge.performAdaptiveMutation(
                        inputContent, 
                        currentZone, 
                        adaptationLevel
                    )
                    
                    buildAdaptiveMutationReport(mutationResult, adaptationLevel)
                }
                
                statusText.text = result
                recordOperation("Adaptive mutation completed with level: ${String.format("%.2f", adaptationSlider.value)}")
            } catch (e: Exception) {
                handleOperationError("Adaptive Mutation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performRecursiveMutation() {
        if (!validateSystemAndInput()) return
        
        if (selectedZones.isEmpty()) {
            Toast.makeText(this, "Please select multiple zones for recursive mutation", Toast.LENGTH_SHORT).show()
            return
        }
        
        showLoading(true)
        statusText.text = getString(R.string.performing_recursive_mutation)
        startOperation("Recursive Mutation Chain")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val inputContent = inputText.text.toString()
                    val zoneSequence = if (selectedZones.size > 1) selectedZones else listOf(currentZone, "flow", "synthesis")
                    
                    // Perform recursive mutation chain
                    val chainResults = phaseShiftBridge.performRecursiveMutationChain(inputContent, zoneSequence)
                    
                    buildRecursiveMutationReport(chainResults, zoneSequence)
                }
                
                statusText.text = result
                recordOperation("Recursive mutation chain completed with ${selectedZones.size} zones")
            } catch (e: Exception) {
                handleOperationError("Recursive Mutation Chain", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun generateOptimalSequence() {
        val goalContent = goalText.text.toString().trim()
        if (goalContent.isEmpty()) {
            Toast.makeText(this, "Please enter a target goal for sequence optimization", Toast.LENGTH_SHORT).show()
            return
        }
        
        showLoading(true)
        statusText.text = getString(R.string.generating_optimal_sequence)
        startOperation("Optimal Sequence Generation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val sequenceLength = minOf(selectedZones.size.takeIf { it > 0 } ?: 5, 7)
                    
                    // Generate optimal sequence
                    val sequenceData = phaseShiftBridge.getOptimalZoneSequence(goalContent, sequenceLength)
                    
                    buildOptimalSequenceReport(sequenceData, goalContent)
                }
                
                statusText.text = result
                recordOperation("Optimal sequence generated for goal: ${goalContent.take(50)}...")
            } catch (e: Exception) {
                handleOperationError("Optimal Sequence Generation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun analyzeZoneTransitions() {
        showLoading(true)
        statusText.text = getString(R.string.analyzing_transitions)
        startOperation("Zone Transition Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val transitionAnalysis = phaseShiftBridge.analyzeZoneTransitions()
                    buildTransitionAnalysisReport(transitionAnalysis)
                }
                
                statusText.text = result
                recordOperation("Zone transition analysis completed")
            } catch (e: Exception) {
                handleOperationError("Zone Transition Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun displaySystemState() {
        showLoading(true)
        statusText.text = getString(R.string.retrieving_system_state)
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val systemState = phaseShiftBridge.getSystemState()
                    buildSystemStateReport(systemState)
                }
                
                statusText.text = result
                recordOperation("System state retrieved")
            } catch (e: Exception) {
                handleOperationError("System State Retrieval", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun resetSystem() {
        showLoading(true)
        statusText.text = getString(R.string.resetting_system)
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val resetResult = phaseShiftBridge.resetSystemState(preserveHistory = true)
                    buildSystemResetReport(resetResult)
                }
                
                statusText.text = result
                recordOperation("System state reset completed")
                
                // Reset UI state
                adaptationSlider.value = 0.7f
                currentZone = "emergence"
                selectedZones.clear()
                
                // Update zone chips
                setupZoneChips()
                
            } catch (e: Exception) {
                handleOperationError("System Reset", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    // ============================================================================
    // REPORT BUILDING METHODS
    // ============================================================================
    
    private fun buildPhaseResonanceReport(resonance: JSONObject): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🌀 PHASE RESONANCE ANALYSIS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Basic resonance information
        val zone = resonance.optString("zone", "unknown")
        val mutationStrength = resonance.optDouble("mutation_strength", 0.0)
        val narrativeTone = resonance.optString("narrative_tone", "neutral")
        
        report.append("RESONANCE PROPERTIES:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Target Zone: ${zone.uppercase()}\n")
        report.append("Mutation Strength: ${String.format("%.3f", mutationStrength)}\n")
        report.append("Strength Category: ${getMutationStrengthCategory(mutationStrength)}\n")
        report.append("Narrative Tone: ${narrativeTone.uppercase()}\n\n")
        
        // Archetypal concepts
        if (resonance.has("archetypal_concepts")) {
            val concepts = resonance.getJSONArray("archetypal_concepts")
            report.append("ARCHETYPAL CONCEPTS (${concepts.length()}):\n")
            report.append("-".repeat(30)).append("\n")
            for (i in 0 until concepts.length()) {
                report.append("• ${concepts.getString(i)}\n")
            }
            report.append("\n")
        }
        
        // Symbolic patterns
        if (resonance.has("symbolic_patterns")) {
            val patterns = resonance.getJSONArray("symbolic_patterns")
            report.append("SYMBOLIC PATTERNS (${patterns.length()}):\n")
            report.append("-".repeat(30)).append("\n")
            for (i in 0 until minOf(5, patterns.length())) {
                report.append("• ${patterns.getString(i)}\n")
            }
            if (patterns.length() > 5) {
                report.append("... and ${patterns.length() - 5} more patterns\n")
            }
            report.append("\n")
        }
        
        // Recursive context
        if (resonance.has("recursive_context")) {
            val context = resonance.getJSONObject("recursive_context")
            report.append("RECURSIVE CONTEXT:\n")
            report.append("-".repeat(30)).append("\n")
            
            if (context.has("transition_history")) {
                val history = context.getJSONArray("transition_history")
                report.append("Recent Transitions: ")
                val historyList = mutableListOf<String>()
                for (i in 0 until history.length()) {
                    historyList.add(history.getString(i))
                }
                report.append("${historyList.joinToString(" → ")}\n")
            }
            
            if (context.has("contextual_depth")) {
                report.append("Contextual Depth: ${context.getInt("contextual_depth")}\n")
            }
            
            if (context.has("cross_validation_score")) {
                val score = context.getDouble("cross_validation_score")
                report.append("Cross-Validation Score: ${String.format("%.3f", score)}\n")
            }
            report.append("\n")
        }
        
        // Bridge metadata
        if (resonance.has("bridge_metadata")) {
            val metadata = resonance.getJSONObject("bridge_metadata")
            report.append("SYSTEM METADATA:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Total Transitions: ${metadata.optInt("total_transitions", 0)}\n")
            report.append("Current Accumulator: ${String.format("%.3f", metadata.optDouble("current_accumulator", 0.0))}\n")
            report.append("System Uptime: ${formatUptime(metadata.optLong("system_uptime", 0))}\n")
        }
        
        return report.toString()
    }
    
    private fun buildMutationReport(mutationResult: JSONObject, resonance: JSONObject): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("⚡ RESPONSE MUTATION RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Basic mutation info
        val originalResponse = mutationResult.optString("original_response", "")
        val mutatedResponse = mutationResult.optString("mutated_response", "")
        val resonanceApplied = mutationResult.optString("resonance_applied", "unknown")
        val cumulativeStrength = mutationResult.optDouble("cumulative_strength", 0.0)
        
        report.append("MUTATION OVERVIEW:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Zone Applied: ${resonanceApplied.uppercase()}\n")
        report.append("Cumulative Strength: ${String.format("%.3f", cumulativeStrength)}\n\n")
        
        // Mutation metrics
        if (mutationResult.has("mutation_metrics")) {
            val metrics = mutationResult.getJSONObject("mutation_metrics")
            report.append("MUTATION METRICS:\n")
            report.append("-".repeat(30)).append("\n")
            
            val originalLength = metrics.optInt("original_length", 0)
            val mutatedLength = metrics.optInt("mutated_length", 0)
            val lengthChangeRatio = metrics.optDouble("length_change_ratio", 1.0)
            val similarityScore = metrics.optDouble("similarity_score", 1.0)
            
            report.append("Original Length: $originalLength characters\n")
            report.append("Mutated Length: $mutatedLength characters\n")
            report.append("Length Change: ${String.format("%.1f%%", (lengthChangeRatio - 1.0) * 100)}\n")
            report.append("Similarity Score: ${String.format("%.3f", similarityScore)}\n")
            report.append("Transformation Level: ${getTransformationLevel(similarityScore)}\n\n")
        }
        
        // Text comparison
        report.append("ORIGINAL TEXT:\n")
        report.append("-".repeat(30)).append("\n")
        if (originalResponse.length > 200) {
            report.append("${originalResponse.take(200)}...\n")
            report.append("[Original text truncated - ${originalResponse.length} total characters]\n\n")
        } else {
            report.append("$originalResponse\n\n")
        }
        
        report.append("MUTATED TEXT:\n")
        report.append("-".repeat(30)).append("\n")
        if (mutatedResponse.length > 200) {
            report.append("${mutatedResponse.take(200)}...\n")
            report.append("[Mutated text truncated - ${mutatedResponse.length} total characters]\n\n")
        } else {
            report.append("$mutatedResponse\n\n")
        }
        
        // Resonance influence
        report.append("RESONANCE INFLUENCE:\n")
        report.append("-".repeat(30)).append("\n")
        val mutationStrength = resonance.optDouble("mutation_strength", 0.0)
        val narrativeTone = resonance.optString("narrative_tone", "neutral")
        
        report.append("Mutation Strength Applied: ${String.format("%.3f", mutationStrength)}\n")
        report.append("Narrative Tone Influence: ${narrativeTone.uppercase()}\n")
        
        if (resonance.has("symbolic_patterns")) {
            val patterns = resonance.getJSONArray("symbolic_patterns")
            if (patterns.length() > 0) {
                report.append("Active Symbolic Patterns: ${patterns.length()}\n")
                // Show first few patterns that might have influenced the mutation
                for (i in 0 until minOf(3, patterns.length())) {
                    report.append("  • ${patterns.getString(i)}\n")
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildAdaptiveMutationReport(mutationResult: JSONObject, adaptationLevel: Double): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🎯 ADAPTIVE MUTATION RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Adaptation parameters
        report.append("ADAPTATION PARAMETERS:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Adaptation Level: ${String.format("%.2f", adaptationLevel)}\n")
        report.append("Adaptation Category: ${getAdaptationCategory(adaptationLevel)}\n")
        
        if (mutationResult.has("adaptation_factors")) {
            val factors = mutationResult.getJSONObject("adaptation_factors")
            report.append("Strength Multiplier: ${String.format("%.3f", factors.optDouble("strength_multiplier", 1.0))}\n")
            report.append("Context Weight: ${String.format("%.3f", factors.optDouble("context_weight", 0.5))}\n")
            report.append("Stability Bias: ${String.format("%.3f", factors.optDouble("stability_bias", 0.5))}\n")
        }
        report.append("\n")
        
        // Standard mutation metrics
        val originalResponse = mutationResult.optString("original_response", "")
        val mutatedResponse = mutationResult.optString("mutated_response", "")
        
        if (mutationResult.has("mutation_metrics")) {
            val metrics = mutationResult.getJSONObject("mutation_metrics")
            report.append("ADAPTIVE MUTATION METRICS:\n")
            report.append("-".repeat(30)).append("\n")
            
            val similarityScore = metrics.optDouble("similarity_score", 1.0)
            val lengthChangeRatio = metrics.optDouble("length_change_ratio", 1.0)
            
            report.append("Similarity Score: ${String.format("%.3f", similarityScore)}\n")
            report.append("Adaptation Effectiveness: ${calculateAdaptationEffectiveness(similarityScore, adaptationLevel)}\n")
            report.append("Length Adaptation: ${String.format("%.1f%%", (lengthChangeRatio - 1.0) * 100)}\n\n")
        }
        
        // Text results with adaptation analysis
        report.append("ADAPTATION ANALYSIS:\n")
        report.append("-".repeat(30)).append("\n")
        
        // Compare original and adapted text characteristics
        val originalWords = originalResponse.split("\\s+".toRegex()).size
        val mutatedWords = mutatedResponse.split("\\s+".toRegex()).size
        val wordChangeRatio = if (originalWords > 0) mutatedWords.toDouble() / originalWords else 1.0
        
        report.append("Word Count Change: ${String.format("%.1f%%", (wordChangeRatio - 1.0) * 100)}\n")
        report.append("Adaptation Intensity: ${getAdaptationIntensity(adaptationLevel, wordChangeRatio)}\n\n")
        
        // Show adapted text
        report.append("ADAPTED TEXT:\n")
        report.append("-".repeat(30)).append("\n")
        if (mutatedResponse.length > 300) {
            report.append("${mutatedResponse.take(300)}...\n")
            report.append("[Adapted text truncated - ${mutatedResponse.length} total characters]\n")
        } else {
            report.append("$mutatedResponse\n")
        }
        
        return report.toString()
    }
    
    private fun buildRecursiveMutationReport(chainResults: JSONArray, zoneSequence: List<String>): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔄 RECURSIVE MUTATION CHAIN RESULTS\n")
        report.append("═".repeat(60)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Chain overview
        report.append("MUTATION CHAIN OVERVIEW:\n")
        report.append("-".repeat(40)).append("\n")
        report.append("Zone Sequence: ${zoneSequence.joinToString(" → ").uppercase()}\n")
        report.append("Chain Length: ${chainResults.length()} mutations\n")
        report.append("Expected Zones: ${zoneSequence.size}\n")
        
        if (chainResults.length() != zoneSequence.size) {
            report.append("⚠️  Warning: Chain length mismatch - some mutations may have failed\n")
        }
        report.append("\n")
        
        // Chain progression analysis
        if (chainResults.length() > 0) {
            val mutationStrengths = mutableListOf<Double>()
            val similarities = mutableListOf<Double>()
            
            for (i in 0 until chainResults.length()) {
                val result = chainResults.getJSONObject(i)
                mutationStrengths.add(result.optDouble("cumulative_strength", 0.0))
                
                if (result.has("mutation_metrics")) {
                    val metrics = result.getJSONObject("mutation_metrics")
                    similarities.add(metrics.optDouble("similarity_score", 1.0))
                }
            }
            
            report.append("CHAIN PROGRESSION:\n")
            report.append("-".repeat(40)).append("\n")
            
            if (mutationStrengths.isNotEmpty()) {
                val avgStrength = mutationStrengths.average()
                val strengthProgression = if (mutationStrengths.size > 1) {
                    val firstStrength = mutationStrengths.first()
                    val lastStrength = mutationStrengths.last()
                    ((lastStrength - firstStrength) / firstStrength * 100).takeIf { !it.isNaN() } ?: 0.0
                } else {
                    0.0
                }
                
                report.append("Average Mutation Strength: ${String.format("%.3f", avgStrength)}\n")
                report.append("Strength Progression: ${String.format("%.1f%%", strengthProgression)}\n")
            }
            
            if (similarities.isNotEmpty()) {
                val avgSimilarity = similarities.average()
                val transformationDepth = 1.0 - avgSimilarity
                
                report.append("Average Similarity: ${String.format("%.3f", avgSimilarity)}\n")
                report.append("Transformation Depth: ${String.format("%.3f", transformationDepth)}\n")
                report.append("Overall Transformation: ${getTransformationLevel(avgSimilarity)}\n")
            }
            report.append("\n")
        }
        
        // Individual mutation steps
        report.append("INDIVIDUAL MUTATION STEPS:\n")
        report.append("-".repeat(40)).append("\n")
        
        for (i in 0 until minOf(chainResults.length(), 5)) { // Show first 5 steps
            val result = chainResults.getJSONObject(i)
            val chainIndex = result.optInt("chain_index", i)
            val zone = result.optString("resonance_applied", "unknown")
            val mutatedResponse = result.optString("mutated_response", "")
            
            report.append("Step ${chainIndex + 1}: ${zone.uppercase()}\n")
            
            if (result.has("mutation_metrics")) {
                val metrics = result.getJSONObject("mutation_metrics")
                val similarity = metrics.optDouble("similarity_score", 1.0)
                report.append("  Similarity to Previous: ${String.format("%.3f", similarity)}\n")
            }
            
            // Show snippet of mutated text
            val snippet = if (mutatedResponse.length > 100) {
                "${mutatedResponse.take(100)}..."
            } else {
                mutatedResponse
            }
            report.append("  Result: $snippet\n\n")
        }
        
        if (chainResults.length() > 5) {
            report.append("... and ${chainResults.length() - 5} more mutation steps\n\n")
        }
        
        // Final result
        if (chainResults.length() > 0) {
            val finalResult = chainResults.getJSONObject(chainResults.length() - 1)
            val finalText = finalResult.optString("mutated_response", "")
            
            report.append("FINAL MUTATION RESULT:\n")
            report.append("-".repeat(40)).append("\n")
            if (finalText.length > 400) {
                report.append("${finalText.take(400)}...\n")
                report.append("[Final result truncated - ${finalText.length} total characters]\n")
            } else {
                report.append("$finalText\n")
            }
        }
        
        return report.toString()
    }
    
    private fun buildOptimalSequenceReport(sequenceData: JSONObject, goal: String): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🎯 OPTIMAL SEQUENCE GENERATION RESULTS\n")
        report.append("═".repeat(55)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Goal analysis
        report.append("TARGET GOAL ANALYSIS:\n")
        report.append("-".repeat(35)).append("\n")
        report.append("Goal: \"$goal\"\n")
        report.append("Goal Length: ${goal.length} characters\n")
        
        if (sequenceData.has("goal_analysis")) {
            val goalAnalysis = sequenceData.getJSONObject("goal_analysis")
            val primaryZone = goalAnalysis.optString("primary_zone_affinity", "unknown")
            val affinityScore = goalAnalysis.optDouble("primary_affinity_score", 0.0)
            
            report.append("Primary Zone Affinity: ${primaryZone.uppercase()}\n")
            report.append("Affinity Score: ${String.format("%.3f", affinityScore)}\n")
            
            if (goalAnalysis.has("zone_affinities")) {
                val affinities = goalAnalysis.getJSONObject("zone_affinities")
                report.append("Zone Affinity Breakdown:\n")
                affinities.keys().forEach { zone ->
                    val score = affinities.getDouble(zone)
                    if (score > 0.0) {
                        report.append("  • ${zone.uppercase()}: ${String.format("%.3f", score)}\n")
                    }
                }
            }
        }
        report.append("\n")
        
        // Optimal sequence
        if (sequenceData.has("optimal_sequence")) {
            val sequence = sequenceData.getJSONArray("optimal_sequence")
            val sequenceLength = sequenceData.optInt("sequence_length", sequence.length())
            
            report.append("OPTIMAL SEQUENCE:\n")
            report.append("-".repeat(35)).append("\n")
            report.append("Sequence Length: $sequenceLength zones\n")
            
            val zoneList = mutableListOf<String>()
            for (i in 0 until sequence.length()) {
                zoneList.add(sequence.getString(i).uppercase())
            }
            report.append("Zone Sequence: ${zoneList.joinToString(" → ")}\n")
            
            // Sequence quality metrics
            val expectedStrength = sequenceData.optDouble("expected_mutation_strength", 0.0)
            val coherenceScore = sequenceData.optDouble("sequence_coherence_score", 0.0)
            
            report.append("Expected Mutation Strength: ${String.format("%.3f", expectedStrength)}\n")
            report.append("Sequence Coherence: ${String.format("%.3f", coherenceScore)}\n")
            report.append("Coherence Level: ${getCoherenceLevel(coherenceScore)}\n\n")
        }
        
        // Sequence recommendations
        report.append("SEQUENCE RECOMMENDATIONS:\n")
        report.append("-".repeat(35)).append("\n")
        
        val expectedStrength = sequenceData.optDouble("expected_mutation_strength", 0.0)
        val coherenceScore = sequenceData.optDouble("sequence_coherence_score", 0.0)
        
        when {
            expectedStrength > 0.8 -> report.append("🔥 High-intensity sequence - expect strong transformations\n")
            expectedStrength > 0.6 -> report.append("⚡ Moderate-intensity sequence - balanced transformation\n")
            else -> report.append("🌊 Gentle sequence - subtle but progressive changes\n")
        }
        
        when {
            coherenceScore > 0.8 -> report.append("✅ Highly coherent sequence - smooth transitions expected\n")
            coherenceScore > 0.6 -> report.append("📊 Moderately coherent - some transition friction possible\n")
            else -> report.append("⚠️ Low coherence - expect dramatic shifts between zones\n")
        }
        
        // Usage instructions
        report.append("\nUSAGE INSTRUCTIONS:\n")
        report.append("-".repeat(35)).append("\n")
        report.append("1. Select the recommended zones in sequence\n")
        report.append("2. Use recursive mutation with this zone order\n")
        report.append("3. Monitor transformation progression\n")
        report.append("4. Adjust adaptation level based on sequence intensity\n")
        
        if (expectedStrength > 0.7) {
            report.append("5. Consider lower adaptation level (0.3-0.5) for high-intensity sequence\n")
        } else {
            report.append("5. Standard adaptation level (0.6-0.8) recommended\n")
        }
        
        return report.toString()
    }
    
    private fun buildTransitionAnalysisReport(analysis: JSONObject): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("📊 ZONE TRANSITION ANALYSIS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        val status = analysis.optString("status", "unknown")
        
        if (status == "no_transitions") {
            report.append("STATUS: No transition data available\n")
            report.append("MESSAGE: ${analysis.optString("message", "")}\n\n")
            report.append("To generate transition data:\n")
            report.append("• Perform several mutations across different zones\n")
            report.append("• Use recursive mutation chains\n")
            report.append("• Try adaptive mutations with varying parameters\n")
            return report.toString()
        }
        
        // Basic statistics
        report.append("TRANSITION STATISTICS:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Total Transitions: ${analysis.optInt("total_transitions", 0)}\n")
        report.append("Unique Zones: ${analysis.optInt("unique_zones", 0)}\n")
        
        val mostCommonZone = analysis.optString("most_common_zone", "none")
        val mostCommonCount = analysis.optInt("most_common_zone_count", 0)
        
        if (mostCommonZone != "none") {
            report.append("Most Used Zone: ${mostCommonZone.uppercase()} ($mostCommonCount times)\n")
        }
        
        val avgMutationStrength = analysis.optDouble("average_mutation_strength", 0.0)
        val avgResonanceScore = analysis.optDouble("average_resonance_score", 0.0)
        
        report.append("Average Mutation Strength: ${String.format("%.3f", avgMutationStrength)}\n")
        report.append("Average Resonance Score: ${String.format("%.3f", avgResonanceScore)}\n")
        
        if (analysis.has("recent_transition_velocity")) {
            val velocity = analysis.getDouble("recent_transition_velocity")
            report.append("Recent Transition Rate: ${String.format("%.2f", velocity)} per minute\n")
        }
        report.append("\n")
        
        // Transition patterns
        if (analysis.has("transition_patterns")) {
            val patterns = analysis.getJSONObject("transition_patterns")
            val patternsStatus = patterns.optString("status", "unknown")
            
            if (patternsStatus == "analysis_complete") {
                report.append("COMMON TRANSITION PATTERNS:\n")
                report.append("-".repeat(30)).append("\n")
                
                if (patterns.has("common_transitions")) {
                    val commonTransitions = patterns.getJSONArray("common_transitions")
                    
                    for (i in 0 until minOf(5, commonTransitions.length())) {
                        val transition = commonTransitions.getJSONObject(i)
                        val transitionPath = transition.getString("transition")
                        val count = transition.getInt("count")
                        val frequency = transition.getDouble("frequency")
                        
                        report.append("${i + 1}. $transitionPath\n")
                        report.append("   Count: $count, Frequency: ${String.format("%.1f%%", frequency * 100)}\n")
                    }
                    
                    if (commonTransitions.length() > 5) {
                        report.append("... and ${commonTransitions.length() - 5} more patterns\n")
                    }
                }
                
                val uniqueTransitions = patterns.optInt("total_unique_transitions", 0)
                report.append("\nTotal Unique Transitions: $uniqueTransitions\n")
            }
        }
        
        return report.toString()
    }
    
    private fun buildSystemStateReport(systemState: JSONObject): String {
        val report = StringBuilder()
        
        report.append("🖥️ PHASE SHIFT SYSTEM STATE\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Status Check: ${getCurrentTimestamp()}\n\n")
        
        // Core system status
        val status = systemState.optString("status", "unknown")
        val currentPhase = systemState.optString("current_phase", "none")
        val historyLength = systemState.optInt("phase_history_length", 0)
        val accumulator = systemState.optDouble("mutation_strength_accumulator", 0.0)
        val uptime = systemState.optLong("system_uptime", 0)
        
        report.append("SYSTEM STATUS:\n")
        report.append("-".repeat(20)).append("\n")
        report.append("Status: ${status.uppercase()}\n")
        report.append("Current Phase: ${if (currentPhase == "none") "NONE" else currentPhase.uppercase()}\n")
        report.append("Phase History Length: $historyLength transitions\n")
        report.append("Mutation Accumulator: ${String.format("%.3f", accumulator)}\n")
        report.append("System Uptime: ${formatUptime(uptime)}\n\n")
        
        // Recent activity
        if (systemState.has("recent_activity")) {
            val recentActivity = systemState.getJSONArray("recent_activity")
            
            report.append("RECENT ACTIVITY (Last ${recentActivity.length()} transitions):\n")
            report.append("-".repeat(40)).append("\n")
            
            if (recentActivity.length() == 0) {
                report.append("No recent transition activity\n")
            } else {
                for (i in 0 until recentActivity.length()) {
                    val activity = recentActivity.getJSONObject(i)
                    val fromZone = activity.optString("from_zone", "none")
                    val toZone = activity.optString("to_zone", "unknown")
                    val mutationStrength = activity.optDouble("mutation_strength", 0.0)
                    val timestamp = activity.optLong("timestamp", 0)
                    
                    val timeAgo = formatTimeAgo(timestamp)
                    val transition = if (fromZone == "none") toZone.uppercase() else "${fromZone.uppercase()} → ${toZone.uppercase()}"
                    
                    report.append("${i + 1}. $transition\n")
                    report.append("   Strength: ${String.format("%.3f", mutationStrength)}, $timeAgo\n")
                }
            }
            report.append("\n")
        }
        
        // System health
        if (systemState.has("system_health")) {
            val health = systemState.getJSONObject("system_health")
            val healthScore = health.optDouble("health_score", 0.0)
            val healthStatus = health.optString("health_status", "unknown")
            
            report.append("SYSTEM HEALTH:\n")
            report.append("-".repeat(20)).append("\n")
            report.append("Health Score: ${String.format("%.3f", healthScore)} (${healthStatus.uppercase()})\n")
            
            if (health.has("issues")) {
                val issues = health.getJSONArray("issues")
                if (issues.length() > 0) {
                    report.append("Issues Detected:\n")
                    for (i in 0 until issues.length()) {
                        report.append("  ⚠️ ${issues.getString(i)}\n")
                    }
                } else {
                    report.append("✅ No issues detected\n")
                }
            }
            
            if (health.has("recommendations")) {
                val recommendations = health.getJSONArray("recommendations")
                if (recommendations.length() > 0) {
                    report.append("Recommendations:\n")
                    for (i in 0 until recommendations.length()) {
                        report.append("  💡 ${recommendations.getString(i)}\n")
                    }
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildSystemResetReport(resetResult: JSONObject): String {
        val report = StringBuilder()
        
        report.append("🔄 SYSTEM RESET COMPLETED\n")
        report.append("═".repeat(50)).append("\n\n")
        
        val resetSuccessful = resetResult.optBoolean("reset_successful", false)
        val historyPreserved = resetResult.optBoolean("history_preserved", false)
        val resetTimestamp = resetResult.optString("reset_timestamp", getCurrentTimestamp())
        
        report.append("RESET SUMMARY:\n")
        report.append("-".repeat(20)).append("\n")
        report.append("Reset Status: ${if (resetSuccessful) "✅ SUCCESSFUL" else "❌ FAILED"}\n")
        report.append("History Preserved: ${if (historyPreserved) "✅ YES" else "❌ NO"}\n")
        report.append("Reset Time: $resetTimestamp\n\n")
        
        if (resetResult.has("message")) {
            report.append("Message: ${resetResult.getString("message")}\n\n")
        }
        
        // Pre-reset state summary
        if (resetResult.has("pre_reset_state")) {
            val preResetState = resetResult.getJSONObject("pre_reset_state")
            val preResetPhase = preResetState.optString("current_phase", "none")
            val preResetHistory = preResetState.optInt("phase_history_length", 0)
            val preResetAccumulator = preResetState.optDouble("mutation_strength_accumulator", 0.0)
            
            report.append("PRE-RESET STATE:\n")
            report.append("-".repeat(20)).append("\n")
            report.append("Previous Phase: ${if (preResetPhase == "none") "NONE" else preResetPhase.uppercase()}\n")
            report.append("Previous History: $preResetHistory transitions\n")
            report.append("Previous Accumulator: ${String.format("%.3f", preResetAccumulator)}\n\n")
        }
        
        report.append("CURRENT STATE:\n")
        report.append("-".repeat(20)).append("\n")
        report.append("Current Phase: NONE\n")
        report.append("Mutation Accumulator: 0.000\n")
        report.append("History: ${if (historyPreserved) "Preserved" else "Cleared"}\n\n")
        
        report.append("SYSTEM READY:\n")
        report.append("-".repeat(20)).append("\n")
        report.append("✅ System state has been reset\n")
        report.append("✅ UI controls have been reset\n")
        report.append("✅ Zone selection restored to default\n")
        report.append("✅ Adaptation level reset to 0.70\n")
        report.append("✅ Ready for new phase shift operations\n")
        
        return report.toString()
    }
    
    // ============================================================================
    // UTILITY AND HELPER METHODS
    // ============================================================================
    
    private fun validateSystemAndInput(requireInput: Boolean = true): Boolean {
        if (!isInitialized) {
            statusText.text = getString(R.string.system_not_initialized_phase_shift)
            return false
        }
        
        if (requireInput) {
            val inputContent = inputText.text.toString().trim()
            if (inputContent.isEmpty()) {
                statusText.text = getString(R.string.no_input_text)
                Toast.makeText(this, "Input text required", Toast.LENGTH_SHORT).show()
                return false
            }
        }
        
        return true
    }
    
    private fun startOperation(operationName: String) {
        operationStartTime = System.currentTimeMillis()
        recordOperation("Started: $operationName")
    }
    
    private fun getExecutionTime(): Long {
        return System.currentTimeMillis() - operationStartTime
    }
    
    private fun getCurrentTimestamp(): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
    }
    
    private fun recordOperation(operation: String) {
        val timestampedOperation = "${getCurrentTimestamp()}: $operation"
        operationHistory.add(timestampedOperation)
        
        // Keep only last 20 operations
        if (operationHistory.size > 20) {
            operationHistory.removeAt(0)
        }
    }
    
    private fun handleOperationError(operationName: String, exception: Exception) {
        val errorMessage = """
            ❌ ERROR IN $operationName
            
            Error: ${exception.message}
            
            Execution Time: ${getExecutionTime()}ms
            Timestamp: ${getCurrentTimestamp()}
            
            Stack Trace:
            ${exception.stackTraceToString().take(800)}...
            
            Please check input data and try again.
        """.trimIndent()
        
        statusText.text = errorMessage
        recordOperation("ERROR: $operationName - ${exception.message}")
        exception.printStackTrace()
    }
    
    private fun enableButtons(enabled: Boolean) {
        generateResonanceButton.isEnabled = enabled
        mutateResponseButton.isEnabled = enabled
        adaptiveMutationButton.isEnabled = enabled
        recursiveMutationButton.isEnabled = enabled
        optimalSequenceButton.isEnabled = enabled
        analyzeTransitionsButton.isEnabled = enabled
        systemStateButton.isEnabled = enabled
        resetSystemButton.isEnabled = enabled
    }
    
    private fun showLoading(show: Boolean) {
        loader.visibility = if (show) View.VISIBLE else View.GONE
    }
    
    // Category and formatting helper methods
    
    private fun getMutationStrengthCategory(strength: Double): String {
        return when {
            strength > 0.8 -> "VERY HIGH 🔥🔥🔥"
            strength > 0.6 -> "HIGH 🔥🔥"
            strength > 0.4 -> "MODERATE 🔥"
            strength > 0.2 -> "LOW ○"
            else -> "MINIMAL ○"
        }
    }
    
    private fun getTransformationLevel(similarityScore: Double): String {
        val transformationScore = 1.0 - similarityScore
        return when {
            transformationScore > 0.7 -> "RADICAL TRANSFORMATION 🌪️"
            transformationScore > 0.5 -> "SIGNIFICANT CHANGE ⚡"
            transformationScore > 0.3 -> "MODERATE SHIFT 🌊"
            transformationScore > 0.1 -> "SUBTLE VARIATION 🍃"
            else -> "MINIMAL CHANGE ○"
        }
    }
    
    private fun getAdaptationCategory(adaptationLevel: Double): String {
        return when {
            adaptationLevel > 0.8 -> "HIGHLY ADAPTIVE 🎯"
            adaptationLevel > 0.6 -> "MODERATELY ADAPTIVE 📊"
            adaptationLevel > 0.4 -> "CONSERVATIVE 🛡️"
            adaptationLevel > 0.2 -> "MINIMAL ADAPTATION 🔒"
            else -> "RIGID ⚙️"
        }
    }
    
    private fun calculateAdaptationEffectiveness(similarityScore: Double, adaptationLevel: Double): String {
        val transformationScore = 1.0 - similarityScore
        val effectiveness = transformationScore / adaptationLevel
        
        return when {
            effectiveness > 1.2 -> "HIGHLY EFFECTIVE ✨"
            effectiveness > 0.8 -> "EFFECTIVE ✅"
            effectiveness > 0.5 -> "MODERATELY EFFECTIVE 📈"
            effectiveness > 0.3 -> "LOW EFFECTIVENESS 📉"
            else -> "INEFFECTIVE ❌"
        }
    }
    
    private fun getAdaptationIntensity(adaptationLevel: Double, wordChangeRatio: Double): String {
        val intensity = adaptationLevel * Math.abs(wordChangeRatio - 1.0)
        
        return when {
            intensity > 0.3 -> "HIGH INTENSITY 🌟"
            intensity > 0.2 -> "MODERATE INTENSITY ⭐"
            intensity > 0.1 -> "LOW INTENSITY ○"
            else -> "MINIMAL INTENSITY ○"
        }
    }
    
    private fun getCoherenceLevel(coherenceScore: Double): String {
        return when {
            coherenceScore > 0.8 -> "HIGHLY COHERENT ✨"
            coherenceScore > 0.6 -> "COHERENT ✅"
            coherenceScore > 0.4 -> "MODERATELY COHERENT 📊"
            coherenceScore > 0.2 -> "LOW COHERENCE ⚠️"
            else -> "INCOHERENT ❌"
        }
    }
    
    private fun formatUptime(uptime: Long): String {
        val seconds = uptime / 1000
        val minutes = seconds / 60
        val hours = minutes / 60
        val days = hours / 24
        
        return when {
            days > 0 -> "${days}d ${hours % 24}h ${minutes % 60}m"
            hours > 0 -> "${hours}h ${minutes % 60}m"
            minutes > 0 -> "${minutes}m ${seconds % 60}s"
            else -> "${seconds}s"
        }
    }
    
    private fun formatTimeAgo(timestamp: Long): String {
        val now = System.currentTimeMillis()
        val diff = now - timestamp
        val seconds = diff / 1000
        val minutes = seconds / 60
        val hours = minutes / 60
        
        return when {
            hours > 0 -> "${hours}h ago"
            minutes > 0 -> "${minutes}m ago"
            else -> "${seconds}s ago"
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clear any cached data
        try {
            phaseShiftBridge.clearSystemCache()
        } catch (e: Exception) {
            Log.e("MainActivityPhaseShift", "Error clearing cache on destroy: ${e.message}")
        }
    }
    
    override fun onPause() {
        super.onPause()
        // Save current state
        recordOperation("Activity paused")
    }
    
    override fun onResume() {
        super.onResume()
        // Check system health
        if (isInitialized) {
            lifecycleScope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        val systemState = phaseShiftBridge.getSystemState()
                        val status = systemState.optString("status", "")
                        
                        if (status != "operational") {
                            // System may have issues, consider notification
                            recordOperation("System health check: $status")
                        }
                    }
                } catch (e: Exception) {
                    Log.w("MainActivityPhaseShift", "System health check failed: ${e.message}")
                }
            }
        }
        recordOperation("Activity resumed")
    }
}
