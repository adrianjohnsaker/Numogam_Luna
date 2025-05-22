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
 * Main Activity for the Arcane Knowledge System
 * Provides comprehensive interface to esoteric and hermetic knowledge operations
 */
class MainActivityArcane : AppCompatActivity() {
    
    private lateinit var arcaneKnowledgeBridge: ArcaneKnowledgeBridge
    private lateinit var arcaneKnowledgeService: ArcaneKnowledgeService
    
    // UI Components
    private lateinit var statusText: TextView
    private lateinit var loader: ProgressBar
    private lateinit var inputText: TextInputEditText
    private lateinit var operationChips: ChipGroup
    
    // Operation Buttons
    private lateinit var numericalResonanceButton: Button
    private lateinit var liminalNarrativeButton: Button
    private lateinit var excessDynamicsButton: Button
    private lateinit var correspondenceNetworkButton: Button
    private lateinit var symbolResonanceButton: Button
    private lateinit var comprehensiveAnalysisButton: Button
    private lateinit var systemStatusButton: Button
    private lateinit var generateSampleButton: Button
    
    // System state
    private var isInitialized = false
    private var currentNarrativeElement: JSONObject? = null
    private var operationStartTime: Long = 0
    private var operationHistory = mutableListOf<String>()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_arcane_knowledge)
        
        initializeViews()
        setupButtonListeners()
        setupChipGroupListener()
        initializeArcaneSystem()
    }
    
    private fun initializeViews() {
        statusText = findViewById(R.id.status_text)
        loader = findViewById(R.id.loader)
        inputText = findViewById(R.id.input_text)
        operationChips = findViewById(R.id.operation_chips)
        
        numericalResonanceButton = findViewById(R.id.numerical_resonance_button)
        liminalNarrativeButton = findViewById(R.id.liminal_narrative_button)
        excessDynamicsButton = findViewById(R.id.excess_dynamics_button)
        correspondenceNetworkButton = findViewById(R.id.correspondence_network_button)
        symbolResonanceButton = findViewById(R.id.symbol_resonance_button)
        comprehensiveAnalysisButton = findViewById(R.id.comprehensive_analysis_button)
        systemStatusButton = findViewById(R.id.system_status_button)
        generateSampleButton = findViewById(R.id.generate_sample_button)
    }
    
    private fun setupButtonListeners() {
        numericalResonanceButton.setOnClickListener { performNumericalResonanceAnalysis() }
        liminalNarrativeButton.setOnClickListener { performLiminalNarrativeGeneration() }
        excessDynamicsButton.setOnClickListener { performExcessDynamicsAnalysis() }
        correspondenceNetworkButton.setOnClickListener { performCorrespondenceNetworkGeneration() }
        symbolResonanceButton.setOnClickListener { performSymbolResonanceCalculation() }
        comprehensiveAnalysisButton.setOnClickListener { performComprehensiveAnalysis() }
        systemStatusButton.setOnClickListener { displaySystemStatus() }
        generateSampleButton.setOnClickListener { generateSampleNarrative() }
    }
    
    private fun setupChipGroupListener() {
        // Add operation type chips
        val operations = listOf(
            "Numerical" to "numerical",
            "Liminal" to "liminal", 
            "Excess" to "excess",
            "Hermetic" to "hermetic",
            "Comprehensive" to "comprehensive"
        )
        
        operations.forEach { (display, value) ->
            val chip = Chip(this).apply {
                text = display
                tag = value
                isCheckable = true
            }
            operationChips.addView(chip)
        }
        
        operationChips.setOnCheckedStateChangeListener { group, checkedIds ->
            if (checkedIds.isNotEmpty()) {
                val selectedChip = findViewById<Chip>(checkedIds.first())
                val operation = selectedChip.tag.toString()
                highlightRelevantOperations(operation)
            }
        }
    }
    
    private fun initializeArcaneSystem() {
        showLoading(true)
        statusText.text = "Initializing Arcane Knowledge System..."
        
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    arcaneKnowledgeBridge = ArcaneKnowledgeBridge.getInstance(applicationContext)
                    arcaneKnowledgeService = ArcaneKnowledgeService(applicationContext)
                    
                    // Test system initialization
                    val systemStatus = arcaneKnowledgeBridge.getSystemStatus()
                    if (systemStatus.optBoolean("system_initialized", false)) {
                        isInitialized = true
                    }
                }
                
                if (isInitialized) {
                    displayInitializationSuccess()
                    enableButtons(true)
                } else {
                    statusText.text = "❌ Failed to initialize Arcane Knowledge System"
                }
            } catch (e: Exception) {
                statusText.text = "❌ Initialization Error: ${e.message}\n\n" +
                        "Stack Trace:\n${e.stackTraceToString().take(500)}..."
                e.printStackTrace()
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun displayInitializationSuccess() {
        val welcomeText = """
            🔮 ARCANE KNOWLEDGE SYSTEM INITIALIZED 🔮
            
            ═══════════════════════════════════════════════
            
            🔢 NUMERICAL CORRESPONDENCES: Ready
               • Numogram zones 1-10 active
               • Digital root calculations enabled
               • Resonance pattern generation online
            
            🌙 LIMINAL THRESHOLD MECHANICS: Ready
               • Ontological boundaries mapped
               • Sacred-profane thresholds active
               • Guardian challenge systems enabled
            
            ⚡ BATAILLEAN EXCESS DYNAMICS: Ready
               • Accumulation-expenditure cycles active
               • Transgression detection enabled
               • Glorious expenditure simulation ready
            
            🜔 HERMETIC SYMBOL NETWORK: Ready
               • Mercury, Sulfur, Salt correspondences loaded
               • Ouroboros transformation patterns active
               • Cross-domain symbol mapping enabled
            
            ═══════════════════════════════════════════════
            
            Enter narrative text or generate sample to begin
            arcane analysis and esoteric transformation.
            
            The veil between worlds grows thin...
        """.trimIndent()
        
        statusText.text = welcomeText
    }
    
    private fun performNumericalResonanceAnalysis() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔢 Calculating numerical resonances and digital root correspondences..."
        startOperation("Numerical Resonance Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    // Get input narrative
                    val narrative = getCurrentNarrativeElement()
                    
                    // Apply numerical resonance
                    val resonanceResult = arcaneKnowledgeBridge.applyNumericalResonance(narrative)
                    
                    // Get specific resonance data
                    val numericValues = extractNumericValues(narrative)
                    val resonanceAnalysis = mutableMapOf<Int, JSONObject>()
                    
                    numericValues.forEach { value ->
                        val resonance = arcaneKnowledgeBridge.calculateNumericalResonance(value)
                        resonanceAnalysis[value] = resonance
                    }
                    
                    buildNumericalResonanceReport(resonanceResult, resonanceAnalysis)
                }
                
                statusText.text = result
                recordOperation("Numerical Resonance Analysis completed successfully")
            } catch (e: Exception) {
                handleOperationError("Numerical Resonance Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performLiminalNarrativeGeneration() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🌙 Generating liminal threshold narrative and boundary dynamics..."
        startOperation("Liminal Narrative Generation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    // Extract boundary concepts from input
                    val inputText = inputText.text.toString()
                    val boundaryConcepts = extractBoundaryConcepts(inputText)
                    
                    // Generate liminal narrative
                    val narrative = arcaneKnowledgeBridge.generateLiminalNarrative(boundaryConcepts)
                    
                    // Get threshold data
                    val thresholds = arcaneKnowledgeBridge.getAllThresholds()
                    
                    buildLiminalNarrativeReport(narrative, thresholds, boundaryConcepts)
                }
                
                statusText.text = result
                recordOperation("Liminal Narrative Generation completed successfully")
            } catch (e: Exception) {
                handleOperationError("Liminal Narrative Generation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performExcessDynamicsAnalysis() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "⚡ Analyzing Bataillean excess dynamics and expenditure patterns..."
        startOperation("Excess Dynamics Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val narrative = getCurrentNarrativeElement()
                    
                    // Calculate narrative surplus
                    val surplus = arcaneKnowledgeBridge.calculateNarrativeSurplus(narrative)
                    
                    // Apply excess dynamics
                    val dynamics = arcaneKnowledgeBridge.applyExcessDynamics(narrative, 0.15)
                    
                    // Simulate excess cycle
                    val cycle = arcaneKnowledgeBridge.simulateExcessCycle(7)
                    
                    // Get current system state
                    val state = arcaneKnowledgeBridge.getExcessDynamicsState()
                    
                    buildExcessDynamicsReport(surplus, dynamics, cycle, state)
                }
                
                statusText.text = result
                recordOperation("Excess Dynamics Analysis completed successfully")
            } catch (e: Exception) {
                handleOperationError("Excess Dynamics Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performCorrespondenceNetworkGeneration() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🜔 Generating hermetic correspondence networks and symbol relationships..."
        startOperation("Correspondence Network Generation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    // Extract hermetic symbols from input
                    val inputText = inputText.text.toString()
                    val symbols = extractHermeticSymbols(inputText)
                    
                    val networks = mutableMapOf<String, JSONObject>()
                    
                    // Generate network for each symbol (limit to 3 to avoid overload)
                    symbols.take(3).forEach { symbol ->
                        val network = arcaneKnowledgeBridge.generateCorrespondenceNetwork(symbol)
                        networks[symbol] = network
                    }
                    
                    // Get all available symbols
                    val allSymbols = arcaneKnowledgeBridge.getAllHermeticSymbols()
                    
                    buildCorrespondenceNetworkReport(networks, allSymbols, symbols)
                }
                
                statusText.text = result
                recordOperation("Correspondence Network Generation completed successfully")
            } catch (e: Exception) {
                handleOperationError("Correspondence Network Generation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performSymbolResonanceCalculation() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔯 Calculating hermetic symbol resonances and transformation pathways..."
        startOperation("Symbol Resonance Calculation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    // Extract symbols from input
                    val inputText = inputText.text.toString()
                    val symbols = extractHermeticSymbols(inputText)
                    
                    val resonances = mutableListOf<JSONObject>()
                    val pathways = mutableListOf<JSONObject>()
                    
                    // Calculate resonances between symbol pairs
                    for (i in symbols.indices) {
                        for (j in i + 1 until symbols.size) {
                            val symbol1 = symbols[i]
                            val symbol2 = symbols[j]
                            
                            val resonance = arcaneKnowledgeBridge.calculateSymbolResonance(symbol1, symbol2)
                            resonances.add(resonance)
                            
                            // Get transformation pathways
                            val pathway1to2 = arcaneKnowledgeBridge.getTransformationPathway(symbol1, symbol2)
                            val pathway2to1 = arcaneKnowledgeBridge.getTransformationPathway(symbol2, symbol1)
                            
                            if (!pathway1to2.has("result") || pathway1to2.getString("result") != "no_pathway") {
                                pathways.add(pathway1to2)
                            }
                            if (!pathway2to1.has("result") || pathway2to1.getString("result") != "no_pathway") {
                                pathways.add(pathway2to1)
                            }
                        }
                    }
                    
                    buildSymbolResonanceReport(resonances, pathways, symbols)
                }
                
                statusText.text = result
                recordOperation("Symbol Resonance Calculation completed successfully")
            } catch (e: Exception) {
                handleOperationError("Symbol Resonance Calculation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performComprehensiveAnalysis() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🌟 Performing comprehensive arcane analysis across all knowledge systems..."
        startOperation("Comprehensive Arcane Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val narrative = getCurrentNarrativeElement()
                    
                    // Perform comprehensive analysis
                    val comprehensiveResult = arcaneKnowledgeBridge.performComprehensiveAnalysis(narrative)
                    
                    // Get recommendations
                    val recommendations = arcaneKnowledgeBridge.getRecommendedOperations(narrative)
                    
                    // Get system status
                    val systemStatus = arcaneKnowledgeBridge.getSystemStatus()
                    
                    buildComprehensiveAnalysisReport(comprehensiveResult, recommendations, systemStatus)
                }
                
                statusText.text = result
                recordOperation("Comprehensive Arcane Analysis completed successfully")
            } catch (e: Exception) {
                handleOperationError("Comprehensive Arcane Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun displaySystemStatus() {
        showLoading(true)
        statusText.text = "📊 Retrieving system status and diagnostics..."
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val systemStatus = arcaneKnowledgeBridge.getSystemStatus()
                    val dynamicsState = arcaneKnowledgeBridge.getExcessDynamicsState()
                    
                    buildSystemStatusReport(systemStatus, dynamicsState)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleOperationError("System Status Check", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun generateSampleNarrative() {
        showLoading(true)
        statusText.text = "📜 Generating sample arcane narrative for analysis..."
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val sampleNarrative = arcaneKnowledgeBridge.generateSampleNarrativeElement()
                    currentNarrativeElement = sampleNarrative
                    
                    // Update input field with sample content
                    val content = sampleNarrative.optString("content", "")
                    
                    buildSampleNarrativeReport(sampleNarrative, content)
                }
                
                // Update UI on main thread
                val content = currentNarrativeElement?.optString("content", "") ?: ""
                inputText.setText(content)
                
                statusText.text = result
                recordOperation("Sample narrative generated")
            } catch (e: Exception) {
                handleOperationError("Sample Generation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    // ============================================================================
    // REPORT BUILDING METHODS
    // ============================================================================
    
    private fun buildNumericalResonanceReport(
        resonanceResult: JSONObject,
        resonanceAnalysis: Map<Int, JSONObject>
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔢 NUMERICAL RESONANCE ANALYSIS RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Primary resonance data
        if (resonanceResult.has("numerical_resonance")) {
            val numericalResonance = resonanceResult.getJSONObject("numerical_resonance")
            val primaryNumber = numericalResonance.optInt("primary_number", 0)
            val resonanceStrength = numericalResonance.optDouble("resonance_strength", 0.0)
            
            report.append("PRIMARY RESONANCE:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Primary Number: $primaryNumber\n")
            report.append("Resonance Strength: ${String.format("%.3f", resonanceStrength)}\n")
            report.append("Strength Category: ${getResonanceCategory(resonanceStrength)}\n\n")
        }
        
        // Individual number analysis
        if (resonanceAnalysis.isNotEmpty()) {
            report.append("INDIVIDUAL NUMBER ANALYSIS:\n")
            report.append("-".repeat(30)).append("\n")
            
            resonanceAnalysis.forEach { (value, analysis) ->
                val digitalRoot = analysis.optInt("digital_root", 0)
                val name = analysis.optString("name", "Unknown")
                
                report.append("Number $value (Digital Root: $digitalRoot):\n")
                report.append("  • Zone: $name\n")
                
                if (analysis.has("properties")) {
                    val properties = analysis.getJSONObject("properties")
                    properties.keys().forEach { key ->
                        report.append("  • ${key.capitalize()}: ${properties.getString(key)}\n")
                    }
                }
                
                if (analysis.has("correspondences")) {
                    val correspondences = analysis.getJSONObject("correspondences")
                    correspondences.keys().forEach { domain ->
                        val values = correspondences.getJSONArray(domain)
                        val valuesList = mutableListOf<String>()
                        for (i in 0 until values.length()) {
                            valuesList.add(values.getString(i))
                        }
                        report.append("  • ${domain.capitalize()}: ${valuesList.joinToString(", ")}\n")
                    }
                }
                report.append("\n")
            }
        }
        
        // Resonance patterns
        if (resonanceResult.has("numerical_resonance")) {
            val numericalResonance = resonanceResult.getJSONObject("numerical_resonance")
            if (numericalResonance.has("resonances")) {
                report.append("RESONANCE PATTERNS:\n")
                report.append("-".repeat(30)).append("\n")
                
                val resonances = numericalResonance.getJSONArray("resonances")
                for (i in 0 until resonances.length()) {
                    val resonance = resonances.getJSONObject(i)
                    if (resonance.has("resonance_patterns")) {
                        val patterns = resonance.getJSONObject("resonance_patterns")
                        patterns.keys().forEach { patternType ->
                            val pattern = patterns.getJSONObject(patternType)
                            val intensity = pattern.optDouble("intensity", 0.0)
                            report.append("• ${patternType.replace("_", " ").capitalize()}: ")
                            report.append("Intensity ${String.format("%.2f", intensity)}\n")
                        }
                    }
                }
                report.append("\n")
            }
        }
        
        return report.toString()
    }
    
    private fun buildLiminalNarrativeReport(
        narrative: JSONObject,
        thresholds: JSONArray,
        boundaryConcepts: List<String>
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🌙 LIMINAL THRESHOLD NARRATIVE RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        report.append("BOUNDARY CONCEPTS ANALYZED:\n")
        report.append("-".repeat(30)).append("\n")
        boundaryConcepts.forEach { concept ->
            report.append("• $concept\n")
        }
        report.append("\n")
        
        // Narrative details
        report.append("GENERATED NARRATIVE:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Title: ${narrative.optString("title", "Untitled")}\n")
        report.append("Type: ${narrative.optString("type", "Unknown")}\n")
        report.append("Primary Threshold: ${narrative.optString("primary_threshold", "Unknown")}\n\n")
        
        if (narrative.has("domains")) {
            val domains = narrative.getJSONObject("domains")
            report.append("Domains:\n")
            report.append("  From: ${domains.optString("domain_a", "Unknown")}\n")
            report.append("  To: ${domains.optString("domain_b", "Unknown")}\n\n")
        }
        
        // Narrative content (truncated)
        val content = narrative.optString("content", "")
        if (content.isNotEmpty()) {
            report.append("NARRATIVE CONTENT:\n")
            report.append("-".repeat(30)).append("\n")
            if (content.length > 800) {
                report.append("${content.take(800)}...\n\n")
                report.append("[Content truncated - Full narrative contains ${content.length} characters]\n\n")
            } else {
                report.append("$content\n\n")
            }
        }
        
        // Available thresholds
        if (thresholds.length() > 0) {
            report.append("AVAILABLE THRESHOLDS (${thresholds.length()}):\n")
            report.append("-".repeat(30)).append("\n")
            for (i in 0 until minOf(5, thresholds.length())) {
                val threshold = thresholds.getJSONObject(i)
                val name = threshold.optString("name", "Unknown")
                val domainA = threshold.optString("domain_a", "Unknown")
                val domainB = threshold.optString("domain_b", "Unknown")
                val permeability = threshold.optDouble("permeability", 0.0)
                
                report.append("• $name\n")
                report.append("  $domainA ↔ $domainB\n")
                report.append("  Permeability: ${String.format("%.2f", permeability)}\n\n")
            }
            if (thresholds.length() > 5) {
                report.append("... and ${thresholds.length() - 5} more thresholds\n")
            }
        }
        
        return report.toString()
    }
    
    private fun buildExcessDynamicsReport(
        surplus: JSONObject,
        dynamics: JSONObject,
        cycle: JSONObject,
        state: JSONObject
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("⚡ BATAILLEAN EXCESS DYNAMICS ANALYSIS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Current system state
        report.append("CURRENT SYSTEM STATE:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Accumulation Level: ${String.format("%.3f", state.optDouble("accumulation", 0.0))}\n")
        report.append("Current Phase: ${state.optString("current_phase", "unknown").uppercase()}\n")
        report.append("Pressure: ${String.format("%.3f", state.optDouble("pressure", 0.0))}\n")
        report.append("Entropy: ${String.format("%.3f", state.optDouble("entropy", 0.0))}\n")
        report.append("Transgression Level: ${String.format("%.3f", state.optDouble("transgression_level", 0.0))}\n\n")
        
        // Narrative surplus analysis
        report.append("NARRATIVE SURPLUS ANALYSIS:\n")
        report.append("-".repeat(30)).append("\n")
        val narrativeSurplus = surplus.optDouble("narrative_surplus", 0.0)
        val pressureLevel = surplus.optDouble("pressure", 0.0)
        val expenditurePotential = surplus.optDouble("expenditure_potential", 0.0)
        
        report.append("Surplus Energy: ${String.format("%.3f", narrativeSurplus)}\n")
        report.append("Surplus Category: ${getSurplusCategory(narrativeSurplus)}\n")
        report.append("Pressure Level: ${String.format("%.3f", pressureLevel)}\n")
        report.append("Expenditure Potential: ${String.format("%.3f", expenditurePotential)}\n\n")
        
        // Cycle simulation results
        if (cycle.has("cycle_steps")) {
            val cycleSteps = cycle.getJSONArray("cycle_steps")
            report.append("EXCESS CYCLE SIMULATION (${cycleSteps.length()} steps):\n")
            report.append("-".repeat(30)).append("\n")
            
            for (i in 0 until minOf(5, cycleSteps.length())) {
                val step = cycleSteps.getJSONObject(i)
                val phase = step.optString("phase", "unknown")
                val accumulation = step.optDouble("accumulation", 0.0)
                val pressure = step.optDouble("pressure", 0.0)
                
                report.append("Step ${i + 1}: ${phase.uppercase()}\n")
                report.append("  Accumulation: ${String.format("%.3f", accumulation)}\n")
                report.append("  Pressure: ${String.format("%.3f", pressure)}\n")
            }
            
            if (cycleSteps.length() > 5) {
                report.append("... ${cycleSteps.length() - 5} more steps\n")
            }
            report.append("\n")
        }
        
        // Dynamics effects
        if (dynamics.has("excess_dynamics")) {
            val excessDynamics = dynamics.getJSONObject("excess_dynamics")
            val eventType = excessDynamics.optString("event_type", "unknown")
            
            report.append("DYNAMICS EFFECTS:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Event Type: ${eventType.replace("_", " ").uppercase()}\n")
            
            if (eventType == "expenditure" && excessDynamics.has("expenditure_details")) {
                val details = excessDynamics.getJSONObject("expenditure_details")
                val mode = details.optString("mode", "unknown")
                val intensity = details.optDouble("intensity", 0.0)
                
                report.append("Expenditure Mode: ${mode.uppercase()}\n")
                report.append("Intensity: ${String.format("%.3f", intensity)}\n")
                
                if (details.has("expenditure_forms")) {
                    val forms = details.getJSONArray("expenditure_forms")
                    report.append("Expenditure Forms:\n")
                    for (i in 0 until minOf(3, forms.length())) {
                        val form = forms.getJSONObject(i)
                        val type = form.optString("type", "unknown")
                        val description = form.optString("description", "")
                        report.append("  • ${type.replace("_", " ").uppercase()}: $description\n")
                    }
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildCorrespondenceNetworkReport(
        networks: Map<String, JSONObject>,
        allSymbols: JSONArray,
        detectedSymbols: List<String>
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🜔 HERMETIC CORRESPONDENCE NETWORK RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        report.append("DETECTED SYMBOLS:\n")
        report.append("-".repeat(30)).append("\n")
        if (detectedSymbols.isNotEmpty()) {
            detectedSymbols.forEach { symbol ->
                report.append("• $symbol\n")
            }
        } else {
            report.append("No hermetic symbols detected in input text.\n")
            report.append("Using default symbols for demonstration.\n")
        }
        report.append("\n")
        
        // Individual network reports
        networks.forEach { (symbol, network) ->
            if (!network.has("error")) {
                report.append("NETWORK FOR: $symbol\n")
                report.append("-".repeat(30)).append("\n")
                
                val nodeCount = network.optInt("node_count", 0)
                val connectionCount = network.optInt("connection_count", 0)
                val density = network.optDouble("network_density", 0.0)
                
                report.append("Nodes: $nodeCount\n")
                report.append("Connections: $connectionCount\n")
                report.append("Network Density: ${String.format("%.3f", density)}\n")
                
                if (network.has("metadata")) {
                    val metadata = network.getJSONObject("metadata")
                    if (metadata.has("primary_domains")) {
                        val domains = metadata.getJSONArray("primary_domains")
                        val domainList = mutableListOf<String>()
                        for (i in 0 until domains.length()) {
                            domainList.add(domains.getString(i))
                        }
                        report.append("Primary Domains: ${domainList.joinToString(", ")}\n")
                    }
                }
                
                if (network.has("nodes")) {
                    val nodes = network.getJSONArray("nodes")
                    report.append("\nKey Correspondences:\n")
                    for (i in 0 until minOf(5, nodes.length())) {
                        val node = nodes.getJSONObject(i)
                        val id = node.optString("id", "")
                        val type = node.optString("type", "")
                        val domain = node.optString("domain", "")
                        
                        if (type != "seed") {
                            report.append("  • $id ($domain - $type)\n")
                        }
                    }
                    if (nodes.length() > 5) {
                        report.append("  ... and ${nodes.length() - 5} more correspondences\n")
                    }
                }
                report.append("\n")
            } else {
                report.append("ERROR FOR $symbol: ${network.optString("error_message", "Unknown error")}\n\n")
            }
        }
        
        // Available symbols summary
        report.append("AVAILABLE HERMETIC SYMBOLS (${allSymbols.length()}):\n")
        report.append("-".repeat(30)).append("\n")
        for (i in 0 until minOf(8, allSymbols.length())) {
            val symbol = allSymbols.getJSONObject(i)
            val name = symbol.optString("name", "")
            val domain = symbol.optString("primary_domain", "")
            val potency = symbol.optDouble("potency", 0.0)
            
            report.append("• $name ($domain) - Potency: ${String.format("%.2f", potency)}\n")
        }
        if (allSymbols.length() > 8) {
            report.append("... and ${allSymbols.length() - 8} more symbols\n")
        }
        
        return report.toString()
    }
    
    private fun buildSymbolResonanceReport(
        resonances: List<JSONObject>,
        pathways: List<JSONObject>,
        symbols: List<String>
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔯 HERMETIC SYMBOL RESONANCE ANALYSIS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        report.append("ANALYZED SYMBOLS:\n")
        report.append("-".repeat(30)).append("\n")
        symbols.forEach { symbol ->
            report.append("• $symbol\n")
        }
        report.append("\n")
        
        // Resonance calculations
        if (resonances.isNotEmpty()) {
            report.append("SYMBOL RESONANCES:\n")
            report.append("-".repeat(30)).append("\n")
            
            resonances.forEach { resonance ->
                if (!resonance.has("error")) {
                    val resonanceValue = resonance.optDouble("resonance", 0.0)
                    val relationship = resonance.optJSONObject("relationship")
                    
                    report.append("Resonance Strength: ${String.format("%.3f", resonanceValue)}\n")
                    report.append("Category: ${getResonanceCategory(resonanceValue)}\n")
                    
                    if (relationship != null) {
                        val fromSelf = relationship.optString("from_self", "unknown")
                        val fromOther = relationship.optString("from_other", "unknown")
                        report.append("Relationship Type: $fromSelf ↔ $fromOther\n")
                    }
                    
                    if (resonance.has("effects")) {
                        val effects = resonance.getJSONArray("effects")
                        if (effects.length() > 0) {
                            report.append("Effects:\n")
                            for (i in 0 until effects.length()) {
                                val effect = effects.getJSONObject(i)
                                val type = effect.optString("type", "")
                                val description = effect.optString("description", "")
                                val intensity = effect.optDouble("intensity", 0.0)
                                
                                report.append("  • ${type.uppercase()}: $description\n")
                                report.append("    Intensity: ${String.format("%.2f", intensity)}\n")
                            }
                        }
                    }
                    report.append("\n")
                }
            }
        }
        
        // Transformation pathways
        if (pathways.isNotEmpty()) {
            report.append("TRANSFORMATION PATHWAYS:\n")
            report.append("-".repeat(30)).append("\n")
            
            pathways.forEach { pathway ->
                val type = pathway.optString("type", "unknown")
                val source = pathway.optString("source", "")
                val target = pathway.optString("target", "")
                
                report.append("$source → $target\n")
                report.append("Type: ${type.replace("_", " ").uppercase()}\n")
                
                if (pathway.has("difficulty")) {
                    val difficulty = pathway.optDouble("difficulty", 0.0)
                    report.append("Difficulty: ${String.format("%.2f", difficulty)}\n")
                }
                
                if (pathway.has("potential_catalysts")) {
                    val catalysts = pathway.getJSONArray("potential_catalysts")
                    if (catalysts.length() > 0) {
                        val catalystList = mutableListOf<String>()
                        for (i in 0 until catalysts.length()) {
                            catalystList.add(catalysts.getString(i))
                        }
                        report.append("Catalysts: ${catalystList.joinToString(", ")}\n")
                    }
                }
                report.append("\n")
            }
        }
        
        return report.toString()
    }
    
    private fun buildComprehensiveAnalysisReport(
        comprehensiveResult: JSONObject,
        recommendations: JSONArray,
        systemStatus: JSONObject
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🌟 COMPREHENSIVE ARCANE ANALYSIS RESULTS\n")
        report.append("═".repeat(60)).append("\n\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Analysis completeness
        val isComplete = comprehensiveResult.optBoolean("analysis_complete", false)
        report.append("Analysis Status: ${if (isComplete) "✅ COMPLETE" else "⚠️ PARTIAL"}\n\n")
        
        // Numerical analysis summary
        if (comprehensiveResult.has("numerical_analysis")) {
            val numericalAnalysis = comprehensiveResult.getJSONObject("numerical_analysis")
            report.append("🔢 NUMERICAL ANALYSIS:\n")
            report.append("  Values Analyzed: ${numericalAnalysis.length()}\n")
            
            // Show first few analyses
            val keys = numericalAnalysis.keys().asSequence().take(3).toList()
            keys.forEach { key ->
                val analysis = numericalAnalysis.getJSONObject(key)
                val digitalRoot = analysis.optInt("digital_root", 0)
                report.append("  • $key → Digital Root: $digitalRoot\n")
            }
            if (numericalAnalysis.length() > 3) {
                report.append("  ... and ${numericalAnalysis.length() - 3} more values\n")
            }
            report.append("\n")
        }
        
        // Correspondence networks summary
        if (comprehensiveResult.has("correspondence_networks")) {
            val networks = comprehensiveResult.getJSONObject("correspondence_networks")
            report.append("🜔 HERMETIC CORRESPONDENCES:\n")
            report.append("  Networks Generated: ${networks.length()}\n")
            
            networks.keys().forEach { symbol ->
                val network = networks.getJSONObject(symbol)
                if (!network.has("error")) {
                    val nodeCount = network.optInt("node_count", 0)
                    val connectionCount = network.optInt("connection_count", 0)
                    report.append("  • $symbol: $nodeCount nodes, $connectionCount connections\n")
                } else {
                    report.append("  • $symbol: ERROR\n")
                }
            }
            report.append("\n")
        }
        
        // Excess dynamics summary
        if (comprehensiveResult.has("excess_dynamics")) {
            val dynamics = comprehensiveResult.getJSONObject("excess_dynamics")
            if (dynamics.has("excess_dynamics")) {
                val excessData = dynamics.getJSONObject("excess_dynamics")
                val eventType = excessData.optString("event_type", "unknown")
                report.append("⚡ EXCESS DYNAMICS:\n")
                report.append("  Event Type: ${eventType.replace("_", " ").uppercase()}\n")
                
                if (excessData.has("system_state")) {
                    val state = excessData.getJSONObject("system_state")
                    val phase = state.optString("current_phase", "unknown")
                    val accumulation = excessData.optDouble("current_accumulation", 0.0)
                    report.append("  Current Phase: ${phase.uppercase()}\n")
                    report.append("  Accumulation: ${String.format("%.3f", accumulation)}\n")
                }
                report.append("\n")
            }
        }
        
        // Recommendations
        if (recommendations.length() > 0) {
            report.append("📋 RECOMMENDED OPERATIONS:\n")
            report.append("-".repeat(30)).append("\n")
            
            for (i in 0 until recommendations.length()) {
                val recommendation = recommendations.getJSONObject(i)
                val operation = recommendation.optString("operation", "")
                val priority = recommendation.optString("priority", "")
                val reason = recommendation.optString("reason", "")
                
                val priorityIcon = when (priority) {
                    "high" -> "🔴"
                    "medium" -> "🟡"
                    "low" -> "🟢"
                    else -> "⚪"
                }
                
                report.append("$priorityIcon ${operation.replace("_", " ").uppercase()}\n")
                report.append("   Reason: $reason\n\n")
            }
        }
        
        // System health
        val systemInitialized = systemStatus.optBoolean("system_initialized", false)
        val status = systemStatus.optString("status", "unknown")
        report.append("🏥 SYSTEM HEALTH:\n")
        report.append("  Status: ${if (systemInitialized) "✅" else "❌"} $status\n")
        report.append("  Components: ${systemStatus.optInt("numerical_correspondences_count", 0)} numerical, ")
        report.append("${systemStatus.optInt("liminal_thresholds_count", 0)} thresholds, ")
        report.append("${systemStatus.optInt("hermetic_symbols_count", 0)} symbols\n")
        
        return report.toString()
    }
    
    private fun buildSystemStatusReport(systemStatus: JSONObject, dynamicsState: JSONObject): String {
        val report = StringBuilder()
        
        report.append("📊 ARCANE KNOWLEDGE SYSTEM STATUS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Core system status
        val initialized = systemStatus.optBoolean("system_initialized", false)
        val status = systemStatus.optString("status", "unknown")
        val paradigm = systemStatus.optString("current_paradigm", "unknown")
        
        report.append("CORE SYSTEM:\n")
        report.append("-".repeat(20)).append("\n")
        report.append("Status: ${if (initialized) "✅ OPERATIONAL" else "❌ OFFLINE"} ($status)\n")
        report.append("Current Paradigm: ${paradigm.uppercase()}\n\n")
        
        // Component counts
        report.append("KNOWLEDGE COMPONENTS:\n")
        report.append("-".repeat(20)).append("\n")
        report.append("Numerical Correspondences: ${systemStatus.optInt("numerical_correspondences_count", 0)}\n")
        report.append("Liminal Thresholds: ${systemStatus.optInt("liminal_thresholds_count", 0)}\n")
        report.append("Hermetic Symbols: ${systemStatus.optInt("hermetic_symbols_count", 0)}\n\n")
        
        // Excess dynamics state
        report.append("EXCESS DYNAMICS STATE:\n")
        report.append("-".repeat(20)).append("\n")
        report.append("Accumulation: ${String.format("%.3f", dynamicsState.optDouble("accumulation", 0.0))}\n")
        report.append("Phase: ${dynamicsState.optString("current_phase", "unknown").uppercase()}\n")
        report.append("Pressure: ${String.format("%.3f", dynamicsState.optDouble("pressure", 0.0))}\n")
        report.append("Entropy: ${String.format("%.3f", dynamicsState.optDouble("entropy", 0.0))}\n")
        report.append("Transgression: ${String.format("%.3f", dynamicsState.optDouble("transgression_level", 0.0))}\n\n")
        
        // Operation history
        if (operationHistory.isNotEmpty()) {
            report.append("RECENT OPERATIONS:\n")
            report.append("-".repeat(20)).append("\n")
            operationHistory.takeLast(5).forEach { operation ->
                report.append("• $operation\n")
            }
            if (operationHistory.size > 5) {
                report.append("... and ${operationHistory.size - 5} earlier operations\n")
            }
        }
        
        return report.toString()
    }
   
    private fun buildSampleNarrativeReport(sampleNarrative: JSONObject, content: String): String {
        val report = StringBuilder()
        
        report.append("📜 SAMPLE ARCANE NARRATIVE GENERATED\n")
        report.append("═".repeat(50)).append("\n\n")
        
        // Narrative metadata
        report.append("NARRATIVE METADATA:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("ID: ${sampleNarrative.optString("id", "unknown")}\n")
        report.append("Title: ${sampleNarrative.optString("title", "Untitled")}\n")
        report.append("Content Length: ${content.length} characters\n")
        
        // Elements
        if (sampleNarrative.has("elements")) {
            val elements = sampleNarrative.getJSONArray("elements")
            report.append("Elements: ")
            val elementList = mutableListOf<String>()
            for (i in 0 until elements.length()) {
                elementList.add(elements.getString(i))
            }
            report.append("${elementList.joinToString(", ")}\n")
        }
        
        // Numeric significance
        if (sampleNarrative.has("numeric_significance")) {
            val numbers = sampleNarrative.getJSONArray("numeric_significance")
            report.append("Significant Numbers: ")
            val numberList = mutableListOf<Int>()
            for (i in 0 until numbers.length()) {
                numberList.add(numbers.getInt(i))
            }
            report.append("${numberList.joinToString(", ")}\n")
        }
        
        // Domains
        if (sampleNarrative.has("domains")) {
            val domains = sampleNarrative.getJSONArray("domains")
            report.append("Domains: ")
            val domainList = mutableListOf<String>()
            for (i in 0 until domains.length()) {
                domainList.add(domains.getString(i))
            }
            report.append("${domainList.joinToString(", ")}\n")
        }
        
        report.append("\n")
        
        // System properties
        report.append("SYSTEM PROPERTIES:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Energy Level: ${String.format("%.2f", sampleNarrative.optDouble("energy_level", 0.0))}\n")
        report.append("Constraint Level: ${String.format("%.2f", sampleNarrative.optDouble("constraint_level", 0.0))}\n")
        report.append("Structure Density: ${String.format("%.2f", sampleNarrative.optDouble("structure_density", 0.0))}\n")
        report.append("Element Count: ${sampleNarrative.optInt("element_count", 0)}\n\n")
        
        // Content preview
        report.append("CONTENT PREVIEW:\n")
        report.append("-".repeat(30)).append("\n")
        if (content.length > 300) {
            report.append("${content.take(300)}...\n\n")
            report.append("[Content has been loaded into the input field for analysis]\n")
        } else {
            report.append("$content\n\n")
        }
        
        report.append("✨ Ready for arcane analysis operations! ✨")
        
        return report.toString()
    }
    
    // ============================================================================
    // UTILITY AND HELPER METHODS
    // ============================================================================
    
    private fun validateSystemAndInput(): Boolean {
        if (!isInitialized) {
            statusText.text = "❌ System not initialized. Please wait for initialization to complete."
            return false
        }
        
        val inputContent = inputText.text.toString().trim()
        if (inputContent.isEmpty()) {
            statusText.text = "⚠️ Please enter narrative text or generate a sample to analyze."
            Toast.makeText(this, "Input text required", Toast.LENGTH_SHORT).show()
            return false
        }
        
        return true
    }
    
    private fun getCurrentNarrativeElement(): JSONObject {
        if (currentNarrativeElement != null) {
            return currentNarrativeElement!!
        }
        
        // Create narrative element from input
        val inputContent = inputText.text.toString()
        return JSONObject().apply {
            put("id", "input_${System.currentTimeMillis()}")
            put("content", inputContent)
            put("timestamp", getCurrentTimestamp())
            put("source", "user_input")
            
            // Add basic properties for analysis
            put("energy_level", 0.5)
            put("constraint_level", 0.4)
            put("structure_density", 0.6)
            put("element_count", inputContent.split(" ").size)
            
            // Detect structural properties
            put("has_central_elements", inputContent.contains("central") || inputContent.contains("core"))
            put("has_multiple_agents", inputContent.contains("they") || inputContent.contains("we"))
            put("has_normative_structure", inputContent.contains("should") || inputContent.contains("must"))
            put("has_sacred_elements", inputContent.contains("sacred") || inputContent.contains("holy"))
            put("has_identity_structures", inputContent.contains("self") || inputContent.contains("identity"))
            put("has_cosmic_order", inputContent.contains("cosmic") || inputContent.contains("universe"))
            put("has_defined_subject", inputContent.contains("I") || inputContent.contains("subject"))
        }
    }
    
    private fun extractNumericValues(narrative: JSONObject): List<Int> {
        val values = mutableListOf<Int>()
        
        // Extract from content
        val content = narrative.optString("content", "")
        val numbers = Regex("\\d+").findAll(content).map { it.value.toInt() }.toList()
        values.addAll(numbers)
        
        // Extract from numeric_significance if present
        if (narrative.has("numeric_significance")) {
            val significantNumbers = narrative.getJSONArray("numeric_significance")
            for (i in 0 until significantNumbers.length()) {
                values.add(significantNumbers.getInt(i))
            }
        }
        
        // Extract from other numeric fields
        narrative.keys().forEach { key ->
            val value = narrative.get(key)
            if (value is Int && key != "element_count") {
                values.add(value)
            }
        }
        
        return values.distinct().filter { it > 0 }
    }
    
    private fun extractBoundaryConcepts(text: String): List<String> {
        val concepts = mutableListOf<String>()
        
        // Common boundary/liminal concepts
        val boundaryKeywords = listOf(
            "threshold", "boundary", "edge", "border", "margin", "limit",
            "between", "within", "beyond", "across", "through",
            "sacred", "profane", "conscious", "unconscious",
            "known", "unknown", "visible", "invisible",
            "material", "spiritual", "physical", "metaphysical",
            "inner", "outer", "above", "below",
            "light", "dark", "order", "chaos"
        )
        
        // Extract concepts that appear in the text
        boundaryKeywords.forEach { keyword ->
            if (text.contains(keyword, ignoreCase = true)) {
                concepts.add(keyword)
            }
        }
        
        // If no specific concepts found, use generic ones
        if (concepts.isEmpty()) {
            concepts.addAll(listOf("known", "unknown", "conscious", "unconscious"))
        }
        
        return concepts.distinct().take(5) // Limit to 5 concepts
    }
    
    private fun extractHermeticSymbols(text: String): List<String> {
        val symbols = mutableListOf<String>()
        val hermeticSymbolNames = listOf(
            "Mercury", "Sulfur", "Salt", "Ouroboros", "Sol", "Luna", 
            "Mars", "Venus", "Jupiter", "Saturn", "Hermes", "Thoth",
            "Phoenix", "Dragon", "Lion", "Eagle", "Serpent", "Tree of Life",
            "Caduceus", "Philosopher's Stone", "Prima Materia"
        )
        
        hermeticSymbolNames.forEach { symbol ->
            if (text.contains(symbol, ignoreCase = true)) {
                symbols.add(symbol)
            }
        }
        
        // If no symbols found, use default for demonstration
        if (symbols.isEmpty()) {
            symbols.addAll(listOf("Mercury", "Sulfur", "Salt"))
        }
        
        return symbols.distinct()
    }
    
    private fun highlightRelevantOperations(operationType: String) {
        // Reset all buttons to normal state
        resetButtonHighlights()
        
        // Highlight relevant buttons based on operation type
        when (operationType) {
            "numerical" -> {
                numericalResonanceButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "liminal" -> {
                liminalNarrativeButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "excess" -> {
                excessDynamicsButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "hermetic" -> {
                correspondenceNetworkButton.setBackgroundColor(getColor(R.color.highlight_color))
                symbolResonanceButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "comprehensive" -> {
                comprehensiveAnalysisButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
        }
    }
    
    private fun resetButtonHighlights() {
        val buttons = listOf(
            numericalResonanceButton, liminalNarrativeButton, excessDynamicsButton,
            correspondenceNetworkButton, symbolResonanceButton, comprehensiveAnalysisButton
        )
        
        buttons.forEach { button ->
            button.setBackgroundColor(getColor(R.color.button_normal))
        }
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
        numericalResonanceButton.isEnabled = enabled
        liminalNarrativeButton.isEnabled = enabled
        excessDynamicsButton.isEnabled = enabled
        correspondenceNetworkButton.isEnabled = enabled
        symbolResonanceButton.isEnabled = enabled
        comprehensiveAnalysisButton.isEnabled = enabled
        systemStatusButton.isEnabled = enabled
        generateSampleButton.isEnabled = enabled
    }
    
    private fun showLoading(show: Boolean) {
        loader.visibility = if (show) View.VISIBLE else View.GONE
    }
    
    // Category helper methods
    private fun getResonanceCategory(strength: Double): String {
        return when {
            strength > 0.8 -> "VERY STRONG ⚡⚡⚡"
            strength > 0.6 -> "STRONG ⚡⚡"
            strength > 0.4 -> "MODERATE ⚡"
            strength > 0.2 -> "WEAK ○"
            else -> "VERY WEAK ○"
        }
    }
    
    private fun getSurplusCategory(surplus: Double): String {
        return when {
            surplus > 0.8 -> "CRITICAL EXCESS 🔥🔥🔥"
            surplus > 0.6 -> "HIGH SURPLUS 🔥🔥"
            surplus > 0.4 -> "MODERATE SURPLUS 🔥"
            surplus > 0.2 -> "LOW SURPLUS ○"
            else -> "MINIMAL SURPLUS ○"
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clear any cached data
        try {
            arcaneKnowledgeBridge.clearSystemCache()
        } catch (e: Exception) {
            Log.e("MainActivityArcane", "Error clearing cache on destroy: ${e.message}")
        }
    }
    
    override fun onPause() {
        super.onPause()
        // Save current narrative element if needed
        currentNarrativeElement = getCurrentNarrativeElement()
    }
    
    override fun onResume() {
        super.onResume()
        // Restore system state if needed
        if (isInitialized) {
            lifecycleScope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        // Quick system health check
                        val status = arcaneKnowledgeBridge.getSystemStatus()
                        val isOperational = status.optString("status", "") == "operational"
                        
                        if (!isOperational) {
                            // System may have been disrupted, reinitialize
                            initializeArcaneSystem()
                        }
                    }
                } catch (e: Exception) {
                    Log.w("MainActivityArcane", "System health check failed: ${e.message}")
                }
            }
        }
    }
}
```
    private 
