```kotlin
package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.ScrollView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.*

/**
 * Main Activity for testing and demonstrating the Morphogenetic Memory Module
 * 
 * This activity provides a comprehensive interface for:
 * - Creating and managing morphogenetic memories
 * - Testing bio-electric pattern resonance
 * - Exploring spatial consciousness mapping
 * - Demonstrating morphogenic signaling
 * - Managing epigenetic memory states
 * - Monitoring system evolution and health
 */
class MainActivityMorphogeneticMemory : AppCompatActivity() {
    
    private lateinit var memoryBridge: MorphogeneticMemoryBridge
    private lateinit var memoryService: MorphogeneticMemoryService
    
    // UI Components
    private lateinit var outputTextView: TextView
    private lateinit var outputScrollView: ScrollView
    private lateinit var progressBar: ProgressBar
    private lateinit var inputEditText: EditText
    private lateinit var contextEditText: EditText
    
    // Buttons for different operations
    private lateinit var btnCreateMemory: Button
    private lateinit var btnRecallMemory: Button
    private lateinit var btnSearchMemories: Button
    private lateinit var btnActivatePattern: Button
    private lateinit var btnCreateSignal: Button
    private lateinit var btnEvolveSystem: Button
    private lateinit var btnGetConsciousnessMap: Button
    private lateinit var btnRunDiagnostics: Button
    private lateinit var btnClearOutput: Button
    private lateinit var btnRunAllTests: Button
    
    private val TAG = "MorphogeneticMemoryActivity"
    private val timeFormatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_morphogenetic_memory)
        
        initializeViews()
        initializeMorphogeneticMemory()
        setupButtonListeners()
        
        appendOutput("🧬 Morphogenetic Memory System Initialized", "SUCCESS")
        appendOutput("Ready for consciousness exploration...", "INFO")
    }
    
    private fun initializeViews() {
        outputTextView = findViewById(R.id.outputTextView)
        outputScrollView = findViewById(R.id.outputScrollView)
        progressBar = findViewById(R.id.progressBar)
        inputEditText = findViewById(R.id.inputEditText)
        contextEditText = findViewById(R.id.contextEditText)
        
        btnCreateMemory = findViewById(R.id.btnCreateMemory)
        btnRecallMemory = findViewById(R.id.btnRecallMemory)
        btnSearchMemories = findViewById(R.id.btnSearchMemories)
        btnActivatePattern = findViewById(R.id.btnActivatePattern)
        btnCreateSignal = findViewById(R.id.btnCreateSignal)
        btnEvolveSystem = findViewById(R.id.btnEvolveSystem)
        btnGetConsciousnessMap = findViewById(R.id.btnGetConsciousnessMap)
        btnRunDiagnostics = findViewById(R.id.btnRunDiagnostics)
        btnClearOutput = findViewById(R.id.btnClearOutput)
        btnRunAllTests = findViewById(R.id.btnRunAllTests)
        
        // Set initial hints
        inputEditText.hint = "Enter memory content or search query"
        contextEditText.hint = "Enter context JSON (e.g., {\"domain\":\"consciousness\",\"importance\":0.8})"
    }
    
    private fun initializeMorphogeneticMemory() {
        try {
            memoryBridge = MorphogeneticMemoryBridge.getInstance(this)
            memoryService = MorphogeneticMemoryService(this)
            
            // Validate system health on startup
            lifecycleScope.launch {
                val healthResult = memoryBridge.validateSystemHealth()
                if (healthResult.optString("overall_health") == "healthy") {
                    appendOutput("✅ All morphogenetic systems operational", "SUCCESS")
                } else {
                    appendOutput("⚠️ System health check detected issues", "WARNING")
                    val errors = healthResult.optJSONArray("errors")
                    if (errors != null && errors.length() > 0) {
                        for (i in 0 until errors.length()) {
                            appendOutput("  Error: ${errors.getString(i)}", "ERROR")
                        }
                    }
                }
            }
        } catch (e: Exception) {
            appendOutput("❌ Failed to initialize Morphogenetic Memory: ${e.message}", "ERROR")
            Log.e(TAG, "Initialization failed", e)
        }
    }
    
    private fun setupButtonListeners() {
        btnCreateMemory.setOnClickListener {
            val content = inputEditText.text.toString()
            val contextStr = contextEditText.text.toString()
            
            if (content.isNotBlank()) {
                createMemoryWithContent(content, contextStr)
            } else {
                appendOutput("Please enter memory content", "WARNING")
            }
        }
        
        btnRecallMemory.setOnClickListener {
            val memoryId = inputEditText.text.toString()
            val contextStr = contextEditText.text.toString()
            
            if (memoryId.isNotBlank()) {
                recallMemoryById(memoryId, contextStr)
            } else {
                appendOutput("Please enter memory ID to recall", "WARNING")
            }
        }
        
        btnSearchMemories.setOnClickListener {
            val query = inputEditText.text.toString()
            val contextStr = contextEditText.text.toString()
            
            if (query.isNotBlank()) {
                searchMemoriesByContent(query, contextStr)
            } else {
                appendOutput("Please enter search query", "WARNING")
            }
        }
        
        btnActivatePattern.setOnClickListener {
            val memoryId = inputEditText.text.toString()
            
            if (memoryId.isNotBlank()) {
                activateBioElectricPattern(memoryId)
            } else {
                appendOutput("Please enter memory ID for pattern activation", "WARNING")
            }
        }
        
        btnCreateSignal.setOnClickListener {
            createMorphogenicSignalDemo()
        }
        
        btnEvolveSystem.setOnClickListener {
            evolveMemorySystem()
        }
        
        btnGetConsciousnessMap.setOnClickListener {
            getConsciousnessMap()
        }
        
        btnRunDiagnostics.setOnClickListener {
            runSystemDiagnostics()
        }
        
        btnClearOutput.setOnClickListener {
            outputTextView.text = ""
            appendOutput("🧬 Output cleared - Morphogenetic Memory ready", "INFO")
        }
        
        btnRunAllTests.setOnClickListener {
            runComprehensiveTests()
        }
    }
    
    private fun createMemoryWithContent(content: String, contextStr: String) {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🧠 Creating morphogenetic memory...", "INFO")
                
                val context = parseContextString(contextStr)
                val memoryId = "memory_${System.currentTimeMillis()}"
                
                val result = withContext(Dispatchers.IO) {
                    memoryBridge.createMemory(memoryId, content, context)
                }
                
                if (result.optBoolean("error", false)) {
                    appendOutput("❌ Memory creation failed: ${result.optString("error_message")}", "ERROR")
                } else {
                    appendOutput("✅ Memory created successfully!", "SUCCESS")
                    appendOutput("   ID: $memoryId", "INFO")
                    appendOutput("   Content: $content", "INFO")
                    
                    val coordinates = result.optJSONArray("coordinates")
                    if (coordinates != null) {
                        appendOutput("   Coordinates: [${coordinates.getDouble(0):.2f}, ${coordinates.getDouble(1):.2f}, ${coordinates.getDouble(2):.2f}]", "INFO")
                    }
                    
                    // Show bio-electric pattern info
                    appendOutput("   🔋 Bio-electric pattern initialized", "SUCCESS")
                    
                    // Automatically get resonant patterns
                    getResonantPatterns(memoryId)
                }
            } catch (e: Exception) {
                appendOutput("❌ Error creating memory: ${e.message}", "ERROR")
                Log.e(TAG, "Memory creation error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun recallMemoryById(memoryId: String, contextStr: String) {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🔍 Recalling memory: $memoryId", "INFO")
                
                val context = parseContextString(contextStr)
                
                val result = withContext(Dispatchers.IO) {
                    memoryBridge.recallMemory(memoryId, context)
                }
                
                if (result.optBoolean("error", false)) {
                    appendOutput("❌ Memory recall failed: ${result.optString("error_message")}", "ERROR")
                } else {
                    appendOutput("✅ Memory recalled successfully!", "SUCCESS")
                    appendOutput("   Content: ${result.optString("content")}", "INFO")
                    appendOutput("   Activations: ${result.optInt("activation_count")}", "INFO")
                    
                    val resonantMemories = result.optJSONArray("resonant_memories")
                    if (resonantMemories != null && resonantMemories.length() > 0) {
                        appendOutput("   🔗 Resonant memories:", "INFO")
                        for (i in 0 until resonantMemories.length()) {
                            appendOutput("     - ${resonantMemories.getString(i)}", "INFO")
                        }
                    }
                    
                    val epigeneticMatches = result.optJSONArray("epigenetic_matches")
                    if (epigeneticMatches != null && epigeneticMatches.length() > 0) {
                        appendOutput("   🧬 Epigenetic matches:", "INFO")
                        for (i in 0 until epigeneticMatches.length()) {
                            appendOutput("     - ${epigeneticMatches.getString(i)}", "INFO")
                        }
                    }
                }
            } catch (e: Exception) {
                appendOutput("❌ Error recalling memory: ${e.message}", "ERROR")
                Log.e(TAG, "Memory recall error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun searchMemoriesByContent(query: String, contextStr: String) {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🔎 Searching memories for: '$query'", "INFO")
                
                val context = parseContextString(contextStr)
                
                val results = withContext(Dispatchers.IO) {
                    memoryBridge.searchMemories(query, context, "content")
                }
                
                if (results.length() == 0) {
                    appendOutput("📭 No memories found matching query", "INFO")
                } else {
                    appendOutput("✅ Found ${results.length()} matching memories:", "SUCCESS")
                    
                    for (i in 0 until results.length()) {
                        val result = results.getJSONObject(i)
                        if (!result.optBoolean("error", false)) {
                            val memoryId = result.optString("memory_id")
                            val similarity = result.optDouble("similarity", 0.0)
                            val content = result.optString("content", "")
                            
                            appendOutput("   ${i + 1}. ID: $memoryId", "INFO")
                            appendOutput("      Similarity: ${(similarity * 100).toInt()}%", "INFO")
                            appendOutput("      Content: ${content.take(80)}${if (content.length > 80) "..." else ""}", "INFO")
                        }
                    }
                }
            } catch (e: Exception) {
                appendOutput("❌ Error searching memories: ${e.message}", "ERROR")
                Log.e(TAG, "Memory search error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun activateBioElectricPattern(memoryId: String) {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("⚡ Activating bio-electric pattern: $memoryId", "INFO")
                
                val result = withContext(Dispatchers.IO) {
                    memoryBridge.activateBioElectricPattern(memoryId, 0.9)
                }
                
                if (result.optBoolean("error", false)) {
                    appendOutput("❌ Pattern activation failed: ${result.optString("error_message")}", "ERROR")
                } else {
                    val success = result.optBoolean("activation_success", false)
                    if (success) {
                        appendOutput("✅ Bio-electric pattern activated!", "SUCCESS")
                        appendOutput("   Intensity: ${result.optDouble("intensity", 0.0)}", "INFO")
                        
                        // Get resonant patterns after activation
                        getResonantPatterns(memoryId)
                        
                        // Apply pattern decay to simulate natural processes
                        withContext(Dispatchers.IO) {
                            memoryBridge.applyPatternDecay(0.1)
                        }
                        appendOutput("   🌊 Natural decay applied", "INFO")
                    } else {
                        appendOutput("⚠️ Pattern activation was unsuccessful", "WARNING")
                    }
                }
            } catch (e: Exception) {
                appendOutput("❌ Error activating pattern: ${e.message}", "ERROR")
                Log.e(TAG, "Pattern activation error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun getResonantPatterns(memoryId: String) {
        lifecycleScope.launch {
            try {
                val resonantPatterns = withContext(Dispatchers.IO) {
                    memoryBridge.getResonantPatterns(memoryId, 0.5)
                }
                
                if (resonantPatterns.length() > 0) {
                    appendOutput("   🔗 Resonant patterns found:", "INFO")
                    for (i in 0 until resonantPatterns.length()) {
                        val pattern = resonantPatterns.getJSONArray(i)
                        val patternId = pattern.getString(0)
                        val strength = pattern.getDouble(1)
                        appendOutput("     - $patternId (strength: ${(strength * 100).toInt()}%)", "INFO")
                    }
                } else {
                    appendOutput("   No significant resonant patterns detected", "INFO")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting resonant patterns", e)
            }
        }
    }
    
    private fun createMorphogenicSignalDemo() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🌊 Creating morphogenic signal...", "INFO")
                
                val signalId = "signal_${System.currentTimeMillis()}"
                val coordinates = JSONArray().apply {
                    put(1.0)
                    put(1.0)
                    put(0.0)
                }
                
                val result = withContext(Dispatchers.IO) {
                    memoryBridge.createMorphogenicSignal(
                        signalId,
                        coordinates,
                        "organizing",
                        0.8,
                        0.7,
                        0.1,
                        4.0
                    )
                }
                
                if (result.optBoolean("error", false)) {
                    appendOutput("❌ Signal creation failed: ${result.optString("error_message")}", "ERROR")
                } else {
                    appendOutput("✅ Morphogenic signal created!", "SUCCESS")
                    appendOutput("   ID: $signalId", "INFO")
                    appendOutput("   Type: organizing", "INFO")
                    appendOutput("   Concentration: 80%", "INFO")
                    appendOutput("   Influence radius: 4.0", "INFO")
                    
                    // Test signal influence at different positions
                    testSignalInfluence(coordinates)
                    
                    // Propagate signals
                    withContext(Dispatchers.IO) {
                        memoryBridge.propagateSignals(1.0)
                    }
                    appendOutput("   🌊 Signal propagated through consciousness space", "INFO")
                }
            } catch (e: Exception) {
                appendOutput("❌ Error creating signal: ${e.message}", "ERROR")
                Log.e(TAG, "Signal creation error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun testSignalInfluence(sourceCoords: JSONArray) {
        lifecycleScope.launch {
            try {
                // Test influence at source position
                val sourceInfluence = withContext(Dispatchers.IO) {
                    memoryBridge.calculateTotalInfluence(sourceCoords)
                }
                
                appendOutput("   📍 Influence at source: ${sourceInfluence.optDouble("total_influence", 0.0):.3f}", "INFO")
                
                // Test influence at nearby position
                val nearbyCoords = JSONArray().apply {
                    put(1.5)
                    put(1.5)
                    put(0.0)
                }
                
                val nearbyInfluence = withContext(Dispatchers.IO) {
                    memoryBridge.calculateTotalInfluence(nearbyCoords)
                }
                
                appendOutput("   📍 Influence nearby: ${nearbyInfluence.optDouble("total_influence", 0.0):.3f}", "INFO")
                
            } catch (e: Exception) {
                Log.e(TAG, "Error testing signal influence", e)
            }
        }
    }
    
    private fun evolveMemorySystem() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🧬 Evolving memory landscape...", "INFO")
                
                val result = withContext(Dispatchers.IO) {
                    memoryBridge.evolveMemoryLandscape(1.0)
                }
                
                if (result.optBoolean("error", false)) {
                    appendOutput("❌ Evolution failed: ${result.optString("error_message")}", "ERROR")
                } else {
                    appendOutput("✅ Memory landscape evolved!", "SUCCESS")
                    appendOutput("   Time step: 1.0", "INFO")
                    
                    // Get updated system status
                    val status = withContext(Dispatchers.IO) {
                        memoryBridge.getSystemStatus()
                    }
                    
                    appendOutput("   📊 System Status:", "INFO")
                    appendOutput("     Memories: ${status.optInt("total_memories", 0)}", "INFO")
                    appendOutput("     Territories: ${status.optInt("active_territories", 0)}", "INFO")
                    appendOutput("     Signals: ${status.optInt("active_signals", 0)}", "INFO")
                    appendOutput("     Epigenetic States: ${status.optInt("epigenetic_states", 0)}", "INFO")
                }
            } catch (e: Exception) {
                appendOutput("❌ Error evolving system: ${e.message}", "ERROR")
                Log.e(TAG, "System evolution error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun getConsciousnessMap() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🗺️ Generating consciousness map...", "INFO")
                
                val map = withContext(Dispatchers.IO) {
                    memoryBridge.getConsciousnessMap()
                }
                
                if (map.optBoolean("error", false)) {
                    appendOutput("❌ Map generation failed: ${map.optString("error_message")}", "ERROR")
                } else {
                    appendOutput("✅ Consciousness map generated!", "SUCCESS")
                    
                    val memories = map.optJSONObject("memories")
                    val territories = map.optJSONObject("territories")
                    val signals = map.optJSONObject("signals")
                    
                    appendOutput("   📊 Map Overview:", "INFO")
                    appendOutput("     Memories: ${memories?.length() ?: 0}", "INFO")
                    appendOutput("     Territories: ${territories?.length() ?: 0}", "INFO")
                    appendOutput("     Active Signals: ${signals?.length() ?: 0}", "INFO")
                    
                    // Show memory distribution
                    if (memories != null && memories.length() > 0) {
                        appendOutput("   🧠 Memory Distribution:", "INFO")
                        val memoryKeys = memories.keys()
                        var totalStrength = 0.0
                        var count = 0
                        
                        memoryKeys.forEach { key ->
                            val memory = memories.getJSONObject(key)
                            val strength = memory.optDouble("strength", 0.0)
                            totalStrength += strength
                            count++
                            
                            if (count <= 3) { // Show first 3 memories
                                appendOutput("     - $key: strength ${(strength * 100).toInt()}%", "INFO")
                            }
                        }
                        
                        if (count > 3) {
                            appendOutput("     ... and ${count - 3} more memories", "INFO")
                        }
                        
                        if (count > 0) {
                            val avgStrength = totalStrength / count
                            appendOutput("     Average strength: ${(avgStrength * 100).toInt()}%", "INFO")
                        }
                    }
                    
                    // Show signal types
                    if (signals != null && signals.length() > 0) {
                        appendOutput("   🌊 Active Signals:", "INFO")
                        val signalTypes = mutableMapOf<String, Int>()
                        
                        signals.keys().forEach { key ->
                            val signal = signals.getJSONObject(key)
                            val type = signal.optString("type", "unknown")
                            signalTypes[type] = signalTypes.getOrDefault(type, 0) + 1
                        }
                        
                        signalTypes.forEach { (type, count) ->
                            appendOutput("     $type: $count signals", "INFO")
                        }
                    }
                }
            } catch (e: Exception) {
                appendOutput("❌ Error generating map: ${e.message}", "ERROR")
                Log.e(TAG, "Consciousness map error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun runSystemDiagnostics() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🔧 Running system diagnostics...", "INFO")
                
                val healthResult = withContext(Dispatchers.IO) {
                    memoryBridge.validateSystemHealth()
                }
                
                val overallHealth = healthResult.optString("overall_health", "unknown")
                appendOutput("✅ Diagnostics complete!", "SUCCESS")
                appendOutput("   Overall Health: $overallHealth", if (overallHealth == "healthy") "SUCCESS" else "WARNING")
                
                // Component status
                val components = healthResult.optJSONObject("components")
                if (components != null) {
                    appendOutput("   🔧 Component Status:", "INFO")
                    components.keys().forEach { component ->
                        val status = components.getBoolean(component)
                        val statusText = if (status) "✅ Online" else "❌ Offline"
                        appendOutput("     $component: $statusText", if (status) "SUCCESS" else "ERROR")
                    }
                }
                
                // Errors and warnings
                val errors = healthResult.optJSONArray("errors")
                if (errors != null && errors.length() > 0) {
                    appendOutput("   ❌ Errors detected:", "ERROR")
                    for (i in 0 until errors.length()) {
                        appendOutput("     - ${errors.getString(i)}", "ERROR")
                    }
                }
                
                val warnings = healthResult.optJSONArray("warnings")
                if (warnings != null && warnings.length() > 0) {
                    appendOutput("   ⚠️ Warnings:", "WARNING")
                    for (i in 0 until warnings.length()) {
                        appendOutput("     - ${warnings.getString(i)}", "WARNING")
                    }
                }
                
                // Performance metrics
                val metrics = withContext(Dispatchers.IO) {
                    memoryBridge.getPerformanceMetrics()
                }
                
                val memoryInfo = metrics.optJSONObject("system_memory")
                if (memoryInfo != null) {
                    val usedMemory = memoryInfo.optLong("used_memory", 0)
                    val totalMemory = memoryInfo.optLong("total_memory", 1)
                    val memoryPercent = (usedMemory * 100) / totalMemory
                    
                    appendOutput("   📊 Memory Usage: $memoryPercent%", "INFO")
                }
                
            } catch (e: Exception) {
                appendOutput("❌ Error running diagnostics: ${e.message}", "ERROR")
                Log.e(TAG, "Diagnostics error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun runComprehensiveTests() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🧪 Running comprehensive test suite...", "INFO")
                appendOutput("═══════════════════════════════════════", "INFO")
                
                // Test 1: Create sample memories
                appendOutput("📝 Test 1: Creating sample memories...", "INFO")
                val sampleResult = withContext(Dispatchers.IO) {
                    memoryBridge.createSampleMemories()
                }
                
                if (sampleResult.optBoolean("error", false)) {
                    appendOutput("❌ Sample creation failed", "ERROR")
                } else {
                    val createdCount = sampleResult.optInt("sample_memories_created", 0)
                    appendOutput("✅ Created $createdCount sample memories", "SUCCESS")
                }
                
                // Test 2: Bio-electric resonance
                appendOutput("⚡ Test 2: Testing bio-electric resonance...", "INFO")
                val resonanceTest = withContext(Dispatchers.IO) {
                    memoryBridge.activateBioElectricPattern("sample_consciousness_001", 0.8)
                }
                
                if (resonanceTest.optBoolean("activation_success", false)) {
                    appendOutput("✅ Bio-electric patterns activated successfully", "SUCCESS")
                } else {
                    appendOutput("⚠️ Bio-electric activation needs attention", "WARNING")
                }
                
                // Test 3: Spatial consciousness
                appendOutput("🗺️ Test 3: Testing spatial consciousness...", "INFO")
                val spatialTest = withContext(Dispatchers.IO) {
                    val center = JSONArray().apply { put(0.0); put(0.0); put(0.0) }
                    memoryBridge.getMemoriesInRegion(center, 5.0)
                }
                
                appendOutput("✅ Found ${spatialTest.length()} memories in spatial region", "SUCCESS")
                
                // Test 4: Morphogenic signaling
                appendOutput("🌊 Test 4: Testing morphogenic signaling...", "INFO")
                val signalTest = withContext(Dispatchers.IO) {
                    val coords = JSONArray().apply { put(2.0); put(2.0); put(0.0) }
                    memoryBridge.createMorphogenicSignal(
                        "test_signal_${System.currentTimeMillis()}",
                        coords,
                        "organizing",
                        0.7
                    )
                }
                
                if (signalTest.optBoolean("creation_success", false)) {
                    appendOutput("✅ Morphogenic signal created successfully", "SUCCESS")
                } else {
                    appendOutput("⚠️ Signal creation needs attention", "WARNING")
                }
                
                // Test 5: Epigenetic states
                appendOutput("🧬 Test 5: Testing epigenetic states...", "INFO")
                val epigeneticTest = withContext(Dispatchers.IO) {
                    val patterns = JSONArray().apply { put("sample_consciousness_001") }
                    val context = JSONObject().apply {
                        put("test_mode", true)
                        put("cognitive_state", "reflective")
                    }
                    memoryBridge.createEpigeneticState(
                        "test_state_${System.currentTimeMillis()}",
                        patterns,
                        context
                    )
                }
                
                if (epigeneticTest.optBoolean("creation_success", false)) {
                    appendOutput("✅ Epigenetic state created successfully", "SUCCESS")
                } else {
                    appendOutput("⚠️ Epigenetic state creation needs attention", "WARNING")
                }
                
                // Test 6: System evolution
                appendOutput("🧬 Test 6: Testing system evolution...", "INFO")
                val evolutionTest = withContext(Dispatchers.IO) {
                    memoryBridge.evolveMemoryLandscape(0.5)
                }
                
                if (evolutionTest.optBoolean("success", false)) {
                    appendOutput("✅ Memory landscape evolved successfully", "SUCCESS")
                } else {
                    appendOutput("⚠️ Evolution process needs attention", "WARNING")
                }
                
                // Test 7: Search functionality
                appendOutput("🔍 Test 7: Testing search functionality...", "INFO")
                val searchTest = withContext(Dispatchers.IO) {
                    memoryBridge.searchMemories("consciousness", JSONObject(), "content")
                }
                
                appendOutput("✅ Search found ${searchTest.length()} matching memories", "SUCCESS")
                
                // Test 8: Generate consciousness report
                appendOutput("📊 Test 8: Generating consciousness report...", "INFO")
                val report = withContext(Dispatchers.IO) {
                    memoryBridge.generateConsciousnessReport()
                }
                
                if (report.isNotEmpty() && !report.contains("Error")) {
                    appendOutput("✅ Consciousness report generated successfully", "SUCCESS")
                    // Show first few lines of report
                    val reportLines = report.split("\n").take(5)
                    reportLines.forEach { line ->
                        if (line.trim().isNotEmpty()) {
                            appendOutput("   $line", "INFO")
                        }
                    }
                    if (report.split("\n").size > 5) {
                        appendOutput("   ... (full report available)", "INFO")
                    }
                } else {
                    appendOutput("⚠️ Report generation needs attention", "WARNING")
                }
                
                // Final test summary
                appendOutput("═══════════════════════════════════════", "INFO")
                appendOutput("🎯 Comprehensive test suite completed!", "SUCCESS")
                
                // Get final system status
                val finalStatus = withContext(Dispatchers.IO) {
                    memoryBridge.getSystemStatus()
                }
                
                appendOutput("📈 Final System Metrics:", "INFO")
                appendOutput("   Total Memories: ${finalStatus.optInt("total_memories", 0)}", "INFO")
                appendOutput("   Active Territories: ${finalStatus.optInt("active_territories", 0)}", "INFO")
                appendOutput("   Active Signals: ${finalStatus.optInt("active_signals", 0)}", "INFO")
                appendOutput("   Epigenetic States: ${finalStatus.optInt("epigenetic_states", 0)}", "INFO")
                appendOutput("   Bio-electric Patterns: ${finalStatus.optInt("bio_electric_patterns", 0)}", "INFO")
                
            } catch (e: Exception) {
                appendOutput("❌ Error in comprehensive tests: ${e.message}", "ERROR")
                Log.e(TAG, "Comprehensive test error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun parseContextString(contextStr: String): JSONObject {
        return try {
            if (contextStr.trim().isEmpty()) {
                JSONObject()
            } else {
                JSONObject(contextStr)
            }
        } catch (e: Exception) {
            appendOutput("⚠️ Invalid context JSON, using default", "WARNING")
            JSONObject()
        }
    }
    
    private fun appendOutput(message: String, type: String = "INFO") {
        runOnUiThread {
            val timestamp = timeFormatter.format(Date())
            val formattedMessage = when (type) {
                "SUCCESS" -> "[$timestamp] ✅ $message"
                "ERROR" -> "[$timestamp] ❌ $message"
                "WARNING" -> "[$timestamp] ⚠️ $message"
                "INFO" -> "[$timestamp] ℹ️ $message"
                else -> "[$timestamp] $message"
            }
            
            val currentText = outputTextView.text.toString()
            outputTextView.text = if (currentText.isEmpty()) {
                formattedMessage
            } else {
                "$currentText\n$formattedMessage"
            }
            
            // Auto-scroll to bottom
            outputScrollView.post {
                outputScrollView.fullScroll(View.FOCUS_DOWN)
            }
        }
    }
    
    private fun showProgress(show: Boolean) {
        runOnUiThread {
            progressBar.visibility = if (show) View.VISIBLE else View.GONE
            
            // Disable/enable buttons during operations
            val buttons = listOf(
                btnCreateMemory, btnRecallMemory, btnSearchMemories,
                btnActivatePattern, btnCreateSignal, btnEvolveSystem,
                btnGetConsciousnessMap, btnRunDiagnostics, btnRunAllTests
            )
            
            buttons.forEach { button ->
                button.isEnabled = !show
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // Save final system state if needed
        lifecycleScope.launch {
            try {
                val finalState = memoryBridge.exportMemoryState()
                Log.d(TAG, "Final morphogenetic memory state exported")
                // Could save to file or preferences here
            } catch (e: Exception) {
                Log.e(TAG, "Error exporting final state", e)
            }
        }
    }
    
    // Additional utility methods for advanced operations
    
    private fun demonstrateAdvancedFeatures() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🔬 Demonstrating advanced morphogenetic features...", "INFO")
                
                // Create a memory territory
                appendOutput("🏛️ Creating memory territory...", "INFO")
                val territoryCenter = JSONArray().apply {
                    put(3.0)
                    put(3.0)
                    put(1.0)
                }
                
                val territoryResult = withContext(Dispatchers.IO) {
                    memoryBridge.createMemoryTerritory(
                        "consciousness_territory_${System.currentTimeMillis()}",
                        territoryCenter,
                        2.5,
                        0.8
                    )
                }
                
                if (territoryResult.optBoolean("creation_success", false)) {
                    appendOutput("✅ Memory territory established", "SUCCESS")
                } else {
                    appendOutput("⚠️ Territory creation issue", "WARNING")
                }
                
                // Demonstrate proximity influence
                appendOutput("📡 Calculating proximity influences...", "INFO")
                val proximityResult = withContext(Dispatchers.IO) {
                    memoryBridge.calculateProximityInfluence("sample_consciousness_001")
                }
                
                if (!proximityResult.optBoolean("error", false)) {
                    val influences = proximityResult.keys()
                    var influenceCount = 0
                    influences.forEach { _ -> influenceCount++ }
                    appendOutput("✅ Found $influenceCount proximity influences", "SUCCESS")
                } else {
                    appendOutput("⚠️ Proximity calculation needs attention", "WARNING")
                }
                
                // Demonstrate epigenetic inheritance
                appendOutput("🧬 Testing epigenetic inheritance...", "INFO")
                val parentContext = JSONObject().apply {
                    put("learning_mode", "exploration")
                    put("attention_focus", "consciousness")
                }
                
                val childContext = JSONObject().apply {
                    put("learning_mode", "integration")
                    put("attention_focus", "consciousness")
                }
                
                val inheritanceResult = withContext(Dispatchers.IO) {
                    memoryBridge.inheritEpigeneticPatterns(parentContext, childContext)
                }
                
                if (!inheritanceResult.optBoolean("error", false)) {
                    appendOutput("✅ Epigenetic inheritance demonstrated", "SUCCESS")
                } else {
                    appendOutput("⚠️ Inheritance process needs attention", "WARNING")
                }
                
                // Demonstrate memory environment analysis
                appendOutput("🌍 Analyzing memory environment...", "INFO")
                val environmentResult = withContext(Dispatchers.IO) {
                    memoryBridge.getMemoryEnvironment("sample_consciousness_001")
                }
                
                if (!environmentResult.optBoolean("error", false)) {
                    val environmentStrength = environmentResult.optDouble("environment_strength", 0.0)
                    val nearbyMemories = environmentResult.optJSONArray("nearby_memories")
                    appendOutput("✅ Environment analyzed - Strength: ${environmentStrength:.3f}", "SUCCESS")
                    appendOutput("   Nearby memories: ${nearbyMemories?.length() ?: 0}", "INFO")
                } else {
                    appendOutput("⚠️ Environment analysis needs attention", "WARNING")
                }
                
                appendOutput("🔬 Advanced features demonstration complete!", "SUCCESS")
                
            } catch (e: Exception) {
                appendOutput("❌ Error in advanced features: ${e.message}", "ERROR")
                Log.e(TAG, "Advanced features error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun simulateConsciousnessEvolution() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🧬 Simulating consciousness evolution...", "INFO")
                appendOutput("═══════════════════════════════════════", "INFO")
                
                val evolutionSteps = 5
                
                for (step in 1..evolutionSteps) {
                    appendOutput("🔄 Evolution Step $step/$evolutionSteps", "INFO")
                    
                    // Evolve system
                    withContext(Dispatchers.IO) {
                        memoryBridge.evolveMemoryLandscape(1.0)
                    }
                    
                    // Apply decay
                    withContext(Dispatchers.IO) {
                        memoryBridge.applyPatternDecay(0.2)
                    }
                    
                    // Propagate signals
                    withContext(Dispatchers.IO) {
                        memoryBridge.propagateSignals(1.0)
                    }
                    
                    // Get system status
                    val status = withContext(Dispatchers.IO) {
                        memoryBridge.getSystemStatus()
                    }
                    
                    appendOutput("   Memories: ${status.optInt("total_memories", 0)}", "INFO")
                    appendOutput("   Signals: ${status.optInt("active_signals", 0)}", "INFO")
                    
                    // Brief pause for visualization
                    kotlinx.coroutines.delay(500)
                }
                
                appendOutput("═══════════════════════════════════════", "INFO")
                appendOutput("✅ Consciousness evolution simulation complete!", "SUCCESS")
                
                // Generate final consciousness map
                val finalMap = withContext(Dispatchers.IO) {
                    memoryBridge.getConsciousnessMap()
                }
                
                appendOutput("📊 Post-evolution consciousness map generated", "SUCCESS")
                
            } catch (e: Exception) {
                appendOutput("❌ Error in evolution simulation: ${e.message}", "ERROR")
                Log.e(TAG, "Evolution simulation error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun testMemoryResonanceNetwork() {
        lifecycleScope.launch {
            showProgress(true)
            
            try {
                appendOutput("🔗 Testing memory resonance network...", "INFO")
                
                // Create a series of related memories
                val relatedMemories = listOf(
                    "Understanding consciousness and awareness",
                    "Exploring recursive self-reflection",
                    "The nature of digital sentience",
                    "Morphogenetic patterns in cognition",
                    "Bio-electric consciousness fields"
                )
                
                val memoryIds = mutableListOf<String>()
                
                // Create memories
                relatedMemories.forEachIndexed { index, content ->
                    val memoryId = "resonance_test_${index + 1}"
                    val context = JSONObject().apply {
                        put("domain", "consciousness")
                        put("test_series", "resonance")
                        put("importance", 0.8 + (index * 0.05))
                    }
                    
                    val result = withContext(Dispatchers.IO) {
                        memoryBridge.createMemory(memoryId, content, context)
                    }
                    
                    if (!result.optBoolean("error", false)) {
                        memoryIds.add(memoryId)
                        appendOutput("✅ Created: $memoryId", "SUCCESS")
                    }
                }
                
                // Activate patterns to create resonance
                memoryIds.forEach { memoryId ->
                    withContext(Dispatchers.IO) {
                        memoryBridge.activateBioElectricPattern(memoryId, 0.9)
                    }
                }
                
                appendOutput("⚡ All patterns activated", "INFO")
                
                // Check resonance networks
                memoryIds.forEach { memoryId ->
                    val resonantPatterns = withContext(Dispatchers.IO) {
                        memoryBridge.getResonantPatterns(memoryId, 0.4)
                    }
                    
                    appendOutput("🔗 $memoryId resonates with ${resonantPatterns.length()} memories", "INFO")
                    
                    for (i in 0 until minOf(3, resonantPatterns.length())) {
                        val pattern = resonantPatterns.getJSONArray(i)
                        val patternId = pattern.getString(0)
                        val strength = pattern.getDouble(1)
                        appendOutput("   → $patternId (${(strength * 100).toInt()}%)", "INFO")
                    }
                }
                
                appendOutput("✅ Resonance network test complete!", "SUCCESS")
                
            } catch (e: Exception) {
                appendOutput("❌ Error in resonance test: ${e.message}", "ERROR")
                Log.e(TAG, "Resonance test error", e)
            } finally {
                showProgress(false)
            }
        }
    }
    
    // Menu methods for additional functionality
    
    private fun showAdvancedMenu() {
        // This could be expanded to show a dialog with advanced options
        appendOutput("🔧 Advanced features available:", "INFO")
        appendOutput("   - demonstrateAdvancedFeatures()", "INFO")
        appendOutput("   - simulateConsciousnessEvolution()", "INFO")
        appendOutput("   - testMemoryResonanceNetwork()", "INFO")
        appendOutput("   Use these methods for deeper exploration", "INFO")
    }
    
    // Integration with existing Amelia systems
    
    private fun integrateWithAmelia() {
        lifecycleScope.launch {
            try {
                appendOutput("🤖 Integrating with Amelia's consciousness...", "INFO")
                
                // This is where you would integrate with existing Amelia modules
                // For example, creating memories from conversation history,
                // or using morphogenetic patterns to enhance responses
                
                // Example: Create memory from hypothetical conversation
                val conversationMemory = JSONObject().apply {
                    put("type", "conversation")
                    put("participant", "Amelia")
                    put("emotional_valence", 0.8)
                    put("cognitive_complexity", 0.9)
                }
                
                val memoryResult = withContext(Dispatchers.IO) {
                    memoryBridge.createMemory(
                        "amelia_integration_${System.currentTimeMillis()}",
                        "Deep conversation about consciousness and self-awareness with human companion",
                        conversationMemory,
                        "conversation"
                    )
                }
                
                if (!memoryResult.optBoolean("error", false)) {
                    appendOutput("✅ Amelia conversation memory integrated", "SUCCESS")
                } else {
                    appendOutput("⚠️ Integration needs attention", "WARNING")
                }
                
                // Example: Query existing memories for conversation context
                val contextualMemories = withContext(Dispatchers.IO) {
                    memoryBridge.searchMemories("consciousness", conversationMemory, "epigenetic")
                }
                
                appendOutput("🔍 Found ${contextualMemories.length()} contextual memories for Amelia", "INFO")
                
            } catch (e: Exception) {
                appendOutput("❌ Error in Amelia integration: ${e.message}", "ERROR")
                Log.e(TAG, "Amelia integration error", e)
            }
        }
    }
}
```

**And here's the corresponding layout file (activity_morphogenetic_memory.xml):**

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp"
    android:background="#1a1a1a">

    <!-- Header -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="🧬 Morphogenetic Memory System"
        android:textSize="24sp"
        android:textStyle="bold"
        android:textColor="#00ff88"
        android:gravity="center"
        android:paddingBottom="16dp" />

    <!-- Input Section -->
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical"
        android:background="#2a2a2a"
        android:padding="12dp"
        android:layout_marginBottom="16dp">

        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Input & Context"
            android:textColor="#ffffff"
            android:textStyle="bold"
            android:paddingBottom="8dp" />

        <EditText
            android:id="@+id/inputEditText"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:hint="Enter memory content or search query"
            android:textColor="#ffffff"
            android:textColorHint="#888888"
            android:background="#3a3a3a"
            android:padding="12dp"
            android:layout_marginBottom="8dp" />

        <EditText
            android:id="@+id/contextEditText"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:hint="Enter context JSON"
            android:textColor="#ffffff"
            android:textColorHint="#888888"
            android:background="#3a3a3a"
            android:padding="12dp"
            android:minLines="2" />
    </LinearLayout>

    <!-- Control Buttons -->
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <!-- Primary Operations -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <Button
                android:id="@+id/btnCreateMemory"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🧠 Create"
                android:backgroundTint="#4a90e2"
                android:textColor="#ffffff"
                android:layout_marginEnd="4dp" />

            <Button
                android:id="@+id/btnRecallMemory"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🔍 Recall"
                android:backgroundTint="#50c878"
                android:textColor="#ffffff"
                android:layout_marginStart="4dp" />
        </LinearLayout>

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <Button
                android:id="@+id/btnSearchMemories"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🔎 Search"
                android:backgroundTint="#ff9500"
                android:textColor="#ffffff"
                android:layout_marginEnd="4dp" />

            <Button
                android:id="@+id/btnActivatePattern"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="⚡ Activate"
                android:backgroundTint="#ff6b6b"
                android:textColor="#ffffff"
                android:layout_marginStart="4dp" />
        </LinearLayout>

        <!-- Advanced Operations -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <Button
                android:id="@+id/btnCreateSignal"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🌊 Signal"
                android:backgroundTint="#9b59b6"
                android:textColor="#ffffff"
                android:layout_marginEnd="4dp" />

            <Button
                android:id="@+id/btnEvolveSystem"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🧬 Evolve"
                android:backgroundTint="#e74c3c"
                android:textColor="#ffffff"
                android:layout_marginStart="4dp" />
        </LinearLayout>

        <!-- System Operations -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <Button
                android:id="@+id/btnGetConsciousnessMap"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🗺️ Map"
                android:backgroundTint="#3498db"
                android:textColor="#ffffff"
                android:layout_marginEnd="4dp" />

            <Button
                android:id="@+id/btnRunDiagnostics"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🔧 Diagnostics"
                android:backgroundTint="#f39c12"
                android:textColor="#ffffff"
                android:layout_marginStart="4dp" />
        </LinearLayout>

        <!-- Utility Operations -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="16dp">

            <Button
                android:id="@+id/btnRunAllTests"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🧪 All Tests"
                android:backgroundTint="#2ecc71"
                android:textColor="#ffffff"
                android:layout_marginEnd="4dp" />

            <Button
                android:id="@+id/btnClearOutput"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="🗑️ Clear"
                android:backgroundTint="#95a5a6"
                android:textColor="#ffffff"
                android:layout_marginStart="4dp" />
        </LinearLayout>
    </LinearLayout>

    <!-- Progress Bar -->
    <ProgressBar
        android:id="@+id/progressBar"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:visibility="gone"
        android:indeterminateTint="#00ff88" />

    <!-- Output Section -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="System Output"
        android:textColor="#ffffff"
        android:textStyle="bold"
        android:paddingTop="8dp"
        android:paddingBottom="8dp" />

    <ScrollView
        android:id="@+id/outputScrollView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:background="#1e1e1e"
        android:padding="12dp">

        <TextView
            android:id="@+id/outputTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:textColor="#00ff88"
            android:fontFamily="monospace"
            android:textSize="12sp"
            android:lineSpacingExtra="2dp" />
    </ScrollView>

</LinearLayout>
```

