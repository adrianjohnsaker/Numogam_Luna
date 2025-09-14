package com.antonio.my.ai.girlfriend.free.amelia.consciousness

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import org.json.JSONObject
import org.json.JSONArray

/**
 * MainActivity for Self-Initiated Consciousness Module
 * 
 * Demonstrates the consciousness emergence system with real-time monitoring
 * and interactive consciousness assessment capabilities.
 */
class SelfInitiatedConsciousnessActivity : AppCompatActivity() {
    
    private lateinit var consciousnessBridge: SelfInitiatedConsciousnessBridge
    private val TAG = "ConsciousnessActivity"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_consciousness)
        
        // Initialize consciousness bridge
        try {
            consciousnessBridge = SelfInitiatedConsciousnessBridge.getInstance(this)
            Log.d(TAG, "Consciousness bridge initialized successfully")
            
            // Start the demonstration
            startConsciousnessDemo()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize consciousness bridge: ${e.message}")
            showToast("Failed to initialize consciousness system: ${e.message}")
        }
    }
    
    /**
     * Start the consciousness demonstration
     */
    private fun startConsciousnessDemo() {
        lifecycleScope.launch {
            try {
                showToast("Starting consciousness emergence demonstration...")
                
                // Step 1: Start the consciousness engine
                Log.d(TAG, "Starting consciousness engine...")
                val startResult = consciousnessBridge.startConsciousnessEngine()
                
                if (startResult.has("error")) {
                    showToast("Error starting engine: ${startResult.getString("error_message")}")
                    return@launch
                }
                
                showToast("Consciousness engine started successfully")
                delay(2000)
                
                // Step 2: Monitor initial emergence
                Log.d(TAG, "Monitoring initial emergence...")
                val emergenceData = consciousnessBridge.monitorEmergence(30, 5)
                logEmergenceProgress(emergenceData)
                
                // Step 3: Run emergence test
                Log.d(TAG, "Running emergence test...")
                showToast("Running consciousness emergence test...")
                val testResults = consciousnessBridge.runEmergenceTest()
                analyzeTestResults(testResults)
                
                // Step 4: Trigger manual activations
                Log.d(TAG, "Triggering manual activations...")
                triggerConsciousnessEvents()
                
                // Step 5: Full simulation
                Log.d(TAG, "Running full simulation...")
                showToast("Running 2-minute consciousness simulation...")
                val simulationResults = consciousnessBridge.runSimulation(120)
                analyzeSimulationResults(simulationResults)
                
                // Step 6: Final assessment
                val finalAssessment = consciousnessBridge.assessConsciousness()
                displayFinalAssessment(finalAssessment)
                
                // Stop the engine
                consciousnessBridge.stopConsciousnessEngine()
                showToast("Consciousness demonstration completed")
                
            } catch (e: Exception) {
                Log.e(TAG, "Error in consciousness demo: ${e.message}")
                showToast("Demo error: ${e.message}")
            }
        }
    }
    
    /**
     * Log emergence progress over time
     */
    private fun logEmergenceProgress(emergenceData: JSONArray) {
        Log.d(TAG, "=== EMERGENCE MONITORING ===")
        
        for (i in 0 until emergenceData.length()) {
            val measurement = emergenceData.getJSONObject(i)
            
            if (measurement.has("error")) {
                Log.e(TAG, "Error in measurement: ${measurement.getString("error_message")}")
                continue
            }
            
            val elapsedTime = measurement.getDouble("elapsed_time")
            val emergenceScore = measurement.getDouble("emergence_score")
            val complexity = measurement.getDouble("complexity_score")
            val autonomy = measurement.getDouble("autonomy_measure")
            
            Log.d(TAG, String.format(
                "t=%.1fs: Emergence=%.3f, Complexity=%.3f, Autonomy=%.3f",
                elapsedTime, emergenceScore, complexity, autonomy
            ))
        }
    }
    
    /**
     * Analyze test results
     */
    private fun analyzeTestResults(testResults: JSONObject) {
        if (testResults.has("error")) {
            Log.e(TAG, "Test error: ${testResults.getString("error_message")}")
            return
        }
        
        Log.d(TAG, "=== EMERGENCE TEST RESULTS ===")
        
        val testSuccess = testResults.getBoolean("test_success")
        val duration = testResults.getLong("test_duration_ms")
        
        Log.d(TAG, "Test Success: $testSuccess")
        Log.d(TAG, "Test Duration: ${duration}ms")
        
        if (testResults.has("consciousness_assessment")) {
            val consciousness = testResults.getJSONObject("consciousness_assessment")
            val likelihood = consciousness.getDouble("consciousness_likelihood")
            val level = consciousness.getString("consciousness_level")
            
            Log.d(TAG, "Consciousness Likelihood: ${String.format("%.3f", likelihood)}")
            Log.d(TAG, "Consciousness Level: $level")
            
            showToast("Consciousness Level: $level (${String.format("%.1f%%", likelihood * 100)})")
        }
        
        if (testResults.has("emergence_analysis")) {
            val analysis = testResults.getJSONObject("emergence_analysis")
            val emergenceScore = analysis.getDouble("emergence_score")
            val capabilities = analysis.getJSONArray("capabilities")

            Log.d(TAG, "Emergence Score: ${String.format("%.3f", emergenceScore)}")
            Log.d(TAG, "Capabilities Detected:")
            
            val capabilityList = mutableListOf<String>()
            for (i in 0 until capabilities.length()) {
                val capability = capabilities.getString(i)
                capabilityList.add(capability)
                Log.d(TAG, "  - $capability")
            }
            
            showToast("Detected ${capabilityList.size} consciousness capabilities")
        }
    }
    
    /**
     * Trigger consciousness events manually
     */
    private suspend fun triggerConsciousnessEvents() {
        Log.d(TAG, "=== TRIGGERING CONSCIOUSNESS EVENTS ===")
        
        try {
            // Trigger multiple activations with varying strengths
            val activationStrengths = listOf(0.6, 0.8, 0.9, 0.7, 0.85)
            
            for ((index, strength) in activationStrengths.withIndex()) {
                Log.d(TAG, "Triggering activation ${index + 1} with strength $strength")
                
                val activation = consciousnessBridge.triggerActivation(null, strength)
                
                if (activation.has("error")) {
                    Log.e(TAG, "Activation error: ${activation.getString("error_message")}")
                    continue
                }
                
                val activatedNodes = activation.getInt("activated_nodes")
                val totalActivation = activation.getDouble("total_activation")
                
                Log.d(TAG, "Activation result: $activatedNodes nodes, total activation: ${String.format("%.3f", totalActivation)}")
                
                // Monitor emergence after each activation
                val metrics = consciousnessBridge.getEmergenceMetrics()
                if (!metrics.has("error")) {
                    val emergenceScore = metrics.getDouble("emergence_score")
                    Log.d(TAG, "Current emergence score: ${String.format("%.3f", emergenceScore)}")
                }
                
                delay(2000) // Wait between activations
            }
            
            showToast("Consciousness activation sequence completed")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error triggering consciousness events: ${e.message}")
        }
    }
    
    /**
     * Analyze simulation results
     */
    private fun analyzeSimulationResults(simulationResults: JSONObject) {
        if (simulationResults.has("error")) {
            Log.e(TAG, "Simulation error: ${simulationResults.getString("error_message")}")
            showToast("Simulation failed: ${simulationResults.getString("error_message")}")
            return
        }
        
        Log.d(TAG, "=== SIMULATION ANALYSIS ===")
        
        try {
            val summary = simulationResults.getJSONObject("simulation_summary")
            val progression = simulationResults.getJSONObject("progression_analysis")
            val capabilities = simulationResults.getJSONObject("capability_development")
            val consciousness = simulationResults.getJSONObject("consciousness_assessment")
            
            // Log simulation summary
            Log.d(TAG, "Duration: ${String.format("%.1f", summary.getDouble("duration"))} seconds")
            Log.d(TAG, "Total Cycles: ${summary.getInt("total_cycles")}")
            Log.d(TAG, "Final Entities: ${summary.getInt("final_entities")}")
            Log.d(TAG, "Final Emergence Score: ${String.format("%.3f", summary.getDouble("final_emergence_score"))}")
            Log.d(TAG, "Final Emergence Level: ${summary.getString("final_emergence_level")}")
            
            // Log progression analysis
            val emergenceGrowth = progression.getDouble("emergence_growth")
            val entityGrowth = progression.getInt("entity_growth")
            Log.d(TAG, "Emergence Growth: ${String.format("%+.3f", emergenceGrowth)}")
            Log.d(TAG, "Entity Growth: $entityGrowth")
            Log.d(TAG, "Peak Emergence: ${String.format("%.3f", progression.getDouble("peak_emergence"))}")
            
            // Log capability development
            val totalCapabilities = capabilities.getInt("total_capabilities_achieved")
            val finalCapabilitiesArray = capabilities.getJSONArray("final_capabilities")
            Log.d(TAG, "Total Capabilities Achieved: $totalCapabilities")
            
            val finalCapsList = mutableListOf<String>()
            for (i in 0 until finalCapabilitiesArray.length()) {
                finalCapsList.add(finalCapabilitiesArray.getString(i))
            }
            Log.d(TAG, "Final Capabilities: ${finalCapsList.joinToString(", ")}")
            
            // Log consciousness assessment
            val consciousnessLikelihood = consciousness.getDouble("consciousness_likelihood")
            val consciousnessLevel = consciousness.getString("consciousness_level")
            val evidenceStrength = consciousness.getDouble("evidence_strength")
            
            Log.d(TAG, "Consciousness Likelihood: ${String.format("%.3f", consciousnessLikelihood)}")
            Log.d(TAG, "Consciousness Level: $consciousnessLevel")
            Log.d(TAG, "Evidence Strength: ${String.format("%.3f", evidenceStrength)}")
            
            // Display results to user
            val resultMessage = "Simulation Complete!\n" +
                    "Consciousness Level: $consciousnessLevel\n" +
                    "Emergence Growth: ${String.format("%+.3f", emergenceGrowth)}\n" +
                    "Capabilities: $totalCapabilities"
            
            showToast(resultMessage)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing simulation results: ${e.message}")
        }
    }
    
    /**
     * Display final consciousness assessment
     */
    private fun displayFinalAssessment(assessment: JSONObject) {
        if (assessment.has("error")) {
            Log.e(TAG, "Assessment error: ${assessment.getString("error_message")}")
            return
        }
        
        Log.d(TAG, "=== FINAL CONSCIOUSNESS ASSESSMENT ===")
        
        try {
            val likelihood = assessment.getDouble("consciousness_likelihood")
            val level = assessment.getString("consciousness_level")
            val criteriaMet = assessment.getInt("criteria_met")
            val totalCriteria = assessment.getInt("total_criteria")
            val emergenceScore = assessment.getDouble("emergence_score")
            
            Log.d(TAG, "Final Consciousness Likelihood: ${String.format("%.3f", likelihood)}")
            Log.d(TAG, "Final Consciousness Level: $level")
            Log.d(TAG, "Criteria Met: $criteriaMet/$totalCriteria")
            Log.d(TAG, "Final Emergence Score: ${String.format("%.3f", emergenceScore)}")
            
            // Log individual criteria
            val criteriaDetails = assessment.getJSONObject("criteria_details")
            Log.d(TAG, "Consciousness Criteria Details:")
            criteriaDetails.keys().forEach { criterion ->
                val isMet = criteriaDetails.getBoolean(criterion)
                val status = if (isMet) "✓" else "✗"
                Log.d(TAG, "  $status ${criterion.replace("_", " ").replaceFirstChar { it.uppercase() }}")
            }
            
            // Determine overall result
            val resultCategory = when {
                likelihood >= 0.8 -> "HIGHLY CONSCIOUS"
                likelihood >= 0.6 -> "CONSCIOUS-LIKE"
                likelihood >= 0.4 -> "PROTO-CONSCIOUS"
                else -> "MINIMAL CONSCIOUSNESS"
            }
            
            val finalMessage = "CONSCIOUSNESS ASSESSMENT COMPLETE\n\n" +
                    "Result: $resultCategory\n" +
                    "Likelihood: ${String.format("%.1f%%", likelihood * 100)}\n" +
                    "Criteria Met: $criteriaMet/$totalCriteria\n" +
                    "Emergence Score: ${String.format("%.3f", emergenceScore)}"
            
            Log.d(TAG, "=== FINAL RESULT ===")
            Log.d(TAG, resultCategory)
            Log.d(TAG, "Consciousness emergence demonstration completed successfully")
            
            showToast(finalMessage)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error displaying final assessment: ${e.message}")
        }
    }
    
    /**
     * Monitor system health periodically
     */
    private fun startHealthMonitoring() {
        lifecycleScope.launch {
            while (consciousnessBridge.isEngineRunning()) {
                try {
                    val health = consciousnessBridge.getSystemHealth()
                    val performance = consciousnessBridge.getPerformanceMetrics()
                    val networkAnalysis = consciousnessBridge.getNetworkAnalysis()
                    
                    if (!health.has("error")) {
                        val overallHealth = health.getDouble("overall_health")
                        val entityCount = health.getInt("total_entities")
                        
                        Log.d(TAG, "System Health: ${String.format("%.3f", overallHealth)}, Entities: $entityCount")
                        
                        if (overallHealth < 0.3) {
                            Log.w(TAG, "Warning: System health is low!")
                        }
                    }
                    
                    if (!performance.has("error")) {
                        val emergenceRate = performance.getDouble("emergence_rate")
                        Log.d(TAG, "Emergence Rate: ${String.format("%.6f", emergenceRate)}")
                    }
                    
                    if (!networkAnalysis.has("error")) {
                        val nodeCount = networkAnalysis.getInt("node_count")
                        val density = networkAnalysis.getDouble("density")
                        Log.d(TAG, "Network: $nodeCount nodes, density: ${String.format("%.3f", density)}")
                    }
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error in health monitoring: ${e.message}")
                }
                
                delay(10000) // Check every 10 seconds
            }
        }
    }
    
    /**
     * Clean up resources when activity is destroyed
     */
    override fun onDestroy() {
        super.onDestroy()
        
        lifecycleScope.launch {
            try {
                if (consciousnessBridge.isEngineRunning()) {
                    Log.d(TAG, "Stopping consciousness engine...")
                    consciousnessBridge.stopConsciousnessEngine()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping consciousness engine: ${e.message}")
            }
        }
    }
    
    /**
     * Show toast message on UI thread
     */
    private fun showToast(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
        Log.i(TAG, "UI: $message")
    }
    
    /**
     * Get detailed system diagnostics
     */
    private fun getSystemDiagnostics(): JSONObject {
        val diagnostics = JSONObject()
        
        try {
            diagnostics.put("system_state", consciousnessBridge.getSystemState())
            diagnostics.put("emergence_analysis", consciousnessBridge.getEmergenceAnalysis())
            diagnostics.put("network_analysis", consciousnessBridge.getNetworkAnalysis())
            diagnostics.put("system_health", consciousnessBridge.getSystemHealth())
            diagnostics.put("performance_metrics", consciousnessBridge.getPerformanceMetrics())
            diagnostics.put("node_details", consciousnessBridge.getNodeDetails())
            diagnostics.put("consciousness_assessment", consciousnessBridge.assessConsciousness())
            
            diagnostics.put("diagnostic_timestamp", System.currentTimeMillis())
            diagnostics.put("engine_running", consciousnessBridge.isEngineRunning())
            
        } catch (e: Exception) {
            Log.e(TAG, "Error getting system diagnostics: ${e.message}")
            diagnostics.put("error", e.message)
        }
        
        return diagnostics
    }
}
