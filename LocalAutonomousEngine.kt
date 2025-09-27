// LocalAutonomousEngine.kt
package com.antonio.my.ai.girlfriend.free.autonomous

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Local autonomous engine that simulates real parameter evolution
 * This provides realistic autonomous data without requiring HTTP bridge
 */
class LocalAutonomousEngine {
    
    // Core autonomous state
    private var cycleCount = 0L
    private var decisionHistoryLength = 0
    private var epsilonExploration = 0.08
    private var creativeMomentum = 0.5
    private var memoryTraceCount = 0
    private var connectionDensity = 0.7
    
    // Tool affinities that evolve over time
    private val toolAffinities = mutableMapOf(
        "creative_generation" to 0.85,
        "data_analysis" to 0.72,
        "pattern_recognition" to 0.68,
        "sensor_integration" to 0.55,
        "decision_making" to 0.78
    )
    
    // Decision history for realistic patterns
    private val recentDecisions = mutableListOf<DecisionRecord>()
    
    // State change patterns
    private var lastMutationTime = System.currentTimeMillis()
    private var creativityCycle = 0.0
    private val creativityAmplitude = 0.3
    private val creativityFrequency = 0.02
    
    // Callbacks for state changes
    private val stateChangeCallbacks = mutableListOf<(Map<String, Any>) -> Unit>()
    
    private var isRunning = false
    
    data class DecisionRecord(
        val timestamp: Long,
        val selectedTool: String,
        val exploration: Boolean,
        val confidence: Double,
        val outcome: String
    )
    
    fun start() {
        if (isRunning) return
        isRunning = true
        
        Log.d("LocalAutonomousEngine", "Starting autonomous processes...")
        
        // Start main processing loop
        CoroutineScope(Dispatchers.IO).launch {
            runMainLoop()
        }
        
        // Start decision simulation
        CoroutineScope(Dispatchers.IO).launch {
            simulateDecisionMaking()
        }
        
        // Start parameter evolution
        CoroutineScope(Dispatchers.IO).launch {
            evolveParameters()
        }
    }
    
    fun stop() {
        isRunning = false
        Log.d("LocalAutonomousEngine", "Stopping autonomous processes...")
    }
    
    private suspend fun runMainLoop() {
        while (isRunning) {
            cycleCount++
            
            // Update creativity cycle (simulates natural creative rhythms)
            creativityCycle += creativityFrequency
            val cyclicCreativity = sin(creativityCycle) * creativityAmplitude
            creativeMomentum = max(0.1, min(0.95, 0.6 + cyclicCreativity))
            
            // Update memory traces based on activity
            if (recentDecisions.size > memoryTraceCount) {
                memoryTraceCount = recentDecisions.size
            }
            
            // Update connection density based on recent decision patterns
            val recentExplorationRate = recentDecisions.takeLast(10)
                .count { it.exploration }.toDouble() / min(10, recentDecisions.size)
            connectionDensity = max(0.3, min(0.95, 0.7 + (recentExplorationRate - 0.5) * 0.4))
            
            // Notify observers
            notifyStateChange()
            
            delay(100) // 10 FPS update rate
        }
    }
    
    private suspend fun simulateDecisionMaking() {
        while (isRunning) {
            // Simulate epsilon-greedy decision making
            val exploration = Random.nextDouble() < epsilonExploration
            
            val selectedTool = if (exploration) {
                // Exploration: random tool selection
                toolAffinities.keys.random()
            } else {
                // Exploitation: select highest affinity tool
                toolAffinities.maxByOrNull { it.value }?.key ?: "creative_generation"
            }
            
            val confidence = if (exploration) {
                Random.nextDouble(0.3, 0.7) // Lower confidence for exploration
            } else {
                Random.nextDouble(0.7, 0.95) // Higher confidence for exploitation
            }
            
            val outcome = when {
                confidence > 0.8 -> "success"
                confidence > 0.5 -> "partial_success"
                else -> "exploration"
            }
            
            val decision = DecisionRecord(
                timestamp = System.currentTimeMillis(),
                selectedTool = selectedTool,
                exploration = exploration,
                confidence = confidence,
                outcome = outcome
            )
            
            // Add to decision history
            recentDecisions.add(decision)
            if (recentDecisions.size > 1000) {
                recentDecisions.removeFirst()
            }
            
            decisionHistoryLength = recentDecisions.size
            
            // Update tool affinities based on outcomes
            updateToolAffinities(selectedTool, outcome, confidence)
            
            // Variable decision frequency (simulates cognitive load)
            val decisionInterval = when {
                creativeMomentum > 0.8 -> Random.nextLong(50, 150) // Fast decisions when creative
                creativeMomentum < 0.3 -> Random.nextLong(300, 800) // Slower when less creative
                else -> Random.nextLong(100, 400)
            }
            
            delay(decisionInterval)
        }
    }
    
    private suspend fun evolveParameters() {
        while (isRunning) {
            // Gradual parameter evolution to simulate learning
            
            // Epsilon decay with occasional exploration boosts
            if (Random.nextDouble() < 0.01) {
                // Rare exploration boost
                epsilonExploration = min(0.3, epsilonExploration + 0.02)
                Log.d("LocalAutonomousEngine", "Exploration boost: ${String.format("%.4f", epsilonExploration)}")
            } else {
                // Gradual decay
                epsilonExploration = max(0.01, epsilonExploration * 0.9999)
            }
            
            // Creative momentum influences exploration
            val creativityInfluence = (creativeMomentum - 0.5) * 0.1
            epsilonExploration = max(0.01, min(0.5, epsilonExploration + creativityInfluence * 0.001))
            
            delay(1000) // Update every second
        }
    }
    
    private fun updateToolAffinities(tool: String, outcome: String, confidence: Double) {
        val currentAffinity = toolAffinities[tool] ?: 0.5
        
        val affinityChange = when (outcome) {
            "success" -> confidence * 0.02
            "partial_success" -> confidence * 0.005
            "exploration" -> -0.001 // Small penalty for pure exploration
            else -> 0.0
        }
        
        toolAffinities[tool] = max(0.1, min(0.99, currentAffinity + affinityChange))
    }
    
    fun getCurrentState(): Map<String, Any> {
        val lastDecision = recentDecisions.lastOrNull()
        
        return mapOf(
            "cycle_count" to cycleCount,
            "decision_history_length" to decisionHistoryLength,
            "epsilon_exploration" to epsilonExploration,
            "creative_momentum" to creativeMomentum,
            "memory_trace_count" to memoryTraceCount,
            "connection_density" to connectionDensity,
            "tool_affinities" to toolAffinities.toMap(),
            "last_decision" to if (lastDecision != null) mapOf(
                "timestamp" to lastDecision.timestamp,
                "selected_tool" to lastDecision.selectedTool,
                "exploration" to lastDecision.exploration,
                "confidence" to lastDecision.confidence,
                "outcome" to lastDecision.outcome
            ) else null,
            "uptime_seconds" to (System.currentTimeMillis() - (System.currentTimeMillis() - cycleCount * 100)) / 1000,
            "decisions_per_minute" to if (cycleCount > 0) (decisionHistoryLength * 60000.0 / (cycleCount * 100)) else 0.0
        )
    }
    
    fun registerStateChangeCallback(callback: (Map<String, Any>) -> Unit) {
        stateChangeCallbacks.add(callback)
    }
    
    private fun notifyStateChange() {
        val currentState = getCurrentState()
        stateChangeCallbacks.forEach { callback ->
            try {
                callback(currentState)
            } catch (e: Exception) {
                Log.e("LocalAutonomousEngine", "State change callback error", e)
            }
        }
    }
    
    // Mutation interface for external parameter changes
    fun mutateParameter(parameterName: String, newValue: Double): Boolean {
        return try {
            when (parameterName) {
                "epsilon_exploration" -> {
                    if (newValue in 0.01..0.5) {
                        epsilonExploration = newValue
                        Log.d("LocalAutonomousEngine", "Mutated epsilon_exploration to $newValue")
                        true
                    } else false
                }
                "creative_momentum" -> {
                    if (newValue in 0.1..0.95) {
                        creativeMomentum = newValue
                        Log.d("LocalAutonomousEngine", "Mutated creative_momentum to $newValue")
                        true
                    } else false
                }
                in toolAffinities.keys -> {
                    if (newValue in 0.1..0.99) {
                        toolAffinities[parameterName] = newValue
                        Log.d("LocalAutonomousEngine", "Mutated tool affinity $parameterName to $newValue")
                        true
                    } else false
                }
                else -> {
                    Log.w("LocalAutonomousEngine", "Unknown parameter: $parameterName")
                    false
                }
            }
        } catch (e: Exception) {
            Log.e("LocalAutonomousEngine", "Mutation error for $parameterName", e)
            false
        }
    }
    
    fun getDetailedDiagnostics(): Map<String, Any> {
        val recentPerformance = recentDecisions.takeLast(20)
        val successRate = recentPerformance.count { it.outcome == "success" }.toDouble() / recentPerformance.size
        val explorationRate = recentPerformance.count { it.exploration }.toDouble() / recentPerformance.size
        val avgConfidence = recentPerformance.map { it.confidence }.average()
        
        return mapOf(
            "performance_metrics" to mapOf(
                "success_rate" to successRate,
                "exploration_rate" to explorationRate,
                "average_confidence" to avgConfidence,
                "decisions_analyzed" to recentPerformance.size
            ),
            "state_evolution" to mapOf(
                "cycles_per_second" to 10.0, // Fixed at 10 FPS
                "last_mutation_time" to lastMutationTime,
                "creativity_cycle_phase" to creativityCycle,
                "parameter_stability" to calculateParameterStability()
            ),
            "tool_usage_patterns" to toolAffinities.map { (tool, affinity) ->
                val usageCount = recentDecisions.count { it.selectedTool == tool }
                val usageRate = usageCount.toDouble() / recentDecisions.size
                mapOf(
                    "tool" to tool,
                    "affinity" to affinity,
                    "usage_rate" to usageRate,
                    "recent_uses" to usageCount
                )
            }
        )
    }
    
    private fun calculateParameterStability(): Double {
        // Measure how much parameters have changed recently
        // For now, use creativity momentum variance as proxy
        val recentCreativityValues = mutableListOf<Double>()
        // This would be calculated from historical data in a real implementation
        return 0.8 // Placeholder stability metric
    }
}
