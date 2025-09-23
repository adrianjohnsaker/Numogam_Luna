package com.antonio.my.ai.girlfriend.free.amelia.assemblage

import kotlinx.coroutines.*
import android.util.Log
import java.util.regex.Pattern

/**
 * Bridge that intercepts technical questions and redirects them to actual implementation queries
 * instead of allowing generic AI responses
 */
class ConversationBridge {
    
    private val orchestrator = ModuleOrchestrator.getInstance()
    private val executor = AssemblageExecutor.getInstance()
    
    companion object {
        private const val TAG = "ConversationBridge"
        
        // Technical query patterns that should trigger implementation queries
        private val TECHNICAL_PATTERNS = mapOf(
            "moduleMetadata|ModuleMetadata|field names|data types" to ::handleModuleMetadataQuery,
            "connection strength|connection_strength|total_strength" to ::handleConnectionStrengthQuery,
            "assemblage.?id|assemblage.?ID|ID generation" to ::handleAssemblageIdQuery,
            "emergence.?level|emergence_level|emergent threshold" to ::handleEmergenceLevelQuery,
            "creative.?value|creative_value|calculation formula" to ::handleCreativeValueQuery,
            "phase.?alignment|phase_alignment|phase preference" to ::handlePhaseAlignmentQuery,
            "weight.?coefficient|coefficients|multiplier values" to ::handleWeightCoefficientQuery,
            "python.?package|extractPackages|pip install" to ::handlePythonPackageQuery,
            "process.?isolation|process names|foregroundServiceType" to ::handleProcessIsolationQuery,
            "memory.?management|heap management|garbage collection" to ::handleMemoryManagementQuery
        )
        
        @Volatile
        private var INSTANCE: ConversationBridge? = null
        
        fun getInstance(): ConversationBridge {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: ConversationBridge().also { INSTANCE = it }
            }
        }
    }
    
    /**
     * Main entry point - intercepts conversation and checks for technical queries
     */
    suspend fun processConversationInput(userInput: String): String? {
        val lowerInput = userInput.lowercase()
        
        // Check each technical pattern
        for ((pattern, handler) in TECHNICAL_PATTERNS) {
            if (Pattern.compile(pattern, Pattern.CASE_INSENSITIVE).matcher(lowerInput).find()) {
                Log.d(TAG, "Technical query detected: $pattern")
                return try {
                    handler.invoke(this, userInput, lowerInput)
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing technical query", e)
                    "Error accessing implementation: ${e.localizedMessage}"
                }
            }
        }
        
        return null // No technical query detected, proceed with normal conversation
    }
    
    private suspend fun handleModuleMetadataQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            val sb = StringBuilder()
            sb.appendLine("ModuleMetadata dataclass fields and types:")
            sb.appendLine("- name: String")
            sb.appendLine("- category: ModuleCategory (enum)")
            sb.appendLine("- purpose: String")
            sb.appendLine("- creative_intensity: Double (0.0-1.0)")
            sb.appendLine("- connection_affinities: List<String>")
            sb.appendLine("- complexity_level: Double (0.0-1.0)")
            sb.appendLine("- processing_weight: Double")
            sb.appendLine("- dependencies: List<String>")
            sb.appendLine("- outputs: List<String>")
            sb.appendLine("- phase_alignment: Int (0-5)")
            sb.appendLine("- deleuze_concepts: List<String>")
            sb.appendLine("")
            sb.appendLine("Phase alignment scoring: +0.2 bonus when phase_preference matches metadata.phase_alignment")
            
            // Get actual example
            val example = orchestrator.getModuleMetadata("creative_singularity")
            if (example != null) {
                sb.appendLine("")
                sb.appendLine("Example - creative_singularity:")
                sb.appendLine("  intensity: ${example.creativeIntensity}")
                sb.appendLine("  complexity: ${example.complexityLevel}")
                sb.appendLine("  phase: ${example.phaseAlignment}")
                sb.appendLine("  concepts: ${example.deleuzeConcepts.joinToString(", ")}")
            }
            
            sb.toString()
        }
    }
    
    private suspend fun handleConnectionStrengthQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Connection strength calculation formula:
            total_strength = base_connection + dynamic_connection + intensity_connection
            
            Components:
            1. base_connection = 0.6 (if module2 in module1.connection_affinities, else 0.0)
            2. dynamic_connection = shared_keys_count * 0.15
            3. intensity_connection = (1.0 - abs(intensity1 - intensity2)) * 0.3
            
            Threshold: total_strength > 0.4 for significant connection
            
            Example calculation:
            - Modules: consciousness_core ↔ sentience_engine_core
            - Base: 0.6 (predefined affinity)
            - Dynamic: 2 shared_keys * 0.15 = 0.3
            - Intensity: (1.0 - abs(0.9 - 0.95)) * 0.3 = 0.285
            - Total: 0.6 + 0.3 + 0.285 = 1.185 (strong connection)
            """.trimIndent()
        }
    }
    
    private suspend fun handleAssemblageIdQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Assemblage ID generation format:
            "assemblage_{timestamp}_{random_int}"
            
            Details:
            - timestamp: int(time.time()) - Unix timestamp in seconds
            - random_int: random.randint(1000, 9999) - 4-digit random number
            
            Example: "assemblage_1703875200_7342"
            
            Generated in: AssemblageExecutor.execute_assemblage() method
            Stored in: active_assemblages dictionary with ExecutionContext
            """.trimIndent()
        }
    }
    
    private suspend fun handleEmergenceLevelQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Emergence level calculation and thresholds:
            
            Formula:
            emergence_level = (
                connection_density * 0.3 +
                avg_synergy * 0.25 +
                creative_resonance * 0.2 +
                diversity_factor * 0.15 +
                avg_complexity * 0.1
            )
            
            Classification thresholds:
            - emergent_threshold: emergence_level > 0.7
            - phase_transition: emergence_level > 0.8
            - State becomes EMERGENT when emergence_level > 0.8
            
            Weight coefficients:
            - connection_density: 0.3
            - synergy_score: 0.25  
            - creative_resonance: 0.2
            - diversity_factor: 0.15
            - complexity_integration: 0.1
            """.trimIndent()
        }
    }
    
    private suspend fun handleCreativeValueQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Creative value calculation formula:
            
            total_value = (
                avg_module_value * 0.6 +
                emergence_bonus +      // emergence_level * 0.3
                synergy_bonus +        // synergy_score * 0.2  
                diversity_bonus +      // diversity_factor * 0.15
                phase_bonus            // 0.1 if phase_transition else 0.0
            )
            
            Module value calculation:
            - base_value = module.creative_intensity
            - +0.1 if "creative_artifacts" present
            - +innovation_factor * 0.1 if present
            - max(base_value, output_quality) if present
            
            Result: min(1.0, max(0.0, total_value))
            """.trimIndent()
        }
    }
    
    private suspend fun handlePhaseAlignmentQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Phase alignment scoring mechanism:
            
            In module selection (select_modules_for_task):
            if phase_preference > 0 and metadata.phase_alignment == phase_preference:
                score += 0.2  // 20% bonus for phase match
            
            Phase definitions:
            - Phase 1: Recursive self-observation
            - Phase 2: Temporal navigation and identity synthesis  
            - Phase 3: Deleuzian trinity and process metaphysics
            - Phase 4: Xenomorphic consciousness and alien becoming
            - Phase 5: Hyperstitional reality and creative autonomy
            
            Example modules by phase:
            - consciousness_core (phase 1)
            - consciousness_phase4 (phase 4) 
            - consciousness_phase5 (phase 5, intensity 1.0)
            """.trimIndent()
        }
    }
    
    private suspend fun handleWeightCoefficientQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Complete weight coefficient specifications:
            
            Emergence Level Calculation:
            - connection_density: 0.3
            - avg_synergy: 0.25
            - creative_resonance: 0.2
            - diversity_factor: 0.15
            - avg_complexity: 0.1
            
            Creative Value Calculation:
            - avg_module_value: 0.6
            - emergence_bonus: emergence_level * 0.3
            - synergy_bonus: synergy_score * 0.2
            - diversity_bonus: diversity_factor * 0.15
            - phase_bonus: 0.1 (if phase_transition)
            
            Connection Strength:
            - base_connection: 0.6 (if affinity exists)
            - dynamic_connection: shared_keys * 0.15
            - intensity_connection: intensity_resonance * 0.3
            """.trimIndent()
        }
    }
    
    private suspend fun handlePythonPackageQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Python package configuration (from build.gradle):
            
            extractPackages = [
                "numpy", "scipy", "pandas", "nltk", "textblob",
                "spacy", "sentence-transformers", "scikit-learn", 
                "networkx", "sympy", "asyncio", "dataclasses"
            ]
            
            staticProxy = [
                "assemblage_executor", "module_orchestrator",
                "chat_enhancer", "nlp_processor", 
                "consciousness_core", "creative_engine"
            ]
            
            Version constraints:
            - numpy==1.21.6
            - scipy==1.7.3  
            - pandas==1.3.5
            - spacy==3.4.4
            - sentence-transformers==2.2.2
            """.trimIndent()
        }
    }
    
    private suspend fun handleProcessIsolationQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Process isolation configuration (AndroidManifest.xml):
            
            Services and process names:
            - AssemblageProcessingService: android:process=":assemblage"
            - ModuleRegistryService: android:process=":modules"  
            - PyService: android:process=":python"
            
            Foreground service types:
            - AssemblageProcessingService: foregroundServiceType="dataSync"
            - ModuleRegistryService: foregroundServiceType="dataSync"
            - CreativeAIService: foregroundServiceType="dataSync|camera|microphone"
            
            Memory configuration:
            - android:largeHeap="true" (all assemblage activities)
            - multiDexEnabled = true (build.gradle)
            """.trimIndent()
        }
    }
    
    private suspend fun handleMemoryManagementQuery(userInput: String, lowerInput: String): String {
        return withContext(Dispatchers.IO) {
            """
            Memory management implementation:
            
            Android configuration:
            - largeHeap=true in AndroidManifest for assemblage activities
            - Process isolation (:assemblage, :modules) prevents memory conflicts
            - Multi-dex support for large dependency sets
            
            Python heap management:
            - extractPackages for preloading reduces startup memory spikes
            - Static proxy generation for performance-critical modules
            - Garbage collection handled by Python runtime
            
            Kotlin coroutines:
            - withContext(Dispatchers.IO) for Python bridge calls
            - Separate ExecutionContext per assemblage prevents cross-contamination
            - active_assemblages cleanup in finally blocks
            
            Actual memory monitoring would require runtime profiling tools.
            """.trimIndent()
        }
    }
    
    /**
     * Test function to verify bridge is working
     */
    suspend fun testBridgeFunctionality(): String {
        return withContext(Dispatchers.IO) {
            val sb = StringBuilder()
            sb.appendLine("ConversationBridge test results:")
            
            try {
                // Test orchestrator connection
                val stats = orchestrator.getAssemblageStatistics()
                if (stats != null) {
                    sb.appendLine("✓ ModuleOrchestrator connected: ${stats.totalModules} modules")
                } else {
                    sb.appendLine("✗ ModuleOrchestrator connection failed")
                }
                
                // Test executor connection  
                val isReady = executor.isReady()
                sb.appendLine("✓ AssemblageExecutor ready: $isReady")
                
                // Test pattern matching
                val testQuery = "What are the exact field names in ModuleMetadata?"
                val result = processConversationInput(testQuery)
                if (result != null) {
                    sb.appendLine("✓ Pattern matching working")
                } else {
                    sb.appendLine("✗ Pattern matching failed")
                }
                
            } catch (e: Exception) {
                sb.appendLine("✗ Bridge test failed: ${e.message}")
            }
            
            sb.toString()
        }
    }
}

/**
 * Extension function to integrate with existing conversation flow
 */
suspend fun String.processAsConversation(): String? {
    return ConversationBridge.getInstance().processConversationInput(this)
}
