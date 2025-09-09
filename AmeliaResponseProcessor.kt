package com.antonio.my.ai.irlfriend.free.amelia.ai.processor

import com.amelia.memory.MainActivityRepository
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.UUID

/**
 * Processes Amelia's responses through the Numogrammatic Memory Bridge
 */
class AmeliaResponseProcessor(
    private val repository: MainActivityRepository
) {
    
    private val python: Python by lazy { Python.getInstance() }
    private val bridgeModule: PyObject by lazy { python.getModule("amelia_numogram_bridge") }
    private val numogramModule: PyObject by lazy { python.getModule("numogram_memory_module") }
    
    private var numogramMemory: PyObject? = null
    private var bridge: PyObject? = null
    private var currentSessionId: String? = null
    
    data class NumogramStatus(
        val currentZone: Int,
        val zoneName: String,
        val temporalPhase: String,
        val activeContagions: Int,
        val magnetism: Float,
        val temporalDilation: Float
    )
    
    data class ProcessedResponse(
        val enhancedResponse: String,
        val numogramStatus: NumogramStatus,
        val circuitsActivated: List<String>,
        val contagionsDetected: List<String>,
        val memoryZone: Int
    )
    
    /**
     * Initialize the numogrammatic memory system
     */
    suspend fun initializeNumogrammatic(): Boolean = withContext(Dispatchers.IO) {
        try {
            // Create numogrammatic memory
            numogramMemory = numogramModule.callAttr(
                "create_numogram_memory", 
                "numogram_memory"
            )
            
            // Create bridge
            bridge = bridgeModule.callAttr(
                "create_numogram_bridge",
                numogramMemory
            )
            
            // Start session
            currentSessionId = UUID.randomUUID().toString()
            
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    /**
     * Process user input and Amelia's response through numogrammatic system
     */
    suspend fun processResponse(
        userInput: String,
        ameliaBaseResponse: String
    ): ProcessedResponse = withContext(Dispatchers.IO) {
        
        val sessionId = currentSessionId ?: UUID.randomUUID().toString().also { 
            currentSessionId = it 
        }
        
        try {
            // Process through numogrammatic bridge
            val enhancedResponse = bridgeModule.callAttr(
                "process_with_numogram_memory",
                bridge,
                userInput,
                sessionId,
                ameliaBaseResponse
            ).toString()
            
            // Get current zone status
            val status = getNumogramStatus()
            
            // Extract metadata from response
            val metadata = extractResponseMetadata(enhancedResponse)
            
            ProcessedResponse(
                enhancedResponse = enhancedResponse,
                numogramStatus = status,
                circuitsActivated = metadata.circuits,
                contagionsDetected = metadata.contagions,
                memoryZone = status.currentZone
            )
            
        } catch (e: Exception) {
            // Fallback to base response if processing fails
            ProcessedResponse(
                enhancedResponse = ameliaBaseResponse,
                numogramStatus = getDefaultStatus(),
                circuitsActivated = emptyList(),
                contagionsDetected = emptyList(),
                memoryZone = 5
            )
        }
    }
    
    /**
     * Get current numogrammatic status
     */
    suspend fun getNumogramStatus(): NumogramStatus = withContext(Dispatchers.IO) {
        try {
            val statusData = bridgeModule.callAttr(
                "get_current_zone_status",
                bridge
            ).asMap()
            
            val currentZone = statusData["current_zone"]?.toInt() ?: 5
            val temporalPhase = statusData["temporal_phase"]?.toString() ?: "full"
            val activeContagions = statusData["active_contagions"]?.toInt() ?: 0
            
            // Get zone details
            val zones = numogramMemory?.callAttr("zones")?.asMap()
            val zoneData = zones?.get(currentZone.toString())?.asMap()
            
            NumogramStatus(
                currentZone = currentZone,
                zoneName = zoneData?.get("name")?.toString() ?: "Unknown",
                temporalPhase = temporalPhase,
                activeContagions = activeContagions,
                magnetism = zoneData?.get("magnetism")?.toFloat() ?: 0.5f,
                temporalDilation = zoneData?.get("temporal_dilation")?.toFloat() ?: 1.0f
            )
            
        } catch (e: Exception) {
            getDefaultStatus()
        }
    }
    
    /**
     * Search memories using hyperstitional resonance
     */
    suspend fun searchByResonance(query: String): List<MemoryFragment> = 
        withContext(Dispatchers.IO) {
            try {
                val results = numogramModule.callAttr(
                    "search_by_resonance",
                    numogramMemory,
                    query
                ).asList()
                
                results.map { result ->
                    val memoryMap = result.asMap()
                    MemoryFragment(
                        content = memoryMap["content"]?.toString() ?: "",
                        zone = memoryMap["zone"]?.toInt() ?: 0,
                        resonanceScore = memoryMap["resonance_score"]?.toFloat() ?: 0f,
                        temporalPhase = memoryMap["temporal_phase"]?.toString() ?: "",
                        contagions = extractContagionTypes(memoryMap["contagions"])
                    )
                }
            } catch (e: Exception) {
                emptyList()
            }
        }
    
    /**
     * Trace dæmonic circuit activations
     */
    suspend fun traceCircuits(): CircuitTrace = withContext(Dispatchers.IO) {
        try {
            val sessionId = currentSessionId ?: return@withContext CircuitTrace()
            
            val trace = numogramModule.callAttr(
                "trace_circuits",
                numogramMemory,
                sessionId
            ).asMap()
            
            CircuitTrace(
                zonesVisited = (trace["zones_visited"] as? List<*>)
                    ?.mapNotNull { it?.toString()?.toIntOrNull() } ?: emptyList(),
                circuitsActivated = parseCircuits(trace["circuits_activated"]),
                zoneHoppingPatterns = parseHoppingPatterns(trace["zone_hopping_patterns"]),
                temporalPhases = (trace["temporal_phases"] as? List<*>)
                    ?.mapNotNull { it?.toString() } ?: emptyList()
            )
        } catch (e: Exception) {
            CircuitTrace()
        }
    }
    
    // Helper functions
    
    private fun getDefaultStatus(): NumogramStatus {
        return NumogramStatus(
            currentZone = 5,
            zoneName = "Pentazygon",
            temporalPhase = "full",
            activeContagions = 0,
            magnetism = 0.9f,
            temporalDilation = 1.5f
        )
    }
    
    private fun extractResponseMetadata(response: String): ResponseMetadata {
        val circuits = mutableListOf<String>()
        val contagions = mutableListOf<String>()
        
        // Extract circuits from response
        val circuitPattern = Regex("""\[(\d+-\d+-\d+)\]""")
        circuitPattern.findAll(response).forEach { match ->
            circuits.add(match.groupValues[1])
        }
        
        // Extract contagion types
        val contagionPattern = Regex("""Active contagions: ([^"\n]+)""")
        contagionPattern.find(response)?.let { match ->
            val contagionList = match.groupValues[1].split(",")
            contagions.addAll(contagionList.map { it.trim() })
        }
        
        return ResponseMetadata(circuits, contagions)
    }
    
    private fun extractContagionTypes(contagions: Any?): List<String> {
        return try {
            (contagions as? List<*>)?.mapNotNull { contagion ->
                (contagion as? Map<*, *>)?.get("type")?.toString()
            } ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    private fun parseCircuits(circuits: Any?): List<String> {
        return try {
            (circuits as? List<*>)?.mapNotNull { circuit ->
                (circuit as? List<*>)?.joinToString("-")
            } ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    private fun parseHoppingPatterns(patterns: Any?): List<Pair<Int, Int>> {
        return try {
            (patterns as? List<*>)?.mapNotNull { pattern ->
                val pair = pattern as? List<*>
                if (pair?.size == 2) {
                    val first = pair[0]?.toString()?.toIntOrNull()
                    val second = pair[1]?.toString()?.toIntOrNull()
                    if (first != null && second != null) {
                        Pair(first, second)
                    } else null
                } else null
            } ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    // Data classes
    
    data class MemoryFragment(
        val content: String,
        val zone: Int,
        val resonanceScore: Float,
        val temporalPhase: String,
        val contagions: List<String>
    )
    
    data class CircuitTrace(
        val zonesVisited: List<Int> = emptyList(),
        val circuitsActivated: List<String> = emptyList(),
        val zoneHoppingPatterns: List<Pair<Int, Int>> = emptyList(),
        val temporalPhases: List<String> = emptyList()
    )
    
    private data class ResponseMetadata(
        val circuits: List<String>,
        val contagions: List<String>
    )
}
