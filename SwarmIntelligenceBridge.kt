package com.example.swarmintelligence

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import org.json.JSONArray
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Bridge between Kotlin and Python modules for the
 * Adaptive Swarm Intelligence algorithm.
 */
class SwarmIntelligenceBridge(private val context: Context) {
    
    private val pythonLock = ReentrantLock()
    private var pythonInstance: Python? = null
    private var swarmBridge: PyObject? = null
    
    // Status flags
    private var isInitialized = false
    private var isRunning = false
    
    init {
        initializePython()
    }
    
    /**
     * Initialize the Python environment and import the bridge module
     */
    private fun initializePython() {
        pythonLock.withLock {
            if (!Python.isStarted()) {
                try {
                    Python.start(AndroidPlatform(context))
                } catch (e: Exception) {
                    e.printStackTrace()
                    return
                }
            }
            
            try {
                pythonInstance = Python.getInstance()
                
                // Import the swarm intelligence bridge module
                val pythonModule = pythonInstance?.getModule("swarm_intelligence_bridge")
                swarmBridge = pythonModule?.callAttr("SwarmIntelligenceBridge")
                
                isInitialized = true
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }
    
    /**
     * Initialize a new simulation with given parameters
     *
     * @param width Width of environment grid
     * @param height Height of environment grid
     * @param numAgents Number of agents in swarm
     * @param seed Random seed for reproducibility (optional)
     * @return Initialization result as JSON string
     */
    fun initializeSimulation(
        width: Int = 100,
        height: Int = 100,
        numAgents: Int = 20,
        seed: Int? = null
    ): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr(
                    "initialize_simulation",
                    width,
                    height,
                    numAgents,
                    seed
                )
                return result?.toString() ?: "{\"error\": \"Failed to initialize simulation\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Execute simulation step(s)
     *
     * @param numSteps Number of steps to execute
     * @return Results as JSON string
     */
    fun step(numSteps: Int = 1): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr("step", numSteps)
                return result?.toString() ?: "{\"error\": \"Failed to execute simulation step\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Get current environment state for visualization
     *
     * @param includeAgents Whether to include agent positions
     * @param includeSignals Whether to include pheromone signals
     * @return Environment state as JSON string
     */
    fun getEnvironmentState(includeAgents: Boolean = true, includeSignals: Boolean = true): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr(
                    "get_environment_state", 
                    includeAgents,
                    includeSignals
                )
                return result?.toString() ?: "{\"error\": \"Failed to get environment state\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Get detailed information about agents
     *
     * @param agentId Specific agent ID or null for all agents
     * @return Agent details as JSON string
     */
    fun getAgentDetails(agentId: Int? = null): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr("get_agent_details", agentId)
                return result?.toString() ?: "{\"error\": \"Failed to get agent details\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Get information about tasks
     *
     * @param includeCompleted Whether to include completed tasks
     * @param includeFailed Whether to include failed tasks
     * @return Task information as JSON string
     */
    fun getTasks(includeCompleted: Boolean = false, includeFailed: Boolean = false): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr(
                    "get_tasks",
                    includeCompleted,
                    includeFailed
                )
                return result?.toString() ?: "{\"error\": \"Failed to get tasks\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Add a new resource to the environment
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @param value Resource value
     * @return Operation result as JSON string
     */
    fun addResource(x: Int, y: Int, value: Float = 10.0f): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr("add_resource", x, y, value)
                return result?.toString() ?: "{\"error\": \"Failed to add resource\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Add a new task to the environment
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @param difficulty Task difficulty (1.0 to 5.0)
     * @param requiredAgents Number of agents required
     * @return Operation result as JSON string
     */
    fun addTask(
        x: Int, 
        y: Int, 
        difficulty: Float = 2.0f, 
        requiredAgents: Int = 1
    ): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr(
                    "add_task", 
                    x, 
                    y, 
                    difficulty, 
                    requiredAgents
                )
                return result?.toString() ?: "{\"error\": \"Failed to add task\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Get current simulation metrics
     *
     * @return Metrics as JSON string
     */
    fun getMetrics(): String {
        return pythonLock.withLock {
            try {
                val result = swarmBridge?.callAttr("get_metrics")
                return result?.toString() ?: "{\"error\": \"Failed to get metrics\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Export current simulation state
     *
     * @return State as JSON string
     */
    fun exportState(): String {
        return pythonLock.withLock {
            try {
                val stateDict = swarmBridge?.callAttr("export_state")
                return stateDict?.toString() ?: "{\"error\": \"Failed to export state\"}"
            } catch (e: Exception) {
                e.printStackTrace()
                return "{\"error\": \"${e.message}\"}"
            }
        }
    }
    
    /**
     * Import a previously exported state
     *
     * @param stateJson State as JSON string
     * @return Success status
     */
    fun importState(stateJson: String): Boolean {
        return pythonLock.withLock {
            try {
                val stateDict = pythonInstance?.builtins?.callAttr("eval", stateJson)
                val result = swarmBridge?.callAttr("import_state", stateDict)
                return result?.toBoolean() ?: false
            } catch (e: Exception) {
                e.printStackTrace()
                return false
            }
        }
    }
    
    /**
     * Update configuration settings
     *
     * @param updates Map of settings to update
     * @return Success status
     */
    fun updateConfig(updates: Map<String, Any>): Boolean {
        return pythonLock.withLock {
            try {
                val configDict = mapToDict(updates)
                val result = swarmBridge?.callAttr("update_config", configDict)
                return result?.toBoolean() ?: false
            } catch (e: Exception) {
                e.printStackTrace()
                return false
            }
        }
    }
    
    /**
     * Convert a Kotlin Map to a Python dict
     */
    private fun mapToDict(map: Map<String, Any>): PyObject? {
        val dictModule = pythonInstance?.getModule("builtins")
        val dict = dictModule?.callAttr("dict")
        
        map.forEach { (key, value) ->
            when (value) {
                is String -> dict?.callAttr("__setitem__", key, value)
                is Int -> dict?.callAttr("__setitem__", key, value)
                is Float -> dict?.callAttr("__setitem__", key, value)
                is Double -> dict?.callAttr("__setitem__", key, value)
                is Boolean -> dict?.callAttr("__setitem__", key, value)
                is Map<*, *> -> {
                    @Suppress("UNCHECKED_CAST")
                    val nestedDict = mapToDict(value as Map<String, Any>)
                    dict?.callAttr("__setitem__", key, nestedDict)
                }
                is List<*> -> {
                    val pyList = dictModule?.callAttr("list")
                    value.forEach { item ->
                        when (item) {
                            is String, is Int, is Float, is Double, is Boolean -> 
                                pyList?.callAttr("append", item)
                            is Map<*, *> -> {
                                @Suppress("UNCHECKED_CAST")
                                val nestedDict = mapToDict(item as Map<String, Any>)
                                pyList?.callAttr("append", nestedDict)
                            }
                            else -> pyList?.callAttr("append", item.toString())
                        }
                    }
                    dict?.callAttr("__setitem__", key, pyList)
                }
                else -> dict?.callAttr("__setitem__", key, value.toString())
            }
        }
        
        return dict
    }
    
    /**
     * Convert JSON positions array to list of coordinates
     */
    fun parsePositions(jsonArray: JSONArray): List<Pair<Float, Float>> {
        val positions = mutableListOf<Pair<Float, Float>>()
        
        for (i in 0 until jsonArray.length()) {
            val position = jsonArray.getJSONArray(i)
            positions.add(Pair(position.getDouble(0).toFloat(), position.getDouble(1).toFloat()))
        }
        
        return positions
    }
    
    /**
     * Clean up resources when no longer needed
     */
    fun cleanup() {
        pythonLock.withLock {
            swarmBridge = null
        }
    }
    
    /**
     * Data class for Swarm intelligence statistics
     */
    data class SwarmStats(
        val resourcesCollected: Int = 0,
        val tasksCompleted: Int = 0,
        val tasksInProgress: Int = 0,
        val explorationCoverage: Float = 0f,
        val agentCount: Int = 0,
        val currentStep: Int = 0
    )
}
