package com.example.swarmintelligence

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import org.json.JSONArray

/**
 * Manager class for Swarm Intelligence simulation that handles
 * UI updates and background processing.
 */
class SwarmIntelligenceManager(private val context: Context) {
    
    private val TAG = "SwarmManager"
    private val bridge = SwarmIntelligenceBridge(context)
    
    // Simulation status
    private var isRunning = false
    private var simulationSpeed = 1 // Steps per second
    
    // Simulation parameters
    private var environmentWidth = 100
    private var environmentHeight = 100
    private var agentCount = 20
    
    // Callback interfaces
    private var stateUpdateCallback: ((JSONObject) -> Unit)? = null
    private var statsUpdateCallback: ((SwarmIntelligenceBridge.SwarmStats) -> Unit)? = null
    
    /**
     * Initialize a new simulation with given parameters
     */
    fun initializeSimulation(
        width: Int = 100,
        height: Int = 100,
        numAgents: Int = 20,
        seed: Int? = null,
        callback: ((Boolean) -> Unit)? = null
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val resultJson = bridge.initializeSimulation(width, height, numAgents, seed)
                val result = JSONObject(resultJson)
                
                // Update local parameters
                environmentWidth = width
                environmentHeight = height
                agentCount = numAgents
                
                val success = result.optString("status") == "success"
                
                withContext(Dispatchers.Main) {
                    callback?.invoke(success)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing simulation", e)
                withContext(Dispatchers.Main) {
                    callback?.invoke(false)
                }
            }
        }
    }
    
    /**
     * Start continuous simulation with given speed
     */
    fun startSimulation(stepsPerSecond: Int = 1) {
        if (isRunning) return
        
        simulationSpeed = stepsPerSecond
        isRunning = true
        
        CoroutineScope(Dispatchers.IO).launch {
            while (isRunning) {
                try {
                    // Execute step
                    bridge
    /**
     * Start continuous simulation with given speed
     */
    fun startSimulation(stepsPerSecond: Int = 1) {
        if (isRunning) return
        
        simulationSpeed = stepsPerSecond
        isRunning = true
        
        CoroutineScope(Dispatchers.IO).launch {
            while (isRunning) {
                try {
                    // Execute step
                    bridge.step(1)
                    
                    // Update state
                    updateState()
                    
                    // Update stats
                    updateStats()
                    
                    // Delay based on simulation speed
                    delay((1000 / simulationSpeed).toLong())
                } catch (e: Exception) {
                    Log.e(TAG, "Error during simulation run", e)
                    stopSimulation()
                }
            }
        }
    }
    
    /**
     * Stop the running simulation
     */
    fun stopSimulation() {
        isRunning = false
    }
    
    /**
     * Execute a single simulation step
     */
    fun stepSimulation(callback: ((JSONObject) -> Unit)? = null) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val resultJson = bridge.step(1)
                val result = JSONObject(resultJson)
                
                withContext(Dispatchers.Main) {
                    callback?.invoke(result)
                }
                
                // Update state and stats
                updateState()
                updateStats()
            } catch (e: Exception) {
                Log.e(TAG, "Error executing simulation step", e)
                withContext(Dispatchers.Main) {
                    callback?.invoke(JSONObject().put("error", e.message))
                }
            }
        }
    }
    
    /**
     * Get the current environment state for visualization
     */
    fun getEnvironmentState(callback: (JSONObject) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val stateJson = bridge.getEnvironmentState(true, true)
                val state = JSONObject(stateJson)
                
                withContext(Dispatchers.Main) {
                    callback(state)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting environment state", e)
                withContext(Dispatchers.Main) {
                    callback(JSONObject().put("error", e.message))
                }
            }
        }
    }
    
    /**
     * Add a resource to the environment
     */
    fun addResource(x: Int, y: Int, value: Float = 10.0f, callback: ((Boolean) -> Unit)? = null) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val resultJson = bridge.addResource(x, y, value)
                val result = JSONObject(resultJson)
                val success = result.optString("status") == "success"
                
                withContext(Dispatchers.Main) {
                    callback?.invoke(success)
                }
                
                // Update state
                updateState()
            } catch (e: Exception) {
                Log.e(TAG, "Error adding resource", e)
                withContext(Dispatchers.Main) {
                    callback?.invoke(false)
                }
            }
        }
    }
    
    /**
     * Add a task to the environment
     */
    fun addTask(
        x: Int, 
        y: Int, 
        difficulty: Float = 2.0f, 
        requiredAgents: Int = 1,
        callback: ((Boolean) -> Unit)? = null
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val resultJson = bridge.addTask(x, y, difficulty, requiredAgents)
                val result = JSONObject(resultJson)
                val success = result.optString("status") == "success"
                
                withContext(Dispatchers.Main) {
                    callback?.invoke(success)
                }
                
                // Update state
                updateState()
            } catch (e: Exception) {
                Log.e(TAG, "Error adding task", e)
                withContext(Dispatchers.Main) {
                    callback?.invoke(false)
                }
            }
        }
    }
    
    /**
     * Get information about all agents
     */
    fun getAgentDetails(callback: (JSONObject) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val agentsJson = bridge.getAgentDetails()
                val agents = JSONObject(agentsJson)
                
                withContext(Dispatchers.Main) {
                    callback(agents)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting agent details", e)
                withContext(Dispatchers.Main) {
                    callback(JSONObject().put("error", e.message))
                }
            }
        }
    }
    
    /**
     * Get information about tasks
     */
    fun getTasks(includeCompleted: Boolean = false, includeFailed: Boolean = false, callback: (JSONObject) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val tasksJson = bridge.getTasks(includeCompleted, includeFailed)
                val tasks = JSONObject(tasksJson)
                
                withContext(Dispatchers.Main) {
                    callback(tasks)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error getting tasks", e)
                withContext(Dispatchers.Main) {
                    callback(JSONObject().put("error", e.message))
                }
            }
        }
    }
    
    /**
     * Save the current simulation state
     */
    fun saveState(callback: ((String) -> Unit)? = null) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val state = bridge.exportState()
                
                withContext(Dispatchers.Main) {
                    callback?.invoke(state)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error saving state", e)
                withContext(Dispatchers.Main) {
                    callback?.invoke("{\"error\": \"${e.message}\"}")
                }
            }
        }
    }
    
    /**
     * Load a previously saved simulation state
     */
    fun loadState(stateJson: String, callback: ((Boolean) -> Unit)? = null) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val success = bridge.importState(stateJson)
                
                withContext(Dispatchers.Main) {
                    callback?.invoke(success)
                }
                
                if (success) {
                    // Update state and stats
                    updateState()
                    updateStats()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading state", e)
                withContext(Dispatchers.Main) {
                    callback?.invoke(false)
                }
            }
        }
    }
    
    /**
     * Set callback for environment state updates
     */
    fun setStateUpdateCallback(callback: (JSONObject) -> Unit) {
        stateUpdateCallback = callback
    }
    
    /**
     * Set callback for statistics updates
     */
    fun setStatsUpdateCallback(callback: (SwarmIntelligenceBridge.SwarmStats) -> Unit) {
        statsUpdateCallback = callback
    }
    
    /**
     * Update environment state and notify callback
     */
    private suspend fun updateState() {
        val callback = stateUpdateCallback ?: return
        
        try {
            val stateJson = bridge.getEnvironmentState(true, true)
            val state = JSONObject(stateJson)
            
            withContext(Dispatchers.Main) {
                callback(state)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error updating state", e)
        }
    }
    
    /**
     * Update statistics and notify callback
     */
    private suspend fun updateStats() {
        val callback = statsUpdateCallback ?: return
        
        try {
            val metricsJson = bridge.getMetrics()
            val metrics = JSONObject(metricsJson)
            
            if (metrics.has("metrics")) {
                val metricsObj = metrics.getJSONObject("metrics")
                val stats = SwarmIntelligenceBridge.SwarmStats(
                    resourcesCollected = metricsObj.optInt("resources_collected", 0),
                    tasksCompleted = metricsObj.optInt("tasks_completed", 0),
                    tasksInProgress = metrics.optJSONObject("statistics")
                        ?.optJSONObject("task_summary")
                        ?.optInt("in_progress_tasks", 0) ?: 0,
                    explorationCoverage = metricsObj.optDouble("exploration_coverage", 0.0).toFloat(),
                    agentCount = agentCount,
                    currentStep = metrics.optInt("current_step", 0)
                )
                
                withContext(Dispatchers.Main) {
                    callback(stats)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error updating stats", e)
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        stopSimulation()
        bridge.cleanup()
    }
}
