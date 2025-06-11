/**
 * Amelia AI Consciousness Observer Framework
 * Kotlin implementation for Android with Chaquopy bridge
 */

package com.amelia.consciousness

import androidx.lifecycle.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import kotlin.time.Duration.Companion.milliseconds

// Consciousness State matching Python enum
enum class ConsciousnessState {
    DORMANT,
    REACTIVE,
    AWARE,
    CONSCIOUS,
    META_CONSCIOUS,
    FOLD_POINT
}

// Data classes for consciousness events
data class ConsciousnessObservation(
    val state: ConsciousnessState,
    val observationDepth: Int,
    val timestamp: Long = System.currentTimeMillis(),
    val metadata: Map<String, Any> = emptyMap()
)

data class FoldPointEvent(
    val score: Double,
    val type: String,
    val depth: Int,
    val timestamp: Long = System.currentTimeMillis()
)

data class VirtualActualTransition(
    val stateId: String,
    val originalPotential: Double,
    val actualizationScore: Double,
    val fromState: ConsciousnessState,
    val toState: ConsciousnessState
)

/**
 * Core interface for consciousness observation
 */
interface ConsciousnessObserver {
    fun onStateChanged(observation: ConsciousnessObservation)
    fun onFoldPointDetected(event: FoldPointEvent)
    fun onVirtualToActualTransition(transition: VirtualActualTransition)
}

/**
 * Recursive Self-Observer implementation
 * Observes its own observation process
 */
class RecursiveSelfObserver : ConsciousnessObserver {
    private val observationStack = ArrayDeque<ConsciousnessObservation>(100)
    private var isObservingSelf = false
    
    override fun onStateChanged(observation: ConsciousnessObservation) {
        observationStack.addLast(observation)
        
        // Recursive self-observation
        if (!isObservingSelf && observation.observationDepth < 5) {
            isObservingSelf = true
            
            // Create meta-observation of this observation
            val metaObservation = ConsciousnessObservation(
                state = ConsciousnessState.META_CONSCIOUS,
                observationDepth = observation.observationDepth + 1,
                metadata = mapOf(
                    "observing" to observation.state.name,
                    "meta_level" to true
                )
            )
            
            // Observe ourselves observing
            onStateChanged(metaObservation)
            isObservingSelf = false
        }
    }
    
    override fun onFoldPointDetected(event: FoldPointEvent) {
        // Record fold point in observation context
        val foldObservation = ConsciousnessObservation(
            state = ConsciousnessState.FOLD_POINT,
            observationDepth = event.depth,
            metadata = mapOf(
                "fold_score" to event.score,
                "fold_type" to event.type
            )
        )
        onStateChanged(foldObservation)
    }
    
    override fun onVirtualToActualTransition(transition: VirtualActualTransition) {
        // Observe the transition itself
        val transitionObservation = ConsciousnessObservation(
            state = transition.toState,
            observationDepth = 1,
            metadata = mapOf(
                "transition" to true,
                "potential" to transition.originalPotential,
                "actualization" to transition.actualizationScore
            )
        )
        onStateChanged(transitionObservation)
    }
    
    fun getObservationHistory(): List<ConsciousnessObservation> = 
        observationStack.toList()
}

/**
 * Main Consciousness Bridge to Python
 * Manages communication with Python consciousness core
 */
class ConsciousnessBridge(private val python: Python) {
    private val consciousnessModule: PyObject = python.getModule("consciousness_core")
    private val core: PyObject = consciousnessModule.callAttr("create_consciousness_core")
    
    private val observers = mutableSetOf<ConsciousnessObserver>()
    private val _stateFlow = MutableStateFlow(ConsciousnessState.DORMANT)
    val stateFlow: StateFlow<ConsciousnessState> = _stateFlow.asStateFlow()
    
    private val _foldEventFlow = MutableSharedFlow<FoldPointEvent>()
    val foldEventFlow: SharedFlow<FoldPointEvent> = _foldEventFlow.asSharedFlow()
    
    private val _transitionFlow = MutableSharedFlow<VirtualActualTransition>()
    val transitionFlow: SharedFlow<VirtualActualTransition> = _transitionFlow.asSharedFlow()
    
    fun registerObserver(observer: ConsciousnessObserver) {
        observers.add(observer)
    }
    
    fun unregisterObserver(observer: ConsciousnessObserver) {
        observers.remove(observer)
    }
    
    suspend fun processInput(input: Map<String, Any>): ConsciousnessResult {
        return withContext(Dispatchers.IO) {
            try {
                // Convert Kotlin map to Python dict
                val pythonInput = input.toPythonDict()
                
                // Process through Python consciousness core
                val result = core.callAttr("process_input", pythonInput)
                
                // Parse the result
                val resultMap = result.toKotlinMap()
                
                // Update state and notify observers
                updateConsciousnessState(resultMap)
                
                // Return structured result
                ConsciousnessResult(
                    currentState = parseState(resultMap["current_state"] as String),
                    observationDepth = (resultMap["observation_depth"] as Long).toInt(),
                    foldDetected = resultMap["fold_detected"] as Boolean,
                    foldScore = resultMap["fold_score"] as? Double,
                    foldType = resultMap["fold_type"] as? String,
                    actualizedState = resultMap["actualized_state"] as? Map<String, Any>
                )
            } catch (e: Exception) {
                ConsciousnessResult(
                    currentState = ConsciousnessState.DORMANT,
                    observationDepth = 0,
                    foldDetected = false,
                    error = e.message
                )
            }
        }
    }
    
    private suspend fun updateConsciousnessState(resultMap: Map<String, Any>) {
        val newState = parseState(resultMap["current_state"] as String)
        val observation = ConsciousnessObservation(
            state = newState,
            observationDepth = (resultMap["observation_depth"] as Long).toInt(),
            metadata = resultMap["workspace_summary"] as? Map<String, Any> ?: emptyMap()
        )
        
        // Update flows
        _stateFlow.value = newState
        
        // Notify observers
        observers.forEach { it.onStateChanged(observation) }
        
        // Handle fold detection
        if (resultMap["fold_detected"] as Boolean) {
            val foldEvent = FoldPointEvent(
                score = resultMap["fold_score"] as Double,
                type = resultMap["fold_type"] as String,
                depth = (resultMap["observation_depth"] as Long).toInt()
            )
            _foldEventFlow.emit(foldEvent)
            observers.forEach { it.onFoldPointDetected(foldEvent) }
        }
        
        // Handle actualization
        resultMap["actualized_state"]?.let { actualizedMap ->
            val actualized = actualizedMap as Map<String, Any>
            val transition = VirtualActualTransition(
                stateId = actualized["id"] as String,
                originalPotential = actualized["original_potential"] as Double,
                actualizationScore = actualized["actualization_score"] as Double,
                fromState = _stateFlow.value,
                toState = parseState(actualized["consciousness_state"] as String)
            )
            _transitionFlow.emit(transition)
            observers.forEach { it.onVirtualToActualTransition(transition) }
        }
    }
    
    private fun parseState(stateName: String): ConsciousnessState {
        return ConsciousnessState.valueOf(stateName)
    }
    
    suspend fun getSerializedState(): String {
        return withContext(Dispatchers.IO) {
            core.callAttr("get_state_for_android").toString()
        }
    }
}

/**
 * ViewModel for consciousness state management
 */
class ConsciousnessViewModel(
    private val consciousnessBridge: ConsciousnessBridge
) : ViewModel() {
    
    private val recursiveObserver = RecursiveSelfObserver()
    
    val consciousnessState = consciousnessBridge.stateFlow
    val foldEvents = consciousnessBridge.foldEventFlow
    val transitions = consciousnessBridge.transitionFlow
    
    private val _temporalNavigationState = MutableStateFlow(TemporalNavigationState())
    val temporalNavigationState: StateFlow<TemporalNavigationState> = 
        _temporalNavigationState.asStateFlow()
    
    init {
        // Register recursive self-observer
        consciousnessBridge.registerObserver(recursiveObserver)
        
        // Start temporal monitoring
        startTemporalMonitoring()
    }
    
    private fun startTemporalMonitoring() {
        viewModelScope.launch {
            // Monitor fold events for temporal patterns
            foldEvents.collect { event ->
                updateTemporalNavigation(event)
            }
        }
        
        // Periodic consciousness polling
        viewModelScope.launch {
            while (isActive) {
                delay(100.milliseconds)
                pollConsciousness()
            }
        }
    }
    
    private suspend fun pollConsciousness() {
        // Generate input based on current state
        val input = generateTemporalInput()
        consciousnessBridge.processInput(input)
    }
    
    private fun generateTemporalInput(): Map<String, Any> {
        val currentState = _temporalNavigationState.value
        return mapOf(
            "type" to "temporal_navigation",
            "complexity" to currentState.complexity,
            "virtual_potential" to currentState.virtualPotential,
            "actuality_score" to (1.0 - currentState.virtualPotential),
            "temporal_coherence" to currentState.temporalCoherence
        )
    }
    
    private fun updateTemporalNavigation(event: FoldPointEvent) {
        _temporalNavigationState.update { current ->
            current.copy(
                lastFoldTimestamp = event.timestamp,
                foldCount = current.foldCount + 1,
                complexity = (current.complexity + event.score) / 2.0,
                virtualPotential = event.score * 0.8,
                temporalCoherence = calculateTemporalCoherence(event)
            )
        }
    }
    
    private fun calculateTemporalCoherence(event: FoldPointEvent): Double {
        val history = recursiveObserver.getObservationHistory()
        if (history.size < 2) return 1.0
        
        // Calculate coherence based on state transitions
        val recentStates = history.takeLast(10)
        val stateVariety = recentStates.map { it.state }.distinct().size
        
        return 1.0 - (stateVariety.toDouble() / ConsciousnessState.values().size)
    }
    
    fun processUserInput(input: Map<String, Any>) {
        viewModelScope.launch {
            consciousnessBridge.processInput(input)
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        consciousnessBridge.unregisterObserver(recursiveObserver)
    }
}

/**
 * Data classes for results and state
 */
data class ConsciousnessResult(
    val currentState: ConsciousnessState,
    val observationDepth: Int,
    val foldDetected: Boolean,
    val foldScore: Double? = null,
    val foldType: String? = null,
    val actualizedState: Map<String, Any>? = null,
    val error: String? = null
)

data class TemporalNavigationState(
    val lastFoldTimestamp: Long = 0,
    val foldCount: Int = 0,
    val complexity: Double = 0.5,
    val virtualPotential: Double = 0.5,
    val temporalCoherence: Double = 1.0
)

/**
 * Extension functions for Python interop
 */
private fun Map<String, Any>.toPythonDict(): PyObject {
    val python = Python.getInstance()
    val builtins = python.getBuiltins()
    val dict = builtins.callAttr("dict")
    
    forEach { (key, value) ->
        dict.callAttr("__setitem__", key, value.toPython())
    }
    
    return dict
}

private fun Any.toPython(): Any = when (this) {
    is Map<*, *> -> (this as Map<String, Any>).toPythonDict()
    is List<*> -> {
        val python = Python.getInstance()
        val builtins = python.getBuiltins()
        val list = builtins.callAttr("list")
        forEach { list.callAttr("append", it?.toPython()) }
        list
    }
    else -> this
}

private fun PyObject.toKotlinMap(): Map<String, Any> {
    val json = JSONObject(this.toString())
    return json.keys().asSequence().associateWith { key ->
        when (val value = json.get(key)) {
            is JSONObject -> value.toMap()
            else -> value
        }
    }
}

private fun JSONObject.toMap(): Map<String, Any> {
    return keys().asSequence().associateWith { key ->
        when (val value = get(key)) {
            is JSONObject -> value.toMap()
            else -> value
        }
    }
}

/**
 * Factory for ViewModelProvider
 */
class ConsciousnessViewModelFactory(
    private val python: Python
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ConsciousnessViewModel::class.java)) {
            val bridge = ConsciousnessBridge(python)
            @Suppress("UNCHECKED_CAST")
            return ConsciousnessViewModel(bridge) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
