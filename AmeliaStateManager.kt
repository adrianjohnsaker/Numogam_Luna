// AmeliaStateManager.kt
package com.antonio.my.ai.girlfriend.free.amelia

import com.chaquo.python.Python
import com.chaquo.python.PyObject
import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

@Serializable
data class AmeliaParameters(
    @SerialName("epsilon_exploration") val epsilonExploration: Double,
    @SerialName("creative_bias") val creativeBias: Double,
    @SerialName("attention_span") val attentionSpan: Double,
    @SerialName("memory_retention") val memoryRetention: Double,
    @SerialName("risk_tolerance") val riskTolerance: Double,
    @SerialName("curiosity_drive") val curiosityDrive: Double,
    @SerialName("tool_affinities") val toolAffinities: Map<String, Double>,
    @SerialName("attention_weights") val attentionWeights: Map<String, Double>
)

@Serializable
data class CreativityMetrics(
    @SerialName("current_creativity") val currentCreativity: Double,
    @SerialName("novelty_score") val noveltyScore: Double,
    @SerialName("coherence_score") val coherenceScore: Double,
    @SerialName("risk_taking") val riskTaking: Double
)

@Serializable
data class MutationHistory(
    @SerialName("total_mutations") val totalMutations: Int,
    @SerialName("successful_mutations") val successfulMutations: Int,
    @SerialName("recent_mutations") val recentMutations: List<MutationCommand>
)

@Serializable
data class MutationCommand(
    val id: String,
    val timestamp: String,
    @SerialName("mutation_type") val mutationType: String,
    @SerialName("target_parameter") val targetParameter: String,
    @SerialName("old_value") val oldValue: Double,
    @SerialName("new_value") val newValue: Double,
    val reason: String,
    @SerialName("expected_outcome") val expectedOutcome: String
)

@Serializable
data class MutationResult(
    @SerialName("command_id") val commandId: String,
    val success: Boolean,
    @SerialName("actual_change") val actualChange: Double,
    @SerialName("observed_effects") val observedEffects: List<String>,
    @SerialName("creativity_impact") val creativityImpact: Double,
    val timestamp: String,
    val error: String? = null
)

@Serializable
data class AmeliaState(
    val parameters: AmeliaParameters,
    @SerialName("creativity_metrics") val creativityMetrics: CreativityMetrics,
    @SerialName("mutation_history") val mutationHistory: MutationHistory
)

@Serializable
data class MutationSuggestion(
    val type: String,
    val target: String,
    @SerialName("suggested_value") val suggestedValue: Double,
    val reason: String
)

enum class MutationType(val value: String) {
    PARAMETER_ADJUST("parameter_adjust"),
    TOOL_AFFINITY("tool_affinity"),
    CREATIVE_BIAS("creative_bias"),
    EXPLORATION_RATE("exploration_rate"),
    MEMORY_WEIGHT("memory_weight"),
    ATTENTION_FOCUS("attention_focus")
}

class AmeliaStateManager : ViewModel() {
    
    private val python = Python.getInstance()
    private var ameliaInterface: PyObject? = null
    
    private val _currentState = MutableStateFlow<AmeliaState?>(null)
    val currentState: StateFlow<AmeliaState?> = _currentState.asStateFlow()
    
    private val _mutationResults = MutableStateFlow<List<MutationResult>>(emptyList())
    val mutationResults: StateFlow<List<MutationResult>> = _mutationResults.asStateFlow()
    
    private val _suggestions = MutableStateFlow<List<MutationSuggestion>>(emptyList())
    val suggestions: StateFlow<List<MutationSuggestion>> = _suggestions.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }
    
    companion object {
        private const val TAG = "AmeliaStateManager"
    }
    
    init {
        initializePython()
        refreshState()
    }
    
    private fun initializePython() {
        try {
            // Import the Python module
            val ameliaModule = python.getModule("amelia_state_mutation")
            ameliaInterface = ameliaModule.callAttr("AmeliaStateMutationInterface")
            Log.d(TAG, "Python interface initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python interface", e)
        }
    }
    
    fun refreshState() {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val stateJson = ameliaInterface?.callAttr("get_state_json")?.toString()
                if (stateJson != null) {
                    val state = json.decodeFromString<AmeliaState>(stateJson)
                    _currentState.value = state
                    Log.d(TAG, "State refreshed: creativity=${state.creativityMetrics.currentCreativity}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to refresh state", e)
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    fun executeMutation(
        mutationType: MutationType,
        targetParameter: String,
        newValue: Double,
        reason: String,
        expectedOutcome: String
    ) {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val resultJson = ameliaInterface?.callAttr(
                    "execute_mutation_from_android",
                    mutationType.value,
                    targetParameter,
                    newValue,
                    reason,
                    expectedOutcome
                )?.toString()
                
                if (resultJson != null) {
                    val result = json.decodeFromString<MutationResult>(resultJson)
                    
                    // Add to results history
                    val currentResults = _mutationResults.value.toMutableList()
                    currentResults.add(0, result) // Add to front
                    if (currentResults.size > 50) {
                        currentResults.removeAt(currentResults.size - 1) // Keep max 50
                    }
                    _mutationResults.value = currentResults
                    
                    Log.d(TAG, "Mutation executed: ${result.success}, effects: ${result.observedEffects}")
                    
                    // Refresh state after mutation
                    refreshState()
                    refreshSuggestions()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to execute mutation", e)
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    fun refreshSuggestions() {
        viewModelScope.launch {
            try {
                val suggestionsJson = ameliaInterface?.callAttr("get_mutation_suggestions_json")?.toString()
                if (suggestionsJson != null) {
                    val suggestions = json.decodeFromString<List<MutationSuggestion>>(suggestionsJson)
                    _suggestions.value = suggestions
                    Log.d(TAG, "Suggestions refreshed: ${suggestions.size} available")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to refresh suggestions", e)
            }
        }
    }
    
    fun resetParameters() {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                ameliaInterface?.callAttr("reset_parameters")
                _mutationResults.value = emptyList()
                refreshState()
                refreshSuggestions()
                Log.d(TAG, "Parameters reset successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to reset parameters", e)
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    // Convenience methods for common mutations
    fun boostCreativity(amount: Double = 0.1) {
        val currentBias = _currentState.value?.parameters?.creativeBias ?: 0.5
        executeMutation(
            MutationType.CREATIVE_BIAS,
            "creative_bias",
            (currentBias + amount).coerceIn(0.0, 1.0),
            "User requested creativity boost",
            "Enhanced creative output generation"
        )
    }
    
    fun increaseExploration(amount: Double = 0.05) {
        val currentExploration = _currentState.value?.parameters?.epsilonExploration ?: 0.08
        executeMutation(
            MutationType.EXPLORATION_RATE,
            "epsilon_exploration",
            (currentExploration + amount).coerceIn(0.01, 0.5),
            "User wants more experimental behavior",
            "More diverse and novel responses"
        )
    }
    
    fun adjustRiskTolerance(newValue: Double) {
        executeMutation(
            MutationType.PARAMETER_ADJUST,
            "risk_tolerance",
            newValue.coerceIn(0.0, 1.0),
            "User adjusted risk tolerance preference",
            "Modified willingness to take creative risks"
        )
    }
    
    fun enhanceToolAffinity(tool: String, newAffinity: Double) {
        executeMutation(
            MutationType.TOOL_AFFINITY,
            tool,
            newAffinity.coerceIn(0.0, 1.0),
            "User wants to prioritize $tool usage",
            "Increased utilization of $tool capabilities"
        )
    }
}

// MainActivity integration example
class MainActivity : ComponentActivity() {
    
    private lateinit var stateManager: AmeliaStateManager
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Chaquopy
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        stateManager = ViewModelProvider(this)[AmeliaStateManager::class.java]
        
        setContent {
            AmeliaTheme {
                AmeliaStateMutationScreen(stateManager)
            }
        }
