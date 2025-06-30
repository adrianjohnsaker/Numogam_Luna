package com.antonio.my.ai.girlfriend.free.amelia.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.amelia.bridge.phase5.MorphogenesisAnimator
import com.amelia.ui.event.Phase5UiEvent
import com.amelia.ui.state.Phase5UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import android.content.Context

class Phase5ViewModel(private val context: Context) : ViewModel() {

    private val _uiState = MutableStateFlow<Phase5UiState>(Phase5UiState.Idle)
    val uiState: StateFlow<Phase5UiState> = _uiState

    private val mythEngine = MorphogenesisAnimator(context)

    private var cachedSymbols: JSONArray? = null
    private var mythResult: JSONObject? = null

    fun onEvent(event: Phase5UiEvent) {
        when (event) {
            is Phase5UiEvent.InitializePhase5 -> {
                _uiState.value = Phase5UiState.Idle
            }

            is Phase5UiEvent.SubmitSymbolicDreamInput -> {
                cachedSymbols = event.symbols
                _uiState.value = Phase5UiState.SymbolicInputReceived(event.symbols)
            }

            is Phase5UiEvent.TriggerMythogenicProcessing -> {
                cachedSymbols?.let { symbols ->
                    _uiState.value = Phase5UiState.Loading
                    viewModelScope.launch {
                        val result = mythEngine.generateMythogenicStructure(symbols)
                        mythResult = result
                        if (result.has("error")) {
                            _uiState.value = Phase5UiState.Error(result.getString("error"))
                        } else {
                            _uiState.value = Phase5UiState.MythogenicOutputReady(result)
                        }
                    }
                }
            }

            is Phase5UiEvent.GenerateDreamNarrative -> {
                mythResult?.let { result ->
                    val axis = result.optString("axis")
                    val dreamId = result.optString("myth_id")
                    val narrative = generateNarrativeFromTemplate(result)
                    _uiState.value = Phase5UiState.NarrativeGenerated(
                        narrativeText = narrative,
                        mythAxis = axis,
                        dreamId = dreamId
                    )
                }
            }

            is Phase5UiEvent.TraceInterdreamEvolution -> {
                // Placeholder logic
                val trace = JSONObject().apply {
                    put("evolution_map", "coming soon")
                }
                _uiState.value = Phase5UiState.InterdreamEvolutionAvailable(trace)
            }

            is Phase5UiEvent.ResetPhase -> {
                cachedSymbols = null
                mythResult = null
                _uiState.value = Phase5UiState.Idle
            }
        }
    }

    private fun generateNarrativeFromTemplate(result: JSONObject): String {
        val template = result.optJSONObject("template")?.optString("structure") ?: "Unknown"
        val threads = result.optJSONArray("thematic_threads") ?: JSONArray()

        val builder = StringBuilder()
        builder.append("Dream Narrative (").append(template).append("):\n\n")

        for (i in 0 until threads.length()) {
            val thread = threads.getJSONObject(i)
            val stage = thread.optString("stage")
            val symbol = thread.optString("symbol")
            val meaning = thread.optString("meaning")
            builder.append("→ $stage: The symbol '$symbol' conveys $meaning.\n")
        }

        return builder.toString()
    }
}
