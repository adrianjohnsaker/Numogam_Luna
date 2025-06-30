package com.antonio.my.ai.girlfriend.free.amelia.ui.state

import org.json.JSONArray
import org.json.JSONObject

sealed class Phase5UiState {
    object Idle : Phase5UiState()

    object Loading : Phase5UiState()

    data class SymbolicInputReceived(
        val symbols: JSONArray
    ) : Phase5UiState()

    data class MythogenicOutputReady(
        val mythogenicData: JSONObject
    ) : Phase5UiState()

    data class NarrativeGenerated(
        val narrativeText: String,
        val mythAxis: String,
        val dreamId: String
    ) : Phase5UiState()

    data class InterdreamEvolutionAvailable(
        val lineageTrace: JSONObject
    ) : Phase5UiState()

    data class Error(
        val message: String
    ) : Phase5UiState()
}
