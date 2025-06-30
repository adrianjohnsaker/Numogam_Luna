package com.antonio.my.ai.girlfriend.free.amelia.ui.event

import org.json.JSONArray

sealed class Phase5UiEvent {
    object InitializePhase5 : Phase5UiEvent()

    data class SubmitSymbolicDreamInput(
        val symbols: JSONArray
    ) : Phase5UiEvent()

    object TriggerMythogenicProcessing : Phase5UiEvent()

    object GenerateDreamNarrative : Phase5UiEvent()

    object TraceInterdreamEvolution : Phase5UiEvent()

    object ResetPhase : Phase5UiEvent()
}
