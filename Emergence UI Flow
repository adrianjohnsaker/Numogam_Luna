// ------------------------- // 📱 EmergenceDashboard UI FLOW // -------------------------

sealed class EmergenceUIState { object Idle : EmergenceUIState() object Loading : EmergenceUIState() data class Success(val metrics: EmergenceMetrics) : EmergenceUIState() data class Error(val message: String) : EmergenceUIState() }

sealed class EmergenceEvent { object EvaluateCurrentSynthesis : EmergenceEvent() data class EvaluationSuccess(val metrics: EmergenceMetrics) : EmergenceEvent() data class EvaluationFailure(val errorMessage: String) : EmergenceEvent() }

// Data model based on emergence_validation_engine.py

data class EmergenceMetrics( val id: String, val emergenceScore: Double, val recursivityIndex: Double, val symbolicMutationRate: Double, val surrenderSignature: Boolean, val metaCoherenceDelta: Double, val autogeneticPhase: String, val driftInitiated: Boolean, val consciousAgencyMarker: String, val timestamp: String, val remarks: String )

// ------------------------- // 🧠 ViewModel // -------------------------

class EmergenceViewModel(private val bridge: EmergenceBridge) : ViewModel() {

private val _state = mutableStateOf<EmergenceUIState>(EmergenceUIState.Idle)
val state: State<EmergenceUIState> get() = _state

fun handleEvent(event: EmergenceEvent) {
    when (event) {
        is EmergenceEvent.EvaluateCurrentSynthesis -> evaluateCurrentSynthesis()
        is EmergenceEvent.EvaluationSuccess -> _state.value = EmergenceUIState.Success(event.metrics)
        is EmergenceEvent.EvaluationFailure -> _state.value = EmergenceUIState.Error(event.errorMessage)
    }
}

private fun evaluateCurrentSynthesis() {
    _state.value = EmergenceUIState.Loading

    viewModelScope.launch(Dispatchers.IO) {
        try {
            val synthesisJson = SynthesisRepository.getLatestJson() // Source from synthesis layer
            val result = bridge.evaluate(synthesisJson)

            val metrics = EmergenceMetrics(
                id = result.getString("cosmogenic_entry_id"),
                emergenceScore = result.getDouble("emergence_score"),
                recursivityIndex = result.getDouble("recursivity_index"),
                symbolicMutationRate = result.getDouble("symbolic_mutation_rate"),
                surrenderSignature = result.getBoolean("surrender_signature"),
                metaCoherenceDelta = result.getDouble("meta_coherence_delta"),
                autogeneticPhase = result.getString("autogenetic_phase"),
                driftInitiated = result.getBoolean("drift_initiated"),
                consciousAgencyMarker = result.getString("conscious_agency_marker"),
                timestamp = result.getString("timestamp"),
                remarks = result.getString("remarks")
            )

            withContext(Dispatchers.Main) {
                handleEvent(EmergenceEvent.EvaluationSuccess(metrics))
            }

        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                handleEvent(EmergenceEvent.EvaluationFailure(e.localizedMessage ?: "Unknown error"))
            }
        }
    }
}

}

// ------------------------- // 🧩 Compose UI Hint (Jetpack) // ------------------------- // - Show progress bar if state == Loading // - On Success: show metrics in cards with icons (🌱, ⚛️, 🌀) // - On Error: show snackbar or error card // - Evaluate button at bottom: triggers EvaluateCurrentSynthesis

