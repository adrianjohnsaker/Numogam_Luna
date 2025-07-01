// -------------------------
// 📱 EmergenceDashboard UI FLOW
// -------------------------

sealed class EmergenceUIState {
    object Idle : EmergenceUIState()
    object Loading : EmergenceUIState()
    data class Success(val metrics: EmergenceMetrics) : EmergenceUIState()
    data class Error(val message: String) : EmergenceUIState()
}

sealed class EmergenceEvent {
    object EvaluateCurrentSynthesis : EmergenceEvent()
    data class EvaluationSuccess(val metrics: EmergenceMetrics) : EmergenceEvent()
    data class EvaluationFailure(val errorMessage: String) : EmergenceEvent()
}

// Data model based on emergence_validation_engine.py

data class EmergenceMetrics(
    val id: String,
    val emergenceScore: Double,
    val recursivityIndex: Double,
    val symbolicMutationRate: Double,
    val surrenderSignature: Boolean,
    val metaCoherenceDelta: Double,
    val autogeneticPhase: String,
    val driftInitiated: Boolean,
    val consciousAgencyMarker: String,
    val timestamp: String,
    val remarks: String
)
