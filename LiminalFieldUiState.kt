sealed class LiminalFieldUiState {
    object Idle : LiminalFieldUiState()
    object Loading : LiminalFieldUiState()
    data class FieldReady(val glyphs: JSONArray) : LiminalFieldUiState()
    data class Error(val message: String) : LiminalFieldUiState()
}
