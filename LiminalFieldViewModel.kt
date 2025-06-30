class LiminalFieldViewModel(context: Context) : ViewModel() {

    private val bridge = LiminalFieldBridge(context)
    private val _state = MutableStateFlow<LiminalFieldUiState>(LiminalFieldUiState.Idle)
    val state: StateFlow<LiminalFieldUiState> = _state

    fun onEvent(event: LiminalFieldUiEvent) {
        when (event) {
            LiminalFieldUiEvent.GenerateField -> {
                _state.value = LiminalFieldUiState.Loading
                viewModelScope.launch {
                    val result = bridge.generateLiminalField()
                    _state.value = if (result.length() == 0 || result.toString().contains("error")) {
                        LiminalFieldUiState.Error("Could not generate liminal field.")
                    } else {
                        LiminalFieldUiState.FieldReady(result)
                    }
                }
            }

            LiminalFieldUiEvent.Reset -> {
                _state.value = LiminalFieldUiState.Idle
            }
        }
    }
}
