class MetaphysicsViewModel(application: Application) : AndroidViewModel(application) {
    private val orchestrator = TesseractOrchestrator(application)
    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    private val _dimensionalVectors = MutableStateFlow<Map<String, List<Double>>>(emptyMap())
    
    val uiState: StateFlow<UiState> = _uiState
    val dimensionalVectors: StateFlow<Map<String, List<Double>>> = _dimensionalVectors

    init {
        orchestrator.initialize()
        observeState()
    }

    fun processInput(symbolicInput: Map<String, Any>) {
        viewModelScope.launch {
            _uiState.value = UiState.Processing
            
            val symbolicMap = createSymbolicMap(symbolicInput)
            val driftPattern = analyzeTemporalContext()
            val affectSignal = calculateAffectiveResponse()

            orchestrator.processInput(symbolicMap, driftPattern, affectSignal)
        }
    }

    private fun observeState() {
        viewModelScope.launch {
            orchestrator.state.collect { state ->
                state?.let {
                    _dimensionalVectors.value = it.dimensionalVectors
                    updateUiState(it)
                }
            }
        }
        
        viewModelScope.launch {
            orchestrator.errors.collect { error ->
                _uiState.value = UiState.Error(error)
            }
        }
    }

    private fun updateUiState(state: HyperstructureState) {
        _uiState.value = when {
            state.stabilityCoefficient < 0.4 -> UiState.Unstable(state)
            state.integrationDensity > 0.7 -> UiState.HighIntegration(state)
            else -> UiState.Stable(state)
        }
    }

    override fun onCleared() {
        orchestrator.shutdown()
        super.onCleared()
    }

    sealed class UiState {
        object Idle : UiState()
        object Processing : UiState()
        data class Stable(val state: HyperstructureState) : UiState()
        data class HighIntegration(val state: HyperstructureState) : UiState()
        data class Unstable(val state: HyperstructureState) : UiState()
        data class Error(val message: String) : UiState()
    }
}
