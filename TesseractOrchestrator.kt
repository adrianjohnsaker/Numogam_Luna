class TesseractOrchestrator(context: Context) {
    private val bridge: TesseractBridge = ChaquoTesseractBridge(context)
    private val errorHandler = TesseractErrorHandler()
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val _state = MutableStateFlow<HyperstructureState?>(null)
    private val _errors = MutableSharedFlow<String>()

    val state: StateFlow<HyperstructureState?> = _state
    val errors: SharedFlow<String> = _errors

    fun initialize(dimensions: Int = 4, seed: String? = null) {
        scope.launch {
            try {
                bridge.initialize(dimensions, seed)
                preloadCommonPatterns()
                _state.value = bridge.getCurrentState()
            } catch (ex: Exception) {
                _errors.emit(errorHandler.handleInitializationError(ex))
            }
        }
    }

    fun processInput(
        symbolicMap: SymbolicMap,
        driftPattern: TemporalDriftPattern,
        affectSignal: AffectSignal
    ): Job = scope.launch {
        _state.value = _state.value?.copy(processing = true)
        try {
            bridge.receiveInput(symbolicMap, driftPattern, affectSignal)
            val result = bridge.process()
            _state.emit(result)
        } catch (ex: Exception) {
            _errors.emit(errorHandler.handleProcessingError(ex))
        } finally {
            _state.value = _state.value?.copy(processing = false)
        }
    }

    private suspend fun preloadCommonPatterns() {
        val patterns = listOf("archetype_formation", "temporal_convergence")
        patterns.forEach { pattern ->
            bridge.configure(mapOf("preload_pattern" to pattern))
        }
    }

    fun resetSystem(preserveHistory: Boolean = false) {
        scope.launch {
            bridge.reset(preserveHistory)
            _state.value = bridge.getCurrentState()
        }
    }

    fun shutdown() {
        scope.cancel()
    }
}
