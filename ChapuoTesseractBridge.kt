class ChaquoTesseractBridge(context: Context) : TesseractBridge {
    private val python: Python = Python.getInstance()
    private var instance: PyObject? = null
    private val gson = GsonBuilder().registerTypeAdapterFactory(
        RuntimeTypeAdapterFactory.of(SymbolicMap::class.java)
            .registerSubtype(SymbolicMap.NumericMap::class.java)
            .registerSubtype(SymbolicMap.TemporalMap::class.java)
            .registerSubtype(SymbolicMap.CompositeMap::class.java)
    ).create()
    private val lock = Mutex()

    override suspend fun initialize(dimensionCount: Int, initializationSeed: String?) {
        withContext(Dispatchers.Default) {
            lock.withLock {
                val pyClass = python.getModule("tesseract_hyperstructure")
                    .get("TesseractHyperstructure")
                
                instance = if (initializationSeed != null) {
                    pyClass.call(dimensionCount, initializationSeed)
                } else {
                    pyClass.call(dimensionCount)
                }
            }
        }
    }

    override suspend fun receiveInput(
        symbolicMap: SymbolicMap,
        driftPattern: TemporalDriftPattern,
        affectSignal: AffectSignal
    ) {
        withContext(Dispatchers.Default) {
            lock.withLock {
                instance?.callAttr(
                    "receive_input",
                    convertToPython(symbolicMap),
                    convertToPython(driftPattern),
                    convertToPython(affectSignal)
                )
            }
        }
    }

    override suspend fun process(): HyperstructureState = withContext(Dispatchers.Default) {
        lock.withLock {
            val result = instance?.callAttr("process")?.toString()
            gson.fromJson(result, HyperstructureState::class.java)
        }
    }

    override suspend fun getCurrentState(): HyperstructureState = withContext(Dispatchers.Default) {
        lock.withLock {
            val result = instance?.callAttr("get_current_state")?.toString()
            gson.fromJson(result, HyperstructureState::class.java)
        }
    }

    override suspend fun reset(preserveHistory: Boolean) {
        withContext(Dispatchers.Default) {
            lock.withLock {
                instance?.callAttr("reset", preserveHistory)
            }
        }
    }

    override suspend fun configure(parameters: Map<String, Any>) {
        withContext(Dispatchers.Default) {
            lock.withLock {
                parameters.forEach { (key, value) ->
                    instance?.callAttr("configure", key, convertToPython(value))
                }
            }
        }
    }

    private fun convertToPython(data: Any): PyObject {
        return python.getBuiltins().callAttr(
            "eval", 
            gson.toJson(data),
            python.getModule("json").callAttr("loads")
        )
    }
}
