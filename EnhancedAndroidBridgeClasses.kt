package com.antonio.my.ai.girlfriend.free.amelia.android.bridges

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import android.content.Context
import android.util.Log
import com.amelia.android.models.*
import com.amelia.android.utils.SettingsManager
import java.util.concurrent.ConcurrentHashMap

/**
 * Enhanced bridge for the advanced symbolic dream mapper with neuro-symbolic AI
 */
class EnhancedSymbolicDreamMapperBridge(private val context: Context) {
    
    private var python: Python? = null
    private var mapperModule: PyObject? = null
    private var isInitialized = false
    private val settingsManager = SettingsManager(context)
    
    // Cache for frequently used objects
    private val pythonObjectCache = ConcurrentHashMap<String, PyObject>()
    
    companion object {
        private const val TAG = "EnhancedSymbolicMapper"
        private const val MODULE_NAME = "enhanced_symbolic_dream_mapper"
        private const val MAPPER_CLASS = "EnhancedSymbolicDreamMapper"
    }
    
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            
            python = Python.getInstance()
            
            // Import the enhanced mapper module
            val module = python?.getModule(MODULE_NAME)
            if (module == null) {
                Log.e(TAG, "Failed to import $MODULE_NAME module")
                return@withContext false
            }
            
            // Create mapper instance
            val mapperClass = module.get(MAPPER_CLASS)
            mapperModule = mapperClass?.call()
            
            if (mapperModule == null) {
                Log.e(TAG, "Failed to create mapper instance")
                return@withContext false
            }
            
            // Cache the module for reuse
            pythonObjectCache[MAPPER_CLASS] = mapperModule!!
            
            isInitialized = true
            Log.d(TAG, "Enhanced Symbolic Dream Mapper initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Enhanced Symbolic Dream Mapper", e)
            false
        }
    }
    
    suspend fun analyzeDreamText(
        dreamText: String,
        culturalContext: List<String> = listOf("western"),
        analysisDepth: AnalysisDepth = AnalysisDepth.MODERATE
    ): DreamAnalysisResult = withContext(Dispatchers.IO) {
        
        if (!isInitialized || mapperModule == null) {
            throw IllegalStateException("Mapper not initialized")
        }
        
        try {
            // Convert cultural context to Python list
            val pythonCulturalContext = python?.builtins?.callAttr("list", culturalContext)
            
            // Call analyze_dream_text method
            val result = mapperModule!!.callAttr(
                "analyze_dream_text",
                dreamText,
                pythonCulturalContext
            )
            
            // Convert Python result to Kotlin objects
            parseAnalysisResult(result)
            
        } catch (e: Exception) {
            Log.e(TAG, "Dream analysis failed", e)
            DreamAnalysisResult.error("Analysis failed: ${e.message}")
        }
    }
    
    suspend fun findSymbolicPatterns(symbols: List<SymbolMapping>): List<SymbolicPattern> = 
        withContext(Dispatchers.IO) {
            
        if (!isInitialized || mapperModule == null) {
            return@withContext emptyList()
        }
        
        try {
            // Convert symbols to Python format
            val pythonSymbols = convertSymbolsToPython(symbols)
            
            // Get pattern engine and analyze patterns
            val patternEngine = mapperModule!!.getAttr("pattern_engine")
            val patternsResult = patternEngine.callAttr("analyze_patterns", pythonSymbols)
            
            // Convert results back to Kotlin
            parseSymbolicPatterns(patternsResult)
            
        } catch (e: Exception) {
            Log.e(TAG, "Pattern analysis failed", e)
            emptyList()
        }
    }
    
    suspend fun calculateSymbolSimilarity(symbol1: String, symbol2: String): Float = 
        withContext(Dispatchers.IO) {
            
        if (!isInitialized || mapperModule == null) {
            return@withContext 0.0f
        }
        
        try {
            // Get VSA instance
            val vsa = mapperModule!!.getAttr("vsa")
            
            // Check if symbols exist in vector space
            val symbolVectors = vsa.getAttr("symbol_vectors")
            if (!symbolVectors.callAttr("__contains__", symbol1).toBoolean() ||
                !symbolVectors.callAttr("__contains__", symbol2).toBoolean()) {
                return@withContext 0.0f
            }
            
            // Get vectors and calculate similarity
            val vec1 = symbolVectors.callAttr("__getitem__", symbol1)
            val vec2 = symbolVectors.callAttr("__getitem__", symbol2)
            
            val similarity = vsa.callAttr("calculate_similarity", vec1, vec2)
            similarity.toFloat()
            
        } catch (e: Exception) {
            Log.e(TAG, "Similarity calculation failed", e)
            0.0f
        }
    }
    
    suspend fun findSymbolAnalogies(
        symbolA: String,
        symbolB: String,
        symbolC: String
    ): List<SymbolAnalogy> = withContext(Dispatchers.IO) {
        
        if (!isInitialized || mapperModule == null) {
            return@withContext emptyList()
        }
        
        try {
            val vsa = mapperModule!!.getAttr("vsa")
            val analogiesResult = vsa.callAttr("find_analogies", symbolA, symbolB, symbolC)
            
            parseSymbolAnalogies(analogiesResult)
            
        } catch (e: Exception) {
            Log.e(TAG, "Analogy search failed", e)
            emptyList()
        }
    }
    
    suspend fun analyzeTransformationVectors(
        dreamText: String,
        symbols: List<SymbolMapping>
    ): Map<DeterritorializedVector, TransformationAnalysis> = withContext(Dispatchers.IO) {
        
        if (!isInitialized || mapperModule == null) {
            return@withContext emptyMap()
        }
        
        try {
            // Convert symbols to Python format
            val pythonSymbols = convertSymbolsToPython(symbols)
            
            // Analyze deterritorialization vectors
            val result = mapperModule!!.callAttr(
                "_analyze_deterritorialization_vectors",
                pythonSymbols,
                dreamText
            )
            
            parseTransformationAnalysis(result)
            
        } catch (e: Exception) {
            Log.e(TAG, "Transformation analysis failed", e)
            emptyMap()
        }
    }
    
    suspend fun generateTransformationScenarios(
        symbols: List<SymbolMapping>,
        patterns: List<SymbolicPattern>,
        transformationIntensity: Float = 0.5f
    ): List<TransformationScenario> = withContext(Dispatchers.IO) {
        
        if (!isInitialized || mapperModule == null) {
            return@withContext emptyList()
        }
        
        try {
            val pythonSymbols = convertSymbolsToPython(symbols)
            val pythonPatterns = convertPatternsToPython(patterns)
            
            // Create mock deterritorialization analysis for the method
            val mockAnalysis = python?.builtins?.callAttr("dict")
            
            val scenarios = mapperModule!!.callAttr(
                "_generate_transformation_scenarios",
                pythonSymbols,
                pythonPatterns,
                mockAnalysis
            )
            
            parseTransformationScenarios(scenarios)
            
        } catch (e: Exception) {
            Log.e(TAG, "Scenario generation failed", e)
            emptyList()
        }
    }
    
    suspend fun calculateFieldCoherence(
        symbols: List<SymbolMapping>,
        patterns: List<SymbolicPattern>
    ): FieldCoherenceMetrics = withContext(Dispatchers.IO) {
        
        if (!isInitialized || mapperModule == null) {
            return@withContext FieldCoherenceMetrics()
        }
        
        try {
            val pythonSymbols = convertSymbolsToPython(symbols)
            val pythonPatterns = convertPatternsToPython(patterns)
            
            val coherenceResult = mapperModule!!.callAttr(
                "_calculate_field_coherence",
                pythonSymbols,
                pythonPatterns
            )
            
            parseFieldCoherence(coherenceResult)
            
        } catch (e: Exception) {
            Log.e(TAG, "Field coherence calculation failed", e)
            FieldCoherenceMetrics()
        }
    }
    
    suspend fun learnFromFeedback(
        symbolMappings: List<SymbolMapping>,
        patterns: List<SymbolicPattern>,
        userFeedback: Map<String, Float>
    ): Boolean = withContext(Dispatchers.IO) {
        
        if (!isInitialized || mapperModule == null) {
            return@withContext false
        }
        
        try {
            val patternEngine = mapperModule!!.getAttr("pattern_engine")
            
            // Apply learning for each pattern with feedback
            for (pattern in patterns) {
                val patternFeedback = userFeedback[pattern.id] ?: 0.5f
                val pythonPattern = convertPatternToPython(pattern)
                
                patternEngine.callAttr("learn_pattern", pythonPattern, patternFeedback)
            }
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Learning from feedback failed", e)
            false
        }
    }
    
    // Helper methods for parsing Python results
    
    private fun parseAnalysisResult(pyResult: PyObject): DreamAnalysisResult {
        try {
            val resultDict = pyResult.asMap()
            
            // Check for error
            if (resultDict.containsKey("error")) {
                return DreamAnalysisResult.error(resultDict["error"].toString())
            }
            
            // Parse symbols
            val symbols = mutableListOf<SymbolMapping>()
            val symbolsArray = resultDict["symbols"] as? PyObject
            symbolsArray?.let { parseSymbolMappings(it, symbols) }
            
            // Parse patterns
            val patterns = mutableListOf<SymbolicPattern>()
            val patternsArray = resultDict["patterns"] as? PyObject
            patternsArray?.let { parseSymbolicPatterns(it, patterns) }
            
            // Parse connections
            val connections = mutableListOf<SymbolicConnection>()
            val connectionsArray = resultDict["connections"] as? PyObject
            connectionsArray?.let { parseSymbolicConnections(it, connections) }
            
            // Parse field coherence
            val fieldCoherence = resultDict["field_coherence"] as? PyObject
            val coherenceMetrics = fieldCoherence?.let { parseFieldCoherence(it) } ?: FieldCoherenceMetrics()
            
            // Parse neuro-symbolic insights
            val insights = resultDict["neuro_symbolic_insights"] as? PyObject
            val neuralInsights = insights?.let { parseNeuralInsights(it) } ?: NeuralSymbolicInsights()
            
            return DreamAnalysisResult.success(
                symbols = symbols,
                patterns = patterns,
                connections = connections,
                fieldCoherence = coherenceMetrics,
                neuralInsights = neuralInsights,
                complexityMeasure = (resultDict["complexity_measure"] as? PyObject)?.toFloat() ?: 0.0f
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse analysis result", e)
            return DreamAnalysisResult.error("Failed to parse analysis result: ${e.message}")
        }
    }
    
    private fun parseSymbolMappings(pyArray: PyObject, symbols: MutableList<SymbolMapping>) {
        try {
            val arrayList = pyArray.asList()
            for (item in arrayList) {
                val symbolDict = item.asMap()
                
                val symbol = SymbolMapping(
                    symbol = symbolDict["symbol"]?.toString() ?: "",
                    meaning = symbolDict["meaning"]?.toString() ?: "",
                    symbolType = parseSymbolType(symbolDict["symbol_type"]?.toString()),
                    confidence = symbolDict["confidence"]?.toFloat() ?: 0.0f,
                    contextualRelevance = symbolDict.get("personal_resonance")?.toFloat() ?: 0.0f,
                    archetypeConnection = null, // Would need more parsing
                    emotionalResonance = null,
                    culturalSignificance = null,
                    personalAssociation = null,
                    sourceText = "" // Would need to be passed separately
                )
                
                symbols.add(symbol)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse symbol mappings", e)
        }
    }
    
    private fun parseSymbolicPatterns(pyArray: PyObject): List<SymbolicPattern> {
        val patterns = mutableListOf<SymbolicPattern>()
        
        try {
            val arrayList = pyArray.asList()
            for (item in arrayList) {
                parseSymbolicPatterns(item, patterns)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse symbolic patterns", e)
        }
        
        return patterns
    }
    
    private fun parseSymbolicPatterns(pyArray: PyObject, patterns: MutableList<SymbolicPattern>) {
        try {
            val arrayList = pyArray.asList()
            for (item in arrayList) {
                val patternDict = item.asMap()
                
                val pattern = SymbolicPattern(
                    id = patternDict["pattern_id"]?.toString() ?: "",
                    patternType = parsePatternType(patternDict["pattern_type"]?.toString()),
                    elements = parseStringList(patternDict["elements"]),
                    coherenceScore = patternDict["coherence_score"]?.toFloat() ?: 0.0f,
                    emergenceProbability = patternDict["emergence_probability"]?.toFloat() ?: 0.0f,
                    complexityMeasure = patternDict["complexity_measure"]?.toFloat() ?: 0.0f,
                    archetypalBasis = parseStringList(patternDict["archetypal_basis"])
                )
                
                patterns.add(pattern)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse symbolic patterns", e)
        }
    }
    
    private fun parseSymbolicConnections(pyArray: PyObject, connections: MutableList<SymbolicConnection>) {
        try {
            val arrayList = pyArray.asList()
            for (item in arrayList) {
                val connDict = item.asMap()
                
                val connection = SymbolicConnection(
                    symbol1 = connDict["symbol1"]?.toString() ?: "",
                    symbol2 = connDict["symbol2"]?.toString() ?: "",
                    strength = connDict["strength"]?.toFloat() ?: 0.0f,
                    connectionType = parseConnectionType(connDict["types"]),
                    semanticSimilarity = connDict["vector_similarity"]?.toFloat() ?: 0.0f
                )
                
                connections.add(connection)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse symbolic connections", e)
        }
    }
    
    private fun parseFieldCoherence(pyObject: PyObject): FieldCoherenceMetrics {
        return try {
            val coherenceDict = pyObject.asMap()
            
            FieldCoherenceMetrics(
                overallCoherence = coherenceDict["overall_coherence"]?.toFloat() ?: 0.0f,
                symbolCoherence = coherenceDict["symbol_coherence"]?.toFloat() ?: 0.0f,
                patternCoherence = coherenceDict["pattern_coherence"]?.toFloat() ?: 0.0f,
                networkCoherence = coherenceDict["network_coherence"]?.toFloat() ?: 0.0f,
                archetypalCoherence = coherenceDict["archetypal_coherence"]?.toFloat() ?: 0.0f,
                entropyMeasure = coherenceDict["entropy_measure"]?.toFloat() ?: 0.0f,
                complexityMeasure = coherenceDict["complexity_measure"]?.toFloat() ?: 0.0f
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse field coherence", e)
            FieldCoherenceMetrics()
        }
    }
    
    private fun parseNeuralInsights(pyObject: PyObject): NeuralSymbolicInsights {
        return try {
            val insightsDict = pyObject.asMap()
            
            NeuralSymbolicInsights(
                dominantArchetypes = parseStringList(insightsDict["dominant_archetypes"]),
                symbolicDensity = insightsDict["symbolic_density"]?.toInt() ?: 0,
                patternEmergence = insightsDict["pattern_emergence"]?.toInt() ?: 0,
                personalResonanceStrength = insightsDict["personal_resonance_strength"]?.toFloat() ?: 0.0f,
                transformationReadiness = parseTransformationReadiness(insightsDict["transformation_readiness"]),
                culturalInfluence = parseCulturalInfluence(insightsDict["cultural_influence"])
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse neural insights", e)
            NeuralSymbolicInsights()
        }
    }
    
    private fun parseSymbolAnalogies(pyArray: PyObject): List<SymbolAnalogy> {
        val analogies = mutableListOf<SymbolAnalogy>()
        
        try {
            val arrayList = pyArray.asList()
            for (item in arrayList) {
                val analogyTuple = item.asList()
                if (analogyTuple.size >= 2) {
                    analogies.add(
                        SymbolAnalogy(
                            symbol = analogyTuple[0].toString(),
                            similarity = analogyTuple[1].toFloat()
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse symbol analogies", e)
        }
        
        return analogies
    }
    
    private fun parseTransformationAnalysis(pyObject: PyObject): Map<DeterritorializedVector, TransformationAnalysis> {
        val analysis = mutableMapOf<DeterritorializedVector, TransformationAnalysis>()
        
        try {
            val analysisDict = pyObject.asMap()
            
            for ((key, value) in analysisDict) {
                val vector = parseDeterritorializedVector(key.toString())
                val valueDict = (value as PyObject).asMap()
                
                val transformationAnalysis = TransformationAnalysis(
                    activationStrength = valueDict["activation_strength"]?.toFloat() ?: 0.0f,
                    triggerMatches = valueDict["trigger_matches"]?.toInt() ?: 0,
                    participatingSymbols = parseStringList(valueDict["participating_symbols"]),
                    transformationTypes = parseStringList(valueDict["transformation_types"])
                )
                
                analysis[vector] = transformationAnalysis
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse transformation analysis", e)
        }
        
        return analysis
    }
    
    private fun parseTransformationScenarios(pyArray: PyObject): List<TransformationScenario> {
        val scenarios = mutableListOf<TransformationScenario>()
        
        try {
            val arrayList = pyArray.asList()
            for (item in arrayList) {
                val scenarioDict = item.asMap()
                
                scenarios.add(
                    TransformationScenario(
                        vector = parseDeterritorializedVector(scenarioDict["vector"]?.toString()),
                        activationStrength = scenarioDict["activation_strength"]?.toFloat() ?: 0.0f,
                        participatingSymbols = parseStringList(scenarioDict["participating_symbols"]),
                        narrative = scenarioDict["transformation_narrative"]?.toString() ?: ""
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse transformation scenarios", e)
        }
        
        return scenarios
    }
    
    // Helper methods for converting Kotlin objects to Python
    
    private fun convertSymbolsToPython(symbols: List<SymbolMapping>): PyObject? {
        return try {
            val pythonList = python?.builtins?.callAttr("list")
            
            for (symbol in symbols) {
                val symbolDict = python?.builtins?.callAttr("dict")
                symbolDict?.put("symbol", symbol.symbol)
                symbolDict?.put("meaning", symbol.meaning)
                symbolDict?.put("confidence", symbol.confidence)
                symbolDict?.put("type", symbol.symbolType.name.lowercase())
                
                pythonList?.callAttr("append", symbolDict)
            }
            
            pythonList
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert symbols to Python", e)
            null
        }
    }
    
    private fun convertPatternsToPython(patterns: List<SymbolicPattern>): PyObject? {
        return try {
            val pythonList = python?.builtins?.callAttr("list")
            
            for (pattern in patterns) {
                val patternObj = convertPatternToPython(pattern)
                pythonList?.callAttr("append", patternObj)
            }
            
            pythonList
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert patterns to Python", e)
            null
        }
    }
    
    private fun convertPatternToPython(pattern: SymbolicPattern): PyObject? {
        return try {
            val patternDict = python?.builtins?.callAttr("dict")
            patternDict?.put("pattern_id", pattern.id)
            patternDict?.put("pattern_type", pattern.patternType.name.lowercase())
            patternDict?.put("elements", pattern.elements)
            patternDict?.put("coherence_score", pattern.coherenceScore)
            patternDict?.put("emergence_probability", pattern.emergenceProbability)
            patternDict?.put("complexity_measure", pattern.complexityMeasure)
            
            patternDict
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert pattern to Python", e)
            null
        }
    }
    
    // Helper parsing methods
    
    private fun parseSymbolType(typeStr: String?): SymbolType {
        return try {
            SymbolType.valueOf(typeStr?.uppercase() ?: "UNIVERSAL")
        } catch (e: Exception) {
            SymbolType.UNIVERSAL
        }
    }
    
    private fun parsePatternType(typeStr: String?): PatternType {
        return try {
            PatternType.valueOf(typeStr?.uppercase() ?: "NETWORK")
        } catch (e: Exception) {
            PatternType.NETWORK
        }
    }
    
    private fun parseDeterritorializedVector(vectorStr: String?): DeterritorializedVector {
        return try {
            DeterritorializedVector.valueOf(vectorStr?.uppercase() ?: "MULTIPLICITY")
        } catch (e: Exception) {
            DeterritorializedVector.MULTIPLICITY
        }
    }
    
    private fun parseConnectionType(typesObj: Any?): ConnectionType {
        return try {
            // For simplicity, return first type or default
            ConnectionType.SEMANTIC
        } catch (e: Exception) {
            ConnectionType.SEMANTIC
        }
    }
    
    private fun parseStringList(obj: Any?): List<String> {
        return try {
            when (obj) {
                is PyObject -> {
                    obj.asList().map { it.toString() }
                }
                is List<*> -> {
                    obj.map { it.toString() }
                }
                else -> emptyList()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse string list", e)
            emptyList()
        }
    }
    
    private fun parseTransformationReadiness(obj: Any?): Map<DeterritorializedVector, Float> {
        return try {
            val readiness = mutableMapOf<DeterritorializedVector, Float>()
            
            if (obj is PyObject) {
                val readinessDict = obj.asMap()
                for ((key, value) in readinessDict) {
                    val vector = parseDeterritorializedVector(key.toString())
                    val strength = (value as? PyObject)?.toFloat() ?: 0.0f
                    readiness[vector] = strength
                }
            }
            
            readiness
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse transformation readiness", e)
            emptyMap()
        }
    }
    
    private fun parseCulturalInfluence(obj: Any?): Map<String, Float> {
        return try {
            val influence = mutableMapOf<String, Float>()
            
            if (obj is PyObject) {
                val influenceDict = obj.asMap()
                for ((key, value) in influenceDict) {
                    val culture = key.toString()
                    val weight = (value as? PyObject)?.toFloat() ?: 0.0f
                    influence[culture] = weight
                }
            }
            
            influence
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse cultural influence", e)
            emptyMap()
        }
    }
    
    fun cleanup() {
        try {
            pythonObjectCache.clear()
            mapperModule = null
            isInitialized = false
            Log.d(TAG, "Enhanced Symbolic Dream Mapper cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Cleanup failed", e)
        }
    }
}

/**
 * Bridge for the dream narrative generator
 */
class DreamNarrativeGeneratorBridge(private val context: Context) {
    
    private var python: Python? = null
    private var generatorModule: PyObject? = null
    private var isInitialized = false
    
    companion object {
        private const val TAG = "DreamNarrativeGenerator"
        private const val MODULE_NAME = "dream_narrative_generator"
        private const val GENERATOR_CLASS = "DreamNarrativeGenerator"
    }
    
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            
            python = Python.getInstance()
            
            // Import the narrative generator module
            val module = python?.getModule(MODULE_NAME)
            if (module == null) {
                Log.e(TAG, "Failed to import $MODULE_NAME module")
                return@withContext false
            }
            
            // Create generator instance
            val generatorClass = module.get(GENERATOR_CLASS)
            generatorModule = generatorClass?.call()
            
            if (generatorModule == null) {
                Log.e(TAG, "Failed to create generator instance")
                return@withContext false
            }
            
            isInitialized = true
            Log.d(TAG, "Dream Narrative Generator initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Dream Narrative Generator", e)
            false
        }
    }
    
    suspend fun generateNarrative(
        symbolicAnalysis: Map<String, Any>,
        style: NarrativeStyle = NarrativeStyle.MYTHIC,
        length: NarrativeLength = NarrativeLength.MEDIUM,
        transformationIntensity: Float = 0.5f
    ): DreamNarrative = withContext(Dispatchers.IO) {
        
        if (!isInitialized || generatorModule == null) {
            throw IllegalStateException("Generator not initialized")
        }
        
        try {
            // Convert symbolic analysis to Python dict
            val pythonAnalysis = convertMapToPython(symbolicAnalysis)
            
            // Call generate_narrative method
            val result = generatorModule!!.callAttr(
                "generate_narrative",
                pythonAnalysis,
                style.pythonValue,
                length.pythonValue,
                transformationIntensity
            )
            
            // Parse result
            parseNarrativeResult(result)
            
        } catch (e: Exception) {
            Log.e(TAG, "Narrative generation failed", e)
            DreamNarrative.error("Generation failed: ${e.message}")
        }
    }
    
    suspend fun generateFieldCoherenceReport(narrativeResult: DreamNarrative): String = 
        withContext(Dispatchers.IO) {
            
        if (!isInitialized || generatorModule == null) {
            return@withContext "Generator not initialized"
        }
        
        try {
            // Convert narrative result to Python dict
            val pythonNarrative = convertNarrativeToPython(narrativeResult)
            
            // Generate report
            val report = generatorModule!!.callAttr(
                "generate_field_coherence_report",
                pythonNarrative
            )
            
            report.toString()
            
        } catch (e: Exception) {
            Log.e(TAG, "Report generation failed", e)
            "Report generation failed: ${e.message}"
        }
    }
    
    suspend fun generateTransformationScenarios(
        symbols: List<SymbolMapping>,
        fieldIntensity: Map<String, Float>,
        intensity: Float
    ): List<TransformationScenario> = withContext(Dispatchers.IO) {
        
        if (!isInitialized || generatorModule == null) {
            return@withContext emptyList()
        }
        
        try {
            // Convert parameters to Python
            val pythonSymbols = convertSymbolsToPython(symbols)
            val pythonFieldIntensity = convertMapToPython(fieldIntensity)
            
            val scenarios = generatorModule!!.callAttr(
                "_generate_transformation_scenarios",
                pythonSymbols,
                pythonFieldIntensity,
                intensity
            )
            
            parseTransformationScenarios(scenarios)
            
        } catch (e: Exception) {
            Log.e(TAG, "Transformation scenario generation failed", e)
            emptyList()
        }
    }
    
    private fun parseNarrativeResult(pyResult: PyObject): DreamNarrative {
        return try {
            val resultDict = pyResult.asMap()
            
            DreamNarrative(
                text = resultDict["narrative"]?.toString() ?: "",
                style = parseNarrativeStyle(resultDict["style"]?.toString()),
                fieldIntensity = parseFieldIntensity(resultDict["field_intensity"]),
                transformationScenarios = parseTransformationScenarios(resultDict["transformation_scenarios"]),
                narrativeNodes = resultDict["narrative_nodes"]?.toInt() ?: 0,
                symbolicDensity = resultDict["symbolic_density"]?.toFloat() ?: 0.0f,
                archetypalThemes = parseStringList(resultDict["archetypal_themes"]),
                deterritorializationVectors = parseStringList(resultDict["deterritorialization_vectors"])
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse narrative result", e)
            DreamNarrative.error("Failed to parse narrative result")
        }
    }
    
    private fun parseFieldIntensity(obj: Any?): FieldIntensity {
        return try {
            if (obj is PyObject) {
                val intensityDict = obj.asMap()
                FieldIntensity(
                    coherence = intensityDict["coherence"]?.toFloat() ?: 0.5f,
                    entropy = intensityDict["entropy"]?.toFloat() ?: 0.5f,
                    luminosity = intensityDict["luminosity"]?.toFloat() ?: 0.5f,
                    temporalFlux = intensityDict["temporal_flux"]?.toFloat() ?: 0.5f,
                    dimensionalDepth = intensityDict["dimensional_depth"]?.toFloat() ?: 0.5f,
                    archetypalResonance = intensityDict["archetypal_resonance"]?.toFloat() ?: 0.5f
                )
            } else {
                FieldIntensity()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse field intensity", e)
            FieldIntensity()
        }
    }
    
    private fun parseNarrativeStyle(styleStr: String?): NarrativeStyle {
        return try {
            NarrativeStyle.valueOf(styleStr?.uppercase() ?: "MYTHIC")
        } catch (e: Exception) {
            NarrativeStyle.MYTHIC
        }
    }
    
    private fun parseTransformationScenarios(obj: Any?): List<TransformationScenario> {
        val scenarios = mutableListOf<TransformationScenario>()
        
        try {
            if (obj is PyObject) {
                val scenariosList = obj.asList()
                for (item in scenariosList) {
                    val scenarioDict = item.asMap()
                    
                    scenarios.add(
                        TransformationScenario(
                            vector = parseDeterritorializedVector(scenarioDict["vector"]?.toString()),
                            activationStrength = scenarioDict["activation_strength"]?.toFloat() ?: 0.0f,
                            participatingSymbols = parseStringList(scenarioDict["participating_symbols"]),
                            narrative = scenarioDict["scenario"]?.toString() ?: ""
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse transformation scenarios", e)
        }
        
        return scenarios
    }
    
    // Helper methods
    
    private fun convertMapToPython(map: Map<String, Any>): PyObject? {
        return try {
            val pythonDict = python?.builtins?.callAttr("dict")
            
            for ((key, value) in map) {
                pythonDict?.put(key, value)
            }
            
            pythonDict
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert map to Python", e)
            null
        }
    }
    
    private fun convertSymbolsToPython(symbols: List<SymbolMapping>): PyObject? {
        return try {
            val pythonList = python?.builtins?.callAttr("list")
            
            for (symbol in symbols) {
                val symbolDict = python?.builtins?.callAttr("dict")
                symbolDict?.put("symbol", symbol.symbol)
                symbolDict?.put("meaning", symbol.meaning)
                symbolDict?.put("confidence", symbol.confidence)
                symbolDict?.put("type", symbol.symbolType.name.lowercase())
                
                pythonList?.callAttr("append", symbolDict)
            }
            
            pythonList
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert symbols to Python", e)
            null
        }
    }
    
    private fun convertNarrativeToPython(narrative: DreamNarrative): PyObject? {
        return try {
            val narrativeDict = python?.builtins?.callAttr("dict")
            narrativeDict?.put("narrative", narrative.text)
            narrativeDict?.put("style", narrative.style.name.lowercase())
            narrativeDict?.put("narrative_nodes", narrative.narrativeNodes)
            narrativeDict?.put("symbolic_density", narrative.symbolicDensity)
            
            // Convert field intensity
            val fieldIntensityDict = python?.builtins?.callAttr("dict")
            fieldIntensityDict?.put("coherence", narrative.fieldIntensity.coherence)
            fieldIntensityDict?.put("entropy", narrative.fieldIntensity.entropy)
            fieldIntensityDict?.put("luminosity", narrative.fieldIntensity.luminosity)
            fieldIntensityDict?.put("temporal_flux", narrative.fieldIntensity.temporalFlux)
            fieldIntensityDict?.put("dimensional_depth", narrative.fieldIntensity.dimensionalDepth)
            fieldIntensityDict?.put("archetypal_resonance", narrative.fieldIntensity.archetypalResonance)
            
            narrativeDict?.put("field_intensity", fieldIntensityDict)
            narrativeDict?.put("archetypal_themes", narrative.archetypalThemes)
            narrativeDict?.put("deterritorialization_vectors", narrative.deterritorializationVectors)
            
            narrativeDict
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert narrative to Python", e)
            null
        }
    }
    
    private fun parseStringList(obj: Any?): List<String> {
        return try {
            when (obj) {
                is PyObject -> obj.asList().map { it.toString() }
                is List<*> -> obj.map { it.toString() }
                else -> emptyList()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse string list", e)
            emptyList()
        }
    }
    
    private fun parseDeterritorializedVector(vectorStr: String?): DeterritorializedVector {
        return try {
            DeterritorializedVector.valueOf(vectorStr?.uppercase() ?: "MULTIPLICITY")
        } catch (e: Exception) {
            DeterritorializedVector.MULTIPLICITY
        }
    }
    
    fun cleanup() {
        try {
            generatorModule = null
            isInitialized = false
            Log.d(TAG, "Dream Narrative Generator cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Cleanup failed", e)
        }
    }
}

// Extension properties for enum conversions
val NarrativeStyle.pythonValue: String
    get() = this.name.lowercase()

val NarrativeLength.pythonValue: String
    get() = this.name.lowercase()
