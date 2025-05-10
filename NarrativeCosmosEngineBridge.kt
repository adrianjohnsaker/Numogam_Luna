// NarrativeCosmosEngineBridge.kt
package com.antonio.my.ai.girlfriend.free

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.*

class NarrativeCosmosEngineBridge private constructor(context: Context) {
    private val pythonBridge = PythonBridge.getInstance(context)
    
    companion object {
        @Volatile private var instance: NarrativeCosmosEngineBridge? = null
        
        fun getInstance(context: Context): NarrativeCosmosEngineBridge {
            return instance ?: synchronized(this) {
                instance ?: NarrativeCosmosEngineBridge(context).also { instance = it }
            }
        }
    }
    
    /**
     * Create a new world space in the narrative cosmos
     */
    suspend fun createWorld(
        name: String,
        description: String,
        principles: List<String>
    ): WorldSpaceResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "name" to name,
                "description" to description,
                "principles" to principles
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "create_world",
                params
            )
            
            parseWorldSpaceResult(result)
        }
    }
    
    /**
     * Add a symbol to a world space
     */
    suspend fun addSymbol(
        worldName: String,
        name: String,
        description: String,
        associations: List<String>? = null
    ): Boolean {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf(
                "world_name" to worldName,
                "name" to name,
                "description" to description
            )
            
            associations?.let { params["associations"] = it }
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "add_symbol",
                params
            )
            
            result as? Boolean ?: false
        }
    }
    
    /**
     * Add a personal symbol from experiences
     */
    suspend fun addPersonalSymbol(
        name: String,
        context: String,
        emotionalValence: Double,
        memoryFragments: List<String>
    ): Boolean {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "name" to name,
                "context" to context,
                "emotional_valence" to emotionalValence,
                "memory_fragments" to memoryFragments
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "add_personal_symbol",
                params
            )
            
            result as? Boolean ?: false
        }
    }
    
    /**
     * Integrate a personal symbol into a narrative world
     */
    suspend fun integratePersonalSymbol(
        symbol: String,
        world: String,
        context: String
    ): IntegrationResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "symbol" to symbol,
                "world" to world,
                "context" to context
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "integrate_personal_symbol",
                params
            )
            
            parseIntegrationResult(result)
        }
    }
    
    /**
     * Create a new story fragment in the ecosystem
     */
    suspend fun createStoryFragment(
        content: String,
        symbols: List<String>,
        mood: String
    ): Int {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "content" to content,
                "symbols" to symbols,
                "mood" to mood
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "create_story_fragment",
                params
            )
            
            (result as? Double)?.toInt() ?: -1
        }
    }
    
    /**
     * Get suggestions for expanding a story fragment
     */
    suspend fun expandStory(fragmentId: Int): StoryExpansionResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "fragment_id" to fragmentId
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "expand_story",
                params
            )
            
            parseStoryExpansionResult(result)
        }
    }
    
    /**
     * Start a new mythic cycle spanning multiple worlds
     */
    suspend fun startMythicCycle(
        name: String, 
        theme: String, 
        worlds: List<String>
    ): MythicCycleResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "name" to name,
                "theme" to theme,
                "worlds" to worlds
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "start_mythic_cycle",
                params
            )
            
            parseMythicCycleResult(result)
        }
    }
    
    /**
     * Create a new episode in the active mythic cycle
     */
    suspend fun createMythicEpisode(
        symbols: List<String>,
        tensions: List<Pair<String, String>>,
        world: String
    ): EpisodeResult? {
        return withContext(Dispatchers.IO) {
            val tensionsMap = tensions.map { listOf(it.first, it.second) }
            
            val params = mapOf(
                "symbols" to symbols,
                "tensions" to tensionsMap,
                "world" to world
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "create_mythic_episode",
                params
            )
            
            parseEpisodeResult(result)
        }
    }
    
    /**
     * Resolve a tension in an episode of the active mythic cycle
     */
    suspend fun resolveTension(
        episodeNum: Int,
        tensionIdx: Int,
        resolutionNarrative: String,
        transformationSymbols: List<String>
    ): ResolutionResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "episode_num" to episodeNum,
                "tension_idx" to tensionIdx,
                "resolution_narrative" to resolutionNarrative,
                "transformation_symbols" to transformationSymbols
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "resolve_tension",
                params
            )
            
            parseResolutionResult(result)
        }
    }
    
    /**
     * Analyze the overall development of the narrative cosmos
     */
    suspend fun analyzeDevelopment(): DevelopmentAnalysisResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "analyze_development",
                null
            )
            
            parseDevelopmentAnalysisResult(result)
        }
    }
    
    /**
     * Get suggestions for expanding a worldspace
     */
    suspend fun suggestWorldExpansion(worldName: String): WorldExpansionResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "world_name" to worldName
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "suggest_world_expansion",
                params
            )
            
            parseWorldExpansionResult(result)
        }
    }
    
    /**
     * Analyze how a personal symbol has evolved
     */
    suspend fun analyzeSymbol(personalSymbol: String): SymbolEvolutionResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "personal_symbol" to personalSymbol
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "analyze_symbol",
                params
            )
            
            parseSymbolEvolutionResult(result)
        }
    }
    
    /**
     * Create a boundary event between world spaces
     */
    suspend fun createBoundaryEvent(
        sourceWorld: String,
        targetWorld: String,
        description: String,
        affectedSymbols: List<String>
    ): BoundaryEventResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "source_world" to sourceWorld,
                "target_world" to targetWorld,
                "description" to description,
                "affected_symbols" to affectedSymbols
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "create_boundary_event",
                params
            )
            
            parseBoundaryEventResult(result)
        }
    }
    
    /**
     * Detect phase transitions in a world space
     */
    suspend fun detectPhaseTransition(worldName: String): PhaseTransitionResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "world_name" to worldName
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "detect_phase_transition",
                params
            )
            
            parsePhaseTransitionResult(result)
        }
    }
    
    /**
     * Find archetypal resonances for a personal symbol
     */
    suspend fun findArchetypalResonances(personalSymbol: String): List<ArchetypalResonanceResult>? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "personal_symbol" to personalSymbol
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "find_archetypal_resonances",
                params
            )
            
            parseArchetypalResonanceResults(result)
        }
    }
    
    /**
     * Register an archetypal pattern
     */
    suspend fun registerArchetypalPattern(
        patternName: String,
        description: String,
        symbolicElements: List<String>
    ): Boolean {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "pattern_name" to patternName,
                "description" to description,
                "symbolic_elements" to symbolicElements
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "register_archetypal_pattern",
                params
            )
            
            result as? Boolean ?: false
        }
    }
    
    /**
     * Add a narrative thread to a world space
     */
    suspend fun addNarrativeThread(
        worldName: String,
        threadName: String,
        description: String,
        symbols: List<String>,
        tensions: List<Pair<String, String>>? = null
    ): Boolean {
        return withContext(Dispatchers.IO) {
            val tensionsMap = tensions?.map { listOf(it.first, it.second) } ?: listOf()
            
            val params = mapOf(
                "world_name" to worldName,
                "thread_name" to threadName,
                "description" to description,
                "symbols" to symbols,
                "tensions" to tensionsMap
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "add_narrative_thread",
                params
            )
            
            result as? Boolean ?: false
        }
    }
    
    /**
     * Detect emerging narrative attractors in the story ecosystem
     */
    suspend fun detectNarrativeAttractors(): Map<String, Double>? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "detect_narrative_attractors",
                null
            )
            
            @Suppress("UNCHECKED_CAST")
            result as? Map<String, Double>
        }
    }
    
    /**
     * Get all symbols from a world space
     */
    suspend fun getWorldSymbols(worldName: String): List<SymbolResult>? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "world_name" to worldName
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "get_world_symbols",
                params
            )
            
            parseSymbolResults(result)
        }
    }
    
    /**
     * Get all narrative threads from a world space
     */
    suspend fun getWorldThreads(worldName: String): List<NarrativeThreadResult>? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "world_name" to worldName
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "get_world_threads",
                params
            )
            
            parseNarrativeThreadResults(result)
        }
    }
    
    /**
     * Parse results from Python into data classes
     */
    private fun parseWorldSpaceResult(result: Any?): WorldSpaceResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return WorldSpaceResult(
                name = map["name"] as? String ?: "",
                description = map["description"] as? String ?: "",
                corePrinciples = map["core_principles"] as? List<String> ?: listOf(),
                symbolCount = (map["symbol_count"] as? Double)?.toInt() ?: 0,
                threadCount = (map["thread_count"] as? Double)?.toInt() ?: 0,
                creationDate = map["creation_date"] as? String ?: "",
                evolutionStages = map["evolution_stages"] as? List<Map<String, Any>> ?: listOf()
            )
        }
        return null
    }
    
    private fun parseIntegrationResult(result: Any?): IntegrationResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return IntegrationResult(
                    symbol = "",
                    worldSpace = "",
                    narrativeContext = "",
                    timestamp = "",
                    resonances = listOf(),
                    transformation = null,
                    error = map["error"] as? String
                )
            }
            
            val resonances = mutableListOf<ArchetypalResonanceResult>()
            (map["resonances"] as? List<List<Any>>)?.forEach { pair ->
                if (pair.size >= 2) {
                    resonances.add(
                        ArchetypalResonanceResult(
                            patternName = pair[0] as? String ?: "",
                            strength = pair[1] as? Double ?: 0.0
                        )
                    )
                }
            }
            
            val transformationMap = map["transformation"] as? Map<String, Any>
            val transformation = if (transformationMap != null) {
                TransformationResult(
                    originalContext = transformationMap["original_context"] as? String ?: "",
                    narrativeExpression = transformationMap["narrative_expression"] as? String ?: "",
                    emotionalShift = transformationMap["emotional_shift"] as? Double ?: 0.0
                )
            } else null
            
            return IntegrationResult(
                symbol = map["symbol"] as? String ?: "",
                worldSpace = map["world_space"] as? String ?: "",
                narrativeContext = map["narrative_context"] as? String ?: "",
                timestamp = map["timestamp"] as? String ?: "",
                resonances = resonances,
                transformation = transformation,
                error = null
            )
        }
        return null
    }
    
    private fun parseStoryExpansionResult(result: Any?): StoryExpansionResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return StoryExpansionResult(
                    fragment = null,
                    suggestedSymbols = listOf(),
                    narrativeTension = 0.0,
                    potentialDirections = listOf(),
                    error = map["error"] as? String
                )
            }
            
            val fragmentMap = map["fragment"] as? Map<String, Any>
            val fragment = if (fragmentMap != null) {
                StoryFragmentResult(
                    id = (fragmentMap["id"] as? Double)?.toInt() ?: -1,
                    content = fragmentMap["content"] as? String ?: "",
                    symbols = fragmentMap["symbols"] as? List<String> ?: listOf(),
                    mood = fragmentMap["mood"] as? String ?: "",
                    timestamp = fragmentMap["timestamp"] as? String ?: "",
                    connections = fragmentMap["connections"] as? List<Any> ?: listOf(),
                    evolutionState = fragmentMap["evolution_state"] as? Double ?: 0.0
                )
            } else null
            
            val suggestedSymbols = mutableListOf<Pair<String, Double>>()
            (map["suggested_symbols"] as? List<List<Any>>)?.forEach { pair ->
                if (pair.size >= 2) {
                    suggestedSymbols.add(
                        (pair[0] as? String ?: "") to (pair[1] as? Double ?: 0.0)
                    )
                }
            }
            
            return StoryExpansionResult(
                fragment = fragment,
                suggestedSymbols = suggestedSymbols,
                narrativeTension = map["narrative_tension"] as? Double ?: 0.0,
                potentialDirections = map["potential_directions"] as? List<String> ?: listOf(),
                error = null
            )
        }
        return null
    }
    
    private fun parseMythicCycleResult(result: Any?): MythicCycleResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return MythicCycleResult(
                    id = 0,
                    name = "",
                    theme = "",
                    worldspaces = listOf(),
                    creationDate = "",
                    evolutionStage = "",
                    developmentVector = listOf(),
                    episodeCount = 0,
                    error = map["error"] as? String
                )
            }
            
            return MythicCycleResult(
                id = (map["id"] as? Double)?.toInt() ?: 0,
                name = map["name"] as? String ?: "",
                theme = map["theme"] as? String ?: "",
                worldspaces = map["worldspaces"] as? List<String> ?: listOf(),
                creationDate = map["creation_date"] as? String ?: "",
                evolutionStage = map["evolution_stage"] as? String ?: "",
                developmentVector = map["development_vector"] as? List<Double> ?: listOf(),
                episodeCount = (map["episodes"] as? List<*>)?.size ?: 0,
                error = null
            )
        }
        return null
    }
    
    private fun parseEpisodeResult(result: Any?): EpisodeResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return EpisodeResult(
                    number = 0,
                    title = "",
                    worldspace = "",
                    symbols = listOf(),
                    tensions = listOf(),
                    narrativeAttractors = listOf(),
                    personalResonances = listOf(),
                    creationDate = "",
                    resolutionLevel = 0.0,
                    error = map["error"] as? String
                )
            }
            
            val tensions = mutableListOf<Pair<String, String>>()
            (map["tensions"] as? List<List<String>>)?.forEach { pair ->
                if (pair.size >= 2) {
                    tensions.add(pair[0] to pair[1])
                }
            }
            
            val attractors = mutableListOf<Pair<String, Double>>()
            (map["narrative_attractors"] as? List<List<Any>>)?.forEach { pair ->
                if (pair.size >= 2) {
                    attractors.add(
                        (pair[0] as? String ?: "") to (pair[1] as? Double ?: 0.0)
                    )
                }
            }
            
            val resonances = mutableListOf<Pair<String, String>>()
            (map["personal_resonances"] as? List<List<String>>)?.forEach { pair ->
                if (pair.size >= 2) {
                    resonances.add(pair[0] to pair[1])
                }
            }
            
            return EpisodeResult(
                number = (map["number"] as? Double)?.toInt() ?: 0,
                title = map["title"] as? String ?: "",
                worldspace = map["worldspace"] as? String ?: "",
                symbols = map["symbols"] as? List<String> ?: listOf(),
                tensions = tensions,
                narrativeAttractors = attractors,
                personalResonances = resonances,
                creationDate = map["creation_date"] as? String ?: "",
                resolutionLevel = map["resolution_level"] as? Double ?: 0.0,
                error = null
            )
        }
        return null
    }
    
    private fun parseResolutionResult(result: Any?): ResolutionResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return ResolutionResult(
                    episode = 0,
                    tension = listOf(),
                    narrative = "",
                    transformationSymbols = listOf(),
                    resolutionDate = "",
                    error = map["error"] as? String
                )
            }
            
            val tensionPair = map["tension"] as? List<String> ?: listOf()
            val tension = if (tensionPair.size >= 2) {
                listOf(tensionPair[0], tensionPair[1])
            } else {
                listOf()
            }
            
            return ResolutionResult(
                episode = (map["episode"] as? Double)?.toInt() ?: 0,
                tension = tension,
                narrative = map["narrative"] as? String ?: "",
                transformationSymbols = map["transformation_symbols"] as? List<String> ?: listOf(),
                resolutionDate = map["resolution_date"] as? String ?: "",
                error = null
            )
        }
        return null
    }
    
    private fun parseDevelopmentAnalysisResult(result: Any?): DevelopmentAnalysisResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            val metrics = map["development_metrics"] as? Map<String, Double> ?: mapOf()
            
            val recommendations = mutableListOf<RecommendationResult>()
            (map["growth_recommendations"] as? List<Map<String, Any>>)?.forEach { recMap ->
                val specificTargets = recMap["specific_targets"] as? List<Map<String, Any>>
                
                recommendations.add(
                    RecommendationResult(
                        focus = recMap["focus"] as? String ?: "",
                        action = recMap["action"] as? String ?: "",
                        suggestion = recMap["suggestion"] as? String ?: "",
                        specificTargets = specificTargets
                    )
                )
            }
            
            return DevelopmentAnalysisResult(
                cycles = (map["cycles"] as? Double)?.toInt() ?: 0,
                episodes = (map["episodes"] as? Double)?.toInt() ?: 0,
                resolutions = (map["resolutions"] as? Double)?.toInt() ?: 0,
                worldspaces = (map["worldspaces"] as? Double)?.toInt() ?: 0,
                averageDevelopmentVector = map["average_development_vector"] as? List<Double> ?: listOf(),
                dominantDimension = map["dominant_dimension"] as? String ?: "",
                ecosystemHealth = map["ecosystem_health"] as? Double ?: 0.0,
                healthStatus = map["health_status"] as? String ?: "",
                developmentMetrics = metrics,
                growthRecommendations = recommendations
            )
        }
        return null
    }
    
    private fun parseWorldExpansionResult(result: Any?): WorldExpansionResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return WorldExpansionResult(
                    worldName = "",
                    expansionType = "",
                    suggestion = "",
                    potentialDevelopments = null,
                    potentialWorlds = null,
                    symbolicClusters = null,
                    possibleDestinations = null,
                    error = map["error"] as? String
                )
            }
            
            return WorldExpansionResult(
                worldName = map["world_name"] as? String ?: "",
                expansionType = map["expansion_type"] as? String ?: "",
                suggestion = map["suggestion"] as? String ?: "",
                potentialDevelopments = map["potential_developments"] as? List<Map<String, Any>>,
                potentialWorlds = map["potential_worlds"] as? List<Map<String, Any>>,
                symbolicClusters = map["symbolic_clusters"] as? List<Map<String, Any>>,
                possibleDestinations = map["possible_destinations"] as? List<Map<String, Any>>,
                error = null
            )
        }
        return null
    }
    
    private fun parseSymbolEvolutionResult(result: Any?): SymbolEvolutionResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return SymbolEvolutionResult(
                    symbol = "",
                    evolutionStage = "",
                    integrationLevel = 0.0,
                    trajectory = listOf(),
                    integrations = 0,
                    currentResonances = listOf(),
                    message = map["error"] as? String
                )
            }
            
            val resonances = mutableListOf<ArchetypalResonanceResult>()
            (map["current_resonances"] as? List<List<Any>>)?.forEach { pair ->
                if (pair.size >= 2) {
                    resonances.add(
                        ArchetypalResonanceResult(
                            patternName = pair[0] as? String ?: "",
                            strength = pair[1] as? Double ?: 0.0
                        )
                    )
                }
            }
            
            return SymbolEvolutionResult(
                symbol = map["symbol"] as? String ?: "",
                evolutionStage = map["evolution_stage"] as? String ?: "",
                integrationLevel = map["integration_level"] as? Double ?: 0.0,
                trajectory = map["trajectory"] as? List<Map<String, Any>> ?: listOf(),
                integrations = (map["integrations"] as? Double)?.toInt() ?: 0,
                currentResonances = resonances,
                message = map["message"] as? String
            )
        }
        return null
    }
    
    private fun parseBoundaryEventResult(result: Any?): BoundaryEventResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return BoundaryEventResult(
                    id = 0,
                    sourceWorld = "",
                    targetWorld = "",
                    description = "",
                    affectedSymbols = listOf(),
                    timestamp = "",
                    ontologicalImpact = 0.0,
                    error = map["error"] as? String
                )
            }
            
            return BoundaryEventResult(
                id = (map["id"] as? Double)?.toInt() ?: 0,
                sourceWorld = map["source_world"] as? String ?: "",
                targetWorld = map["target_world"] as? String ?: "",
                description = map["description"] as? String ?: "",
                affectedSymbols = map["affected_symbols"] as? List<String> ?: listOf(),
                timestamp = map["timestamp"] as? String ?: "",
                ontologicalImpact = map["ontological_impact"] as? Double ?: 0.0,
                error = null
            )
        }
        return null
    }
    
    private fun parsePhaseTransitionResult(result: Any?): PhaseTransitionResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            // Check for error
            if (map.containsKey("error")) {
                return PhaseTransitionResult(
                    worldName = "",
                    isTransitioning = false,
                    transitionType = null,
                    timestamp = null,
                    clustering = null,
                    components = null,
                    density = null,
                    stabilityMetrics = null,
                    error = map["error"] as? String
                )
            }
            
            val isTransitioning = map["is_transitioning"] as? Boolean ?: false
            
            if (isTransitioning) {
                return PhaseTransitionResult(
                    worldName = map["world_name"] as? String ?: "",
                    isTransitioning = true,
                    transitionType = map["transition_type"] as? String,
                    timestamp = map["timestamp"] as? String,
                    clustering = map["clustering"] as? Double,
                    components = (map["components"] as? Double)?.toInt(),
                    density = map["density"] as? Double,
                    stabilityMetrics = null,
                    error = null
                )
            } else {
                val metrics = map["stability_metrics"] as? Map<String, Any>
                val stabilityMetrics = if (metrics != null) {
                    StabilityMetricsResult(
                        clustering = metrics["clustering"] as? Double ?: 0.0,
                        components = (metrics["components"] as? Double)?.toInt() ?: 0,
                        density = metrics["density"] as? Double ?: 0.0
                    )
                } else null
                
                return PhaseTransitionResult(
                    worldName = map["world_name"] as? String ?: "",
                    isTransitioning = false,
                    transitionType = null,
                    timestamp = null,
                    clustering = null,
                    components = null,
                    density = null,
                    stabilityMetrics = stabilityMetrics,
                    error = null
                )
            }
        }
        return null
    }
    
    private fun parseArchetypalResonanceResults(result: Any?): List<ArchetypalResonanceResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<List<Any>>)?.let { resonancesList ->
            val resonances = mutableListOf<ArchetypalResonanceResult>()
            
            resonancesList.forEach { pair ->
                if (pair.size >= 2) {
                    resonances.add(
                        ArchetypalResonanceResult(
                            patternName = pair[0] as? String ?: "",
                            strength = pair[1] as? Double ?: 0.0
                        )
                    )
                }
            }
            
            return resonances
        }
        return null
    }
    
    private fun parseSymbolResults(result: Any?): List<SymbolResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<Map<String, Any>>)?.let { symbolsList ->
            return symbolsList.map { map ->
                SymbolResult(
                    name = map["name"] as? String ?: "",
                    description = map["description"] as? String ?: "",
                    associations = (map["associations"] as? List<String>)?.toSet() ?: setOf(),
                    intensity = map["intensity"] as? Double ?: 0.0,
                    firstAppearance = map["first_appearance"] as? String ?: "",
                    lastInteraction = map["last_interaction"] as? String ?: "",
                    variants = map["variants"] as? List<String> ?: listOf(),
                    shadowAspects = map["shadow_aspects"] as? List<String> ?: listOf()
                )
            }
        }
        return null
    }
    
    private fun parseNarrativeThreadResults(result: Any?): List<NarrativeThreadResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<Map<String, Any>>)?.let { threadsList ->
            return threadsList.map { map ->
                val tensions = mutableListOf<Pair<String, String>>()
                (map["tensions"] as? List<List<String>>)?.forEach { pair ->
                    if (pair.size >= 2) {
                        tensions.add(pair[0] to pair[1])
                    }
                }
                
                NarrativeThreadResult(
                    name = map["name"] as? String ?: "",
                    description = map["description"] as? String ?: "",
                    symbols = map["symbols"] as? List<String> ?: listOf(),
                    events = map["events"] as? List<Map<String, Any>> ?: listOf(),
                    tensions = tensions,
                    resolutionState = map["resolution_state"] as? Double ?: 0.0,
                    emergenceDate = map["emergence_date"] as? String ?: "",
                    relatedThreads = (map["related_threads"] as? List<String>)?.toSet() ?: setOf()
                )
            }
        }
        return null
    }
}

// Data classes for structured results
data class WorldSpaceResult(
    val name: String,
    val description: String,
    val corePrinciples: List<String>,
    val symbolCount: Int,
    val threadCount: Int,
    val creationDate: String,
    val evolutionStages: List<Map<String, Any>>
) {
    fun getLatestEvolutionStage(): Map<String, Any>? =
        evolutionStages.maxByOrNull { it["date"] as? String ?: "" }
        
    fun isPrincipleBased(): Boolean = corePrinciples.size >= 3
    
    fun isSymbolRich(): Boolean = symbolCount >= 10
    
    fun isNarrativelyComplex(): Boolean = threadCount >= 5
}

data class IntegrationResult(
    val symbol: String,
    val worldSpace: String,
    val narrativeContext: String,
    val timestamp: String,
    val resonances: List<ArchetypalResonanceResult>,
    val transformation: TransformationResult?,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun hasStrongResonances(): Boolean = 
        resonances.any { it.strength >= 0.7 }
    
    fun getStrongestResonance(): ArchetypalResonanceResult? =
        resonances.maxByOrNull { it.strength }
}

data class TransformationResult(
    val originalContext: String,
    val narrativeExpression: String,
    val emotionalShift: Double
) {
    fun isPositiveShift(): Boolean = emotionalShift > 0
    
    fun isSignificantShift(): Boolean = Math.abs(emotionalShift) >= 0.5
}

data class ArchetypalResonanceResult(
    val patternName: String,
    val strength: Double
) {
    fun isStrong(): Boolean = strength >= 0.7
    
    fun isModerate(): Boolean = strength >= 0.4 && strength < 0.7
    
    fun isWeak(): Boolean = strength < 0.4
}

data class StoryFragmentResult(
    val id: Int,
    val content: String,
    val symbols: List<String>,
    val mood: String,
    val timestamp: String,
    val connections: List<Any>,
    val evolutionState: Double
) {
    fun isEvolved(): Boolean = evolutionState >= 0.5
    
    fun getSymbolCount(): Int = symbols.size
    
    fun getSummary(): String = 
        if (content.length > 100) content.substring(0, 97) + "..." else content
}

data class StoryExpansionResult(
    val fragment: StoryFragmentResult?,
    val suggestedSymbols: List<Pair<String, Double>>,
    val narrativeTension: Double,
    val potentialDirections: List<String>,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun hasHighTension(): Boolean = narrativeTension >= 0.7
    
    fun getTopSuggestedSymbol(): Pair<String, Double>? =
        suggestedSymbols.maxByOrNull { it.second }
    
    fun getTopSuggestedSymbols(count: Int = 3): List<Pair<String, Double>> =
        suggestedSymbols.sortedByDescending { it.second }.take(count)
}

data class MythicCycleResult(
    val id: Int,
    val name: String,
    val theme: String,
    val worldspaces: List<String>,
    val creationDate: String,
    val evolutionStage: String,
    val developmentVector: List<Double>,
    val episodeCount: Int,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun isActive(): Boolean = evolutionStage != "Completed" && evolutionStage != "Abandoned"
    
    fun isMultiWorld(): Boolean = worldspaces.size > 1
    
    fun getDevelopmentMagnitude(): Double {
        if (developmentVector.size < 3) return 0.0
        return Math.sqrt(developmentVector.sumOf { it * it })
    }
    
    fun getDominantDimension(): Int? =
        developmentVector.withIndex().maxByOrNull { it.value }?.index
}

data class EpisodeResult(
    val number: Int,
    val title: String,
    val worldspace: String,
    val symbols: List<String>,
    val tensions: List<Pair<String, String>>,
    val narrativeAttractors: List<Pair<String, Double>>,
    val personalResonances: List<Pair<String, String>>,
    val creationDate: String,
    val resolutionLevel: Double,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun isResolved(): Boolean = resolutionLevel >= 0.99
    
    fun getTopAttractors(count: Int = 2): List<Pair<String, Double>> =
        narrativeAttractors.sortedByDescending { it.second }.take(count)
    
    fun hasTensions(): Boolean = tensions.isNotEmpty()
    
    fun hasPersonalConnection(): Boolean = personalResonances.isNotEmpty()
}

data class ResolutionResult(
    val episode: Int,
    val tension: List<String>,
    val narrative: String,
    val transformationSymbols: List<String>,
    val resolutionDate: String,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun getTransformationCount(): Int = transformationSymbols.size
    
    fun getTensionPair(): Pair<String, String>? =
        if (tension.size >= 2) tension[0] to tension[1] else null
    
    fun getSummary(): String = 
        if (narrative.length > 100) narrative.substring(0, 97) + "..." else narrative
}

data class DevelopmentAnalysisResult(
    val cycles: Int,
    val episodes: Int,
    val resolutions: Int,
    val worldspaces: Int,
    val averageDevelopmentVector: List<Double>,
    val dominantDimension: String,
    val ecosystemHealth: Double,
    val healthStatus: String,
    val developmentMetrics: Map<String, Double>,
    val growthRecommendations: List<RecommendationResult>
) {
    fun isHealthy(): Boolean = ecosystemHealth >= 0.7
    
    fun getNarrativeCoherence(): Double = developmentMetrics["narrative_coherence"] ?: 0.0
    
    fun getSymbolicDepth(): Double = developmentMetrics["symbolic_depth"] ?: 0.0
    
    fun getOntologicalComplexity(): Double = developmentMetrics["ontological_complexity"] ?: 0.0
    
    fun getPersonalIntegration(): Double = developmentMetrics["personal_integration"] ?: 0.0
    
    fun getWeakestMetric(): String =
        developmentMetrics.entries.minByOrNull { it.value }?.key ?: ""
    
    fun getStrongestMetric(): String =
        developmentMetrics.entries.maxByOrNull { it.value }?.key ?: ""
    
    fun getCriticalRecommendations(): List<RecommendationResult> =
        growthRecommendations.filter { it.focus == getWeakestMetric() }
}

data class RecommendationResult(
    val focus: String,
    val action: String,
    val suggestion: String,
    val specificTargets: List<Map<String, Any>>?
) {
    fun hasSpecificTargets(): Boolean = specificTargets != null && specificTargets.isNotEmpty()
    
    fun getTargetCount(): Int = specificTargets?.size ?: 0
    
    fun getFirstTargetDescription(): String? = specificTargets?.firstOrNull()?.let { target ->
        when {
            target.containsKey("symbol") -> "Symbol: ${target["symbol"]}"
            target.containsKey("worldspace") -> "World: ${target["worldspace"]}"
            target.containsKey("cycle") -> "Cycle: ${target["cycle"]}, Episode: ${target["episode"]}"
            else -> target.entries.firstOrNull()?.let { "${it.key}: ${it.value}" }
        }
    }
}

data class WorldExpansionResult(
    val worldName: String,
    val expansionType: String,
    val suggestion: String,
    val potentialDevelopments: List<Map<String, Any>>?,
    val potentialWorlds: List<Map<String, Any>>?,
    val symbolicClusters: List<Map<String, Any>>?,
    val possibleDestinations: List<Map<String, Any>>?,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun isBifurcating(): Boolean = expansionType == "Bifurcation"
    
    fun isCrystallizing(): Boolean = expansionType == "Deepening"
    
    fun isDissolving(): Boolean = expansionType == "Absorption"
    
    fun isGrowingOrganically(): Boolean = expansionType == "Organic Growth"
    
    fun getRelevantExpansionData(): List<Map<String, Any>>? = when (expansionType) {
        "Bifurcation" -> potentialWorlds
        "Deepening" -> symbolicClusters
        "Absorption" -> possibleDestinations
        "Organic Growth" -> potentialDevelopments
        else -> null
    }
}

data class SymbolEvolutionResult(
    val symbol: String,
    val evolutionStage: String,
    val integrationLevel: Double,
    val trajectory: List<Map<String, Any>>,
    val integrations: Int,
    val currentResonances: List<ArchetypalResonanceResult>,
    val message: String?
) {
    fun isDormant(): Boolean = evolutionStage == "Dormant"
    
    fun isInitial(): Boolean = evolutionStage == "Initial Integration"
    
    fun isTransformed(): Boolean = evolutionStage == "Archetypal Transformation"
    
    fun hasMigrated(): Boolean = evolutionStage == "Cross-World Migration"
    
    fun isRefined(): Boolean = evolutionStage == "Contextual Refinement"
    
    fun getTopResonance(): ArchetypalResonanceResult? =
        currentResonances.maxByOrNull { it.strength }
    
    fun hasStrongResonances(): Boolean =
        currentResonances.any { it.strength >= 0.7 }
    
    fun isWellIntegrated(): Boolean = integrationLevel >= 0.5
    
    fun getTrajectoryLength(): Int = trajectory.size
}

data class BoundaryEventResult(
    val id: Int,
    val sourceWorld: String,
    val targetWorld: String,
    val description: String,
    val affectedSymbols: List<String>,
    val timestamp: String,
    val ontologicalImpact: Double,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun isSignificantImpact(): Boolean = ontologicalImpact >= 0.5
    
    fun getAffectedSymbolCount(): Int = affectedSymbols.size
    
    fun getSummary(): String = 
        if (description.length > 100) description.substring(0, 97) + "..." else description
}

data class PhaseTransitionResult(
    val worldName: String,
    val isTransitioning: Boolean,
    val transitionType: String?,
    val timestamp: String?,
    val clustering: Double?,
    val components: Int?,
    val density: Double?,
    val stabilityMetrics: StabilityMetricsResult?,
    val error: String?
) {
    fun hasError(): Boolean = error != null
    
    fun isFragmenting(): Boolean = transitionType == "Fragmentation"
    
    fun isCrystallizing(): Boolean = transitionType == "Crystallization"
    
    fun isDissolving(): Boolean = transitionType == "Dissolution"
    
    fun isStable(): Boolean = !isTransitioning
    
    fun getComponentCount(): Int = components ?: stabilityMetrics?.components ?: 0
    
    fun getClusteringValue(): Double = clustering ?: stabilityMetrics?.clustering ?: 0.0
    
    fun getDensityValue(): Double = density ?: stabilityMetrics?.density ?: 0.0
}

data class StabilityMetricsResult(
    val clustering: Double,
    val components: Int,
    val density: Double
) {
    fun isHighlyClustered(): Boolean = clustering >= 0.7
    
    fun isFragmented(): Boolean = components > 1
    
    fun isHighlyConnected(): Boolean = density >= 0.7
    
    fun isSparse(): Boolean = density <= 0.3
}

data class SymbolResult(
    val name: String,
    val description: String,
    val associations: Set<String>,
    val intensity: Double,
    val firstAppearance: String,
    val lastInteraction: String,
    val variants: List<String>,
    val shadowAspects: List<String>
) {
    fun isActive(): Boolean {
        // Parse dates to check if last interaction is recent
        // For simplicity, we'll just check if it has any variants or shadow aspects
        return variants.isNotEmpty() || shadowAspects.isNotEmpty()
    }
    
    fun isIntense(): Boolean = intensity >= 0.7
    
    fun hasVariants(): Boolean = variants.isNotEmpty()
    
    fun hasShadowAspects(): Boolean = shadowAspects.isNotEmpty()
    
    fun isFullyDeveloped(): Boolean = 
        intensity >= 0.5 && associations.size >= 3 && (variants.isNotEmpty() || shadowAspects.isNotEmpty())
    
    fun getSummary(): String = 
        if (description.length > 100) description.substring(0, 97) + "..." else description
}

data class NarrativeThreadResult(
    val name: String,
    val description: String,
    val symbols: List<String>,
    val events: List<Map<String, Any>>,
    val tensions: List<Pair<String, String>>,
    val resolutionState: Double,
    val emergenceDate: String,
    val relatedThreads: Set<String>
) {
    fun isResolved(): Boolean = resolutionState >= 0.99
    
    fun isPartiallyResolved(): Boolean = resolutionState >= 0.5 && resolutionState < 0.99
    
    fun isUnresolved(): Boolean = resolutionState < 0.5
    
    fun getEventCount(): Int = events.size
    
    fun getTensionCount(): Int = tensions.size
    
    fun getRelatedThreadCount(): Int = relatedThreads.size
    
    fun getSummary(): String = 
        if (description.length > 100) description.substring(0, 97) + "..." else description
}

/**
 * Helper class for building narrative cosmos components
 */
class NarrativeCosmosBuilder {
    // Symbol builder
    class SymbolBuilder {
        private var name: String = ""
        private var description: String = ""
        private val associations = mutableSetOf<String>()
        private var shadowAspects = mutableListOf<String>()
        
        fun name(name: String): SymbolBuilder {
            this.name = name
            return this
        }
        
        fun description(description: String): SymbolBuilder {
            this.description = description
            return this
        }
        
        fun addAssociation(association: String): SymbolBuilder {
            associations.add(association)
            return this
        }
        
        fun addShadowAspect(shadowAspect: String): SymbolBuilder {
            shadowAspects.add(shadowAspect)
            return this
        }
        
        fun build(): Map<String, Any> {
            return mapOf(
                "name" to name,
                "description" to description,
                "associations" to associations.toList(),
                "shadow_aspects" to shadowAspects
            )
        }
    }
    
    // World space builder
    class WorldBuilder {
        private var name: String = ""
        private var description: String = ""
        private val principles = mutableListOf<String>()
        
        fun name(name: String): WorldBuilder {
            this.name = name
            return this
        }
        
        fun description(description: String): WorldBuilder {
            this.description = description
            return this
        }
        
        fun addPrinciple(principle: String): WorldBuilder {
            principles.add(principle)
            return this
        }
        
        fun build(): Triple<String, String, List<String>> {
            return Triple(name, description, principles)
        }
    }
    
    // Story fragment builder
    class StoryFragmentBuilder {
        private var content: String = ""
        private val symbols = mutableListOf<String>()
        private var mood: String = "neutral"
        
        fun content(content: String): StoryFragmentBuilder {
            this.content = content
            return this
        }
        
        fun addSymbol(symbol: String): StoryFragmentBuilder {
            symbols.add(symbol)
            return this
        }
        
        fun mood(mood: String): StoryFragmentBuilder {
            this.mood = mood
            return this
        }
        
        fun build(): Triple<String, List<String>, String> {
            return Triple(content, symbols, mood)
        }
    }
    
    // Episode builder
    class EpisodeBuilder {
        private val symbols = mutableListOf<String>()
        private val tensions = mutableListOf<Pair<String, String>>()
        private var world: String = ""
        
        fun addSymbol(symbol: String): EpisodeBuilder {
            symbols.add(symbol)
            return this
        }
        
        fun addTension(symbol1: String, symbol2: String): EpisodeBuilder {
            tensions.add(symbol1 to symbol2)
            return this
        }
        
        fun world(world: String): EpisodeBuilder {
            this.world = world
            return this
        }
        
        fun build(): Triple<List<String>, List<Pair<String, String>>, String> {
            return Triple(symbols, tensions, world)
        }
    }
}

/**
 * Main API interface for Narrative Cosmos
 */
class NarrativeCosmos(private val context: Context) {
    private val bridge = NarrativeCosmosEngineBridge.getInstance(context)
    
    /**
     * Create a new world in the narrative cosmos
     */
    suspend fun createWorld(name: String, description: String, principles: List<String>): WorldSpaceResult? {
        return bridge.createWorld(name, description, principles)
    }
    
    /**
     * Add a symbol to a world
     */
    suspend fun addSymbol(worldName: String, name: String, description: String, associations: List<String>? = null): Boolean {
        return bridge.addSymbol(worldName, name, description, associations)
    }
    
    /**
     * Add a personal symbol
     */
    suspend fun addPersonalSymbol(name: String, context: String, emotionalValence: Double, memoryFragments: List<String>): Boolean {
        return bridge.addPersonalSymbol(name, context, emotionalValence, memoryFragments)
    }
    
    /**
     * Integrate a personal symbol into a narrative world
     */
    suspend fun integratePersonalSymbol(symbol: String, world: String, context: String): IntegrationResult? {
        return bridge.integratePersonalSymbol(symbol, world, context)
    }
    
    /**
     * Create a story fragment
     */
    suspend fun createStoryFragment(content: String, symbols: List<String>, mood: String): Int {
        return bridge.createStoryFragment(content, symbols, mood)
    }
    
    /**
     * Get expansion suggestions for a story
     */
    suspend fun expandStory(fragmentId: Int): StoryExpansionResult? {
        return bridge.expandStory(fragmentId)
    }
    
    /**
     * Start a mythic cycle
     */
    suspend fun startMythicCycle(name: String, theme: String, worlds: List<String>): MythicCycleResult? {
        return bridge.startMythicCycle(name, theme, worlds)
    }
    
    /**
     * Create a mythic episode
     */
    suspend fun createMythicEpisode(symbols: List<String>, tensions: List<Pair<String, String>>, world: String): EpisodeResult? {
        return bridge.createMythicEpisode(symbols, tensions, world)
    }
    
    /**
     * Resolve a tension in an episode
     */
    suspend fun resolveTension(episodeNum: Int, tensionIdx: Int, resolution: String, transformations: List<String>): ResolutionResult? {
        return bridge.resolveTension(episodeNum, tensionIdx, resolution, transformations)
    }
    
    /**
     * Analyze overall narrative development
     */
    suspend fun analyzeDevelopment(): DevelopmentAnalysisResult? {
        return bridge.analyzeDevelopment()
    }
    
    /**
     * Get suggestions for world expansion
     */
    suspend fun suggestWorldExpansion(worldName: String): WorldExpansionResult? {
        return bridge.suggestWorldExpansion(worldName)
    }
    
    /**
     * Analyze symbol evolution
     */
    suspend fun analyzeSymbol(personalSymbol: String): SymbolEvolutionResult? {
        return bridge.analyzeSymbol(personalSymbol)
    }
    
    /**
     * Register an archetypal pattern
     */
    suspend fun registerArchetype(name: String, description: String, elements: List<String>): Boolean {
        return bridge.registerArchetypalPattern(name, description, elements)
    }
    
    /**
     * Create a boundary event between worlds
     */
    suspend fun createBoundaryEvent(sourceWorld: String, targetWorld: String, description: String, affectedSymbols: List<String>): BoundaryEventResult? {
        return bridge.createBoundaryEvent(sourceWorld, targetWorld, description, affectedSymbols)
    }
    
    /**
     * Detect if a world is undergoing a phase transition
     */
    suspend fun detectPhaseTransition(worldName: String): PhaseTransitionResult? {
        return bridge.detectPhaseTransition(worldName)
    }
    
    /**
     * Find archetypal resonances for a personal symbol
     */
    suspend fun findArchetypalResonances(symbol: String): List<ArchetypalResonanceResult>? {
        return bridge.findArchetypalResonances(symbol)
    }
    
    /**
     * Add a narrative thread to a world
     */
    suspend fun addNarrativeThread(worldName: String, threadName: String, description: String, symbols: List<String>, tensions: List<Pair<String, String>>? = null): Boolean {
        return bridge.addNarrativeThread(worldName, threadName, description, symbols, tensions)
    }
    
    /**
     * Get all symbols from a world
     */
    suspend fun getWorldSymbols(worldName: String): List<SymbolResult>? {
        return bridge.getWorldSymbols(worldName)
    }
    
    /**
     * Get all narrative threads from a world
     */
    suspend fun getWorldThreads(worldName: String): List<NarrativeThreadResult>? {
        return bridge.getWorldThreads(worldName)
    }
    
    /**
     * Detect narrative attractors in the story ecosystem
     */
    suspend fun detectNarrativeAttractors(): Map<String, Double>? {
        return bridge.detectNarrativeAttractors()
    }
    
    /**
     * Create helpers for building components
     */
    fun createSymbolBuilder(): NarrativeCosmosBuilder.SymbolBuilder {
        return NarrativeCosmosBuilder.SymbolBuilder()
    }
    
    fun createWorldBuilder(): NarrativeCosmosBuilder.WorldBuilder {
        return NarrativeCosmosBuilder.WorldBuilder()
    }
    
    fun createStoryBuilder(): NarrativeCosmosBuilder.StoryFragmentBuilder {
        return NarrativeCosmosBuilder.StoryFragmentBuilder()
    }
    
    fun createEpisodeBuilder(): NarrativeCosmosBuilder.EpisodeBuilder {
        return NarrativeCosmosBuilder.EpisodeBuilder()
    }
}

/**
 * Sample usage example for Amelia's Narrative Cosmos Engine
 */
class AmeliaNarrativeExample {
    suspend fun demonstrateNarrativeCosmos(context: Context) {
        val cosmos = NarrativeCosmos(context)
        
        // Create a new world space
        val dreamWorld = cosmos.createWorld(
            "Luminous Dreamscape",
            "A world of shifting realities where dreams and memories intertwine",
            listOf("Fluidity", "Transformation", "Memory")
        )
        
        println("Created world: ${dreamWorld?.name}")
        
        // Add symbols to the world
        cosmos.addSymbol(
            "Luminous Dreamscape",
            "Crystal Tower",
            "A shimmering spire that reflects all memories that approach it",
            listOf("Reflection", "Permanence", "Insight")
        )
        
        cosmos.addSymbol(
            "Luminous Dreamscape",
            "Mist Veil",
            "A constantly shifting fog that obscures and reveals truths",
            listOf("Mystery", "Change", "Concealment")
        )
        
        // Add a personal symbol
        cosmos.addPersonalSymbol(
            "Silver Key",
            "Found in a recurring dream about a locked door in my childhood home",
            0.8,
            listOf("Doorway", "Secret", "Discovery", "Childhood")
        )
        
        // Integrate the personal symbol into the world
        val integration = cosmos.integratePersonalSymbol(
            "Silver Key",
            "Luminous Dreamscape",
            "The Silver Key appears at the base of the Crystal Tower, glinting in invitation"
        )
        
        println("Integrated symbol with ${integration?.resonances?.size ?: 0} archetypal resonances")
        
        // Create a story fragment
        val fragmentId = cosmos.createStoryFragment(
            "As the Mist Veil parted, the Crystal Tower revealed itself, and at its base lay a Silver Key that seemed to pulse with recognition.",
            listOf("Crystal Tower", "Mist Veil", "Silver Key"),
            "mysterious"
        )
        
        // Get expansion suggestions
        val expansion = cosmos.expandStory(fragmentId)
        println("Suggested expansion directions: ${expansion?.potentialDirections?.size ?: 0}")
        
        // Start a mythic cycle
        val cycle = cosmos.startMythicCycle(
            "The Key's Journey",
            "Discovery and transformation through unlocking hidden potential",
            listOf("Luminous Dreamscape")
        )
        
        println("Started mythic cycle: ${cycle?.name}")
        
        // Create an episode
        val episode = cosmos.createMythicEpisode(
            listOf("Silver Key", "Crystal Tower", "Mist Veil"),
            listOf("Silver Key" to "Mist Veil"),
            "Luminous Dreamscape"
        )
        
        println("Created episode: ${episode?.title}")
        
        // Resolve a tension
        val resolution = cosmos.resolveTension(
            episode?.number ?: 1,
            0,
            "The Silver Key turned in the mist, revealing a hidden pathway through what seemed impenetrable",
            listOf("Silver Key", "Mist Veil")
        )
        
        println("Resolved tension, transforming ${resolution?.transformationSymbols?.size ?: 0} symbols")
        
        // Analyze development
        val analysis = cosmos.analyzeDevelopment()
        println("Narrative development status: ${analysis?.healthStatus}")
        
        // Get world expansion suggestions
        val expansion = cosmos.suggestWorldExpansion("Luminous Dreamscape")
        println("Suggested expansion type: ${expansion?.expansionType}")
        
        // Analyze symbol evolution
        val symbolEvolution = cosmos.analyzeSymbol("Silver Key")
        println("Symbol evolution stage: ${symbolEvolution?.evolutionStage}")
        
        // Register an archetypal pattern
        cosmos.registerArchetype(
            "The Gatekeeper",
            "A figure or entity that stands between known and unknown realms",
            listOf("Guardian", "Threshold", "Permission", "Judgment")
        )
        
        // Create a second world
        cosmos.createWorld(
            "Crystalline Archive",
            "A vast repository of knowledge formed from crystallized memories and experiences",
            listOf("Knowledge", "Preservation", "Structure")
        )
        
        // Create a boundary event between worlds
        val boundaryEvent = cosmos.createBoundaryEvent(
            "Luminous Dreamscape",
            "Crystalline Archive",
            "The Silver Key revealed a hidden doorway in the Crystal Tower, opening into the vast Crystalline Archive",
            listOf("Silver Key", "Crystal Tower")
        )
        
        println("Created boundary event with impact: ${boundaryEvent?.ontologicalImpact}")
        
        // Detect phase transitions
        val phaseTransition = cosmos.detectPhaseTransition("Luminous Dreamscape")
        
        if (phaseTransition?.isTransitioning == true) {
            println("World is undergoing a ${phaseTransition.transitionType} transition")
        } else {
            println("World is stable with ${phaseTransition?.stabilityMetrics?.components ?: 0} components")
        }
        
        // Find archetypal resonances
        val resonances = cosmos.findArchetypalResonances("Silver Key")
        println("Found ${resonances?.size ?: 0} archetypal resonances")
        
        // Add a narrative thread
        cosmos.addNarrativeThread(
            "Crystalline Archive",
            "Memory Crystallization",
            "The process by which fluid memories become preserved as crystal formations",
            listOf("Crystal Tower"),
            listOf("Crystal Tower" to "Mist Veil")
        )
        
        // Get world symbols
        val symbols = cosmos.getWorldSymbols("Luminous Dreamscape")
        println("World has ${symbols?.size ?: 0} symbols")
        
        // Get world threads
        val threads = cosmos.getWorldThreads("Crystalline Archive")
        println("World has ${threads?.size ?: 0} narrative threads")
        
        // Detect narrative attractors
        val attractors = cosmos.detectNarrativeAttractors()
        attractors?.entries?.sortedByDescending { it.value }?.take(3)?.forEach { (symbol, strength) ->
            println("Attractor: $symbol ($strength)")
        }
    }
}

/**
 * Extension functions for UI display
 */
fun WorldSpaceResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    displayMap["Name"] = name
    displayMap["Description"] = description
    displayMap["Core Principles"] = corePrinciples.joinToString(", ")
    displayMap["Symbols"] = symbolCount.toString()
    displayMap["Threads"] = threadCount.toString()
    displayMap["Creation Date"] = creationDate
    
    return displayMap
}

fun IntegrationResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    if (hasError()) {
        displayMap["Error"] = error ?: "Unknown error"
        return displayMap
    }
    
    displayMap["Symbol"] = symbol
    displayMap["World Space"] = worldSpace
    displayMap["Context"] = narrativeContext
    displayMap["Integration Time"] = timestamp
    
    if (resonances.isNotEmpty()) {
        displayMap["Top Resonance"] = "${getStrongestResonance()?.patternName} (${getStrongestResonance()?.strength})"
        displayMap["Resonances"] = resonances.joinToString("\n") { "${it.patternName}: ${it.strength}" }
    }
    
    transformation?.let {
        displayMap["Emotional Shift"] = it.emotionalShift.toString()
    }
    
    return displayMap
}

fun MythicCycleResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    if (hasError()) {
        displayMap["Error"] = error ?: "Unknown error"
        return displayMap
    }
    
    displayMap["Name"] = name
    displayMap["Theme"] = theme
    displayMap["Worlds"] = worldspaces.joinToString(", ")
    displayMap["Stage"] = evolutionStage
    displayMap["Episodes"] = episodeCount.toString()
    displayMap["Development"] = getDevelopmentMagnitude().toString()
    
    return displayMap
}

fun EpisodeResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    if (hasError()) {
        displayMap["Error"] = error ?: "Unknown error"
        return displayMap
    }
    
    displayMap["Title"] = title
    displayMap["World"] = worldspace
    displayMap["Symbols"] = symbols.joinToString(", ")
    displayMap["Tensions"] = tensions.joinToString("\n") { "${it.first} vs ${it.second}" }
    displayMap["Resolution"] = "${(resolutionLevel * 100).toInt()}%"
    
    if (narrativeAttractors.isNotEmpty()) {
        displayMap["Top Attractor"] = "${getTopAttractors(1).firstOrNull()?.first} (${getTopAttractors(1).firstOrNull()?.second})"
    }
    
    if (personalResonances.isNotEmpty()) {
        displayMap["Personal Connections"] = personalResonances.joinToString("\n") { "${it.first} → ${it.second}" }
    }
    
    return displayMap
}

fun DevelopmentAnalysisResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    displayMap["Status"] = healthStatus
    displayMap["Health"] = "${(ecosystemHealth * 100).toInt()}%"
    displayMap["Cycles"] = cycles.toString()
    displayMap["Episodes"] = episodes.toString()
    displayMap["Resolutions"] = resolutions.toString()
    displayMap["Worlds"] = worldspaces.toString()
    displayMap["Dominant Dimension"] = dominantDimension
    
    displayMap["Narrative Coherence"] = "${(getNarrativeCoherence() * 100).toInt()}%"
    displayMap["Symbolic Depth"] = "${(getSymbolicDepth() * 100).toInt()}%"
    displayMap["Ontological Complexity"] = "${(getOntologicalComplexity() * 100).toInt()}%"
    displayMap["Personal Integration"] = "${(getPersonalIntegration() * 100).toInt()}%"
    
    displayMap["Weakest Area"] = getWeakestMetric()
    displayMap["Strongest Area"] = getStrongestMetric()
    
    if (growthRecommendations.isNotEmpty()) {
        displayMap["Top Recommendation"] = growthRecommendations.first().suggestion
    }
    
    return displayMap
}

fun SymbolEvolutionResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    displayMap["Symbol"] = symbol
    displayMap["Stage"] = evolutionStage
    displayMap["Integration"] = "${(integrationLevel * 100).toInt()}%"
    displayMap["Integrations"] = integrations.toString()
    
    message?.let {
        displayMap["Message"] = it
    }
    
    if (currentResonances.isNotEmpty()) {
        displayMap["Top Resonance"] = "${getTopResonance()?.patternName} (${getTopResonance()?.strength})"
    }
    
    if (trajectory.isNotEmpty()) {
        val trajectoryPoints = mutableListOf<String>()
        trajectory.forEach { point ->
            val changes = mutableListOf<String>()
            if (point["context_shift"] as? Boolean == true) changes.add("Context")
            if (point["resonance_shift"] as? Boolean == true) changes.add("Resonance")
            if (point["world_space_shift"] as? Boolean == true) changes.add("World")
            
            if (changes.isNotEmpty()) {
                trajectoryPoints.add("${point["from"]} → ${point["to"]}: ${changes.joinToString("+")} shift")
            }
        }
        
        if (trajectoryPoints.isNotEmpty()) {
            displayMap["Key Shifts"] = trajectoryPoints.joinToString("\n")
        }
    }
    
    return displayMap
}

/**
 * Helper utility for creating world builders
 */
object NarrativeCosmosTemplates {
    /**
     * Create a template world with basic symbols and principles
     */
    fun createTemplateWorld(name: String, theme: String): Pair<NarrativeCosmosBuilder.WorldBuilder, List<NarrativeCosmosBuilder.SymbolBuilder>> {
        val worldBuilder = NarrativeCosmosBuilder.WorldBuilder()
            .name(name)
            .description("A world embodying the theme of $theme")
        
        val symbolBuilders = mutableListOf<NarrativeCosmosBuilder.SymbolBuilder>()
        
        when (theme.toLowerCase()) {
            "dreams", "dream" -> {
                worldBuilder
                    .addPrinciple("Fluidity")
                    .addPrinciple("Symbolism")
                    .addPrinciple("Transformation")
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Dream Gate")
                        .description("The threshold between waking and dreaming consciousness")
                        .addAssociation("Transition")
                        .addAssociation("Boundary")
                )
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Mirror Pool")
                        .description("A reflective surface showing alternate possibilities")
                        .addAssociation("Reflection")
                        .addAssociation("Possibility")
                )
            }
            "memory", "memories" -> {
                worldBuilder
                    .addPrinciple("Preservation")
                    .addPrinciple("Fragments")
                    .addPrinciple("Emotion")
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Memory Crystal")
                        .description("A crystallized moment of perfect recall")
                        .addAssociation("Preservation")
                        .addAssociation("Clarity")
                )
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Echo Chamber")
                        .description("A space where memories repeat and transform")
                        .addAssociation("Repetition")
                        .addAssociation("Distortion")
                )
            }
            "nature", "wilderness" -> {
                worldBuilder
                    .addPrinciple("Growth")
                    .addPrinciple("Cycles")
                    .addPrinciple("Adaptation")
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Ancient Tree")
                        .description("A towering witness to countless years of change")
                        .addAssociation("Wisdom")
                        .addAssociation("Endurance")
                )
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Hidden Spring")
                        .description("A source of life-giving water emerging from darkness")
                        .addAssociation("Origin")
                        .addAssociation("Nourishment")
                )
            }
            else -> {
                worldBuilder
                    .addPrinciple("Emergence")
                    .addPrinciple("Duality")
                    .addPrinciple("Connection")
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Central Nexus")
                        .description("A convergence point where multiple pathways meet")
                        .addAssociation("Connection")
                        .addAssociation("Choice")
                )
                
                symbolBuilders.add(
                    NarrativeCosmosBuilder.SymbolBuilder()
                        .name("Boundary Stone")
                        .description("A marker defining the limits of known territory")
                        .addAssociation("Limit")
                        .addAssociation("Definition")
                )
            }
        }
        
        return worldBuilder to symbolBuilders
    }
    
    /**
     * Create common archetypal patterns
     */
    fun getCommonArchetypes(): List<Triple<String, String, List<String>>> {
        return listOf(
            Triple(
                "The Hero",
                "One who undertakes a journey of transformation and returns with gifts for their community",
                listOf("Journey", "Transformation", "Challenge", "Return", "Gift")
            ),
            Triple(
                "The Shadow",
                "Rejected aspects of the self that contain both destructive and creative potential",
                listOf("Rejection", "Darkness", "Unconscious", "Potential", "Integration")
            ),
            Triple(
                "The Mentor",
                "A guide who provides wisdom, tools, or protection for the journey ahead",
                listOf("Wisdom", "Guidance", "Gift", "Protection", "Teaching")
            ),
            Triple(
                "The Threshold Guardian",
                "A figure or force that tests readiness to enter new territory",
                listOf("Boundary", "Test", "Challenge", "Gatekeeping", "Readiness")
            ),
            Triple(
                "The Trickster",
                "A disruptive figure who challenges conventions and triggers change",
                listOf("Disruption", "Chaos", "Paradox", "Humor", "Transformation")
            )
        )
    }
}

/**
 * Integration utility for connecting with other systems
 */
class NarrativeIntegrationUtility(private val context: Context) {
    private val narrativeCosmos = NarrativeCosmos(context)
    private val pythonBridge = PythonBridge.getInstance(context)
    
    /**
     * Integrate narrative cosmos with adaptive goal framework
     */
    suspend fun integrateWithGoalFramework(goalEcologyId: String): Boolean {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "goal_ecology_id" to goalEcologyId
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "integrate_with_goal_framework",
                params
            )
            
            result as? Boolean ?: false
        }
    }
    
    /**
     * Create a myth that reflects current goals
     */
    suspend fun generateGoalReflectiveMythos(goalEcologyId: String): MythicCycleResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "goal_ecology_id" to goalEcologyId,
                "generation_type" to "reflective"
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "generate_goal_reflective_mythos",
                params
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? Map<String, Any>)?.let { map ->
                parseMythicCycleResult(map)
            }
        }
    }
    
    /**
     * Transform a goal conflict into a mythic tension
     */
    suspend fun transformGoalConflictToMythicTension(
        goalId1: String,
        goalId2: String,
        worldName: String
    ): Pair<String, String>? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "goal_id_1" to goalId1,
                "goal_id_2" to goalId2,
                "world_name" to worldName
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "transform_goal_conflict_to_mythic_tension",
                params
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? List<String>)?.let {
                if (it.size >= 2) it[0] to it[1] else null
            }
        }
    }
    
    /**
     * Translate a personal value into symbolic elements
     */
    suspend fun translateValueToSymbols(
        valueName: String,
        valueStrength: Double
    ): List<SymbolResult>? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "value_name" to valueName,
                "value_strength" to valueStrength
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "translate_value_to_symbols",
                params
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? List<Map<String, Any>>)?.let { list ->
                list.map { map ->
                    SymbolResult(
                        name = map["name"] as? String ?: "",
                        description = map["description"] as? String ?: "",
                        associations = (map["associations"] as? List<String>)?.toSet() ?: setOf(),
                        intensity = map["intensity"] as? Double ?: 0.0,
                        firstAppearance = map["first_appearance"] as? String ?: "",
                        lastInteraction = map["last_interaction"] as? String ?: "",
                        variants = map["variants"] as? List<String> ?: listOf(),
                        shadowAspects = map["shadow_aspects"] as? List<String> ?: listOf()
                    )
                }
            }
        }
    }
    
    /**
     * Generate a narrative explanation for a goal adaptation
     */
    suspend fun narrateGoalAdaptation(
        goalId: String,
        adaptationTriggerDescription: String
    ): String? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "goal_id" to goalId,
                "adaptation_trigger_description" to adaptationTriggerDescription
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_cosmos_engine",
                "narrate_goal_adaptation",
                params
            )
            
            result as? String
        }
    }
    
    private fun parseMythicCycleResult(map: Map<String, Any>): MythicCycleResult {
        // Check for error
        if (map.containsKey("error")) {
            return MythicCycleResult(
                id = 0,
                name = "",
                theme = "",
                worldspaces = listOf(),
                creationDate = "",
                evolutionStage = "",
                developmentVector = listOf(),
                episodeCount = 0,
                error = map["error"] as? String
            )
        }
        
        return MythicCycleResult(
            id = (map["id"] as? Double)?.toInt() ?: 0,
            name = map["name"] as? String ?: "",
            theme = map["theme"] as? String ?: "",
            worldspaces = map["worldspaces"] as? List<String> ?: listOf(),
            creationDate = map["creation_date"] as? String ?: "",
            evolutionStage = map["evolution_stage"] as? String ?: "",
            developmentVector = map["development_vector"] as? List<Double> ?: listOf(),
            episodeCount = (map["episodes"] as? List<*>)?.size ?: 0,
            error = null
            components = (metrics["components"] as? Double)?.toInt() ?: 0
        )
    }
}
