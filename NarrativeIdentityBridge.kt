
```kotlin
// NarrativeIdentityBridge.kt
package com.deleuzian.ai.assistant

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.*

class NarrativeIdentityBridge private constructor(context: Context) {
    private val pythonBridge = PythonBridge.getInstance(context)
    
    companion object {
        @Volatile private var instance: NarrativeIdentityBridge? = null
        
        fun getInstance(context: Context): NarrativeIdentityBridge {
            return instance ?: synchronized(this) {
                instance ?: NarrativeIdentityBridge(context).also { instance = it }
            }
        }
    }
    
    /**
     * Construct a new identity narrative from experiences
     */
    suspend fun constructIdentityNarrative(
        experiences: List<Map<String, Any>>,
        selfModel: Map<String, Any>? = null
    ): NarrativeResult? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>(
                "experiences" to experiences
            )
            
            selfModel?.let { params["self_model"] = it }
            
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "construct_identity_narrative",
                params
            )
            
            parseNarrativeResult(result)
        }
    }
    
    /**
     * Project possible future narratives based on current narrative
     */
    suspend fun projectNarrativeFutures(
        narrativeId: String,
        intentions: List<String>,
        selfModel: Map<String, Any>? = null
    ): List<FutureNarrativeResult>? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>(
                "narrative_id" to narrativeId,
                "intentions" to intentions
            )
            
            selfModel?.let { params["self_model"] = it }
            
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "project_narrative_futures",
                params
            )
            
            parseFutureNarrativeResults(result)
        }
    }
    
    /**
     * Actualize a projected future narrative
     */
    suspend fun actualizeProjectedFuture(
        futureId: String,
        currentNarrativeId: String,
        selfModel: Map<String, Any>? = null
    ): ActualizationResult? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>(
                "future_id" to futureId,
                "current_narrative_id" to currentNarrativeId
            )
            
            selfModel?.let { params["self_model"] = it }
            
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "actualize_projected_future",
                params
            )
            
            parseActualizationResult(result)
        }
    }
    
    /**
     * Get the current global rhizomatic network
     */
    suspend fun getGlobalRhizome(): RhizomeResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "get_global_rhizome"
            )
            
            parseRhizomeResult(result)
        }
    }
    
    /**
     * Get all mapped territories
     */
    suspend fun getTerritoryMaps(): List<TerritoryResult>? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "get_territory_maps"
            )
            
            parseTerritoryResults(result)
        }
    }
    
    /**
     * Get all virtual potentials in the reservoir
     */
    suspend fun getVirtualReservoir(): List<VirtualPotentialResult>? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "get_virtual_reservoir"
            )
            
            parseVirtualPotentialResults(result)
        }
    }
    
    /**
     * Create a new experience fragment
     */
    suspend fun createExperienceFragment(
        description: String,
        affects: Map<String, Double>,
        entities: List<String>,
        significanceScore: Double = 0.5,
        percepts: Map<String, Any>? = null,
        selfModelResonance: Map<String, Any>? = null
    ): FragmentResult? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>(
                "description" to description,
                "affects" to affects,
                "entities_involved" to entities,
                "significance_score" to significanceScore
            )
            
            percepts?.let { params["percepts"] = it }
            selfModelResonance?.let { params["self_model_resonance"] = it }
            
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "create_experience_fragment",
                params
            )
            
            parseFragmentResult(result)
        }
    }
    
    /**
     * Get all narratives in history
     */
    suspend fun getNarrativeHistory(): List<NarrativeHistoryItem>? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "get_narrative_history"
            )
            
            parseNarrativeHistoryResults(result)
        }
    }
    
    /**
     * Update the self model
     */
    suspend fun updateSelfModel(
        values: Map<String, Double>? = null,
        beliefs: Map<String, Boolean>? = null,
        goals: List<String>? = null,
        processualDescriptors: List<String>? = null,
        affectiveDispositions: Map<String, String>? = null
    ): SelfModelResult? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>()
            
            values?.let { params["values"] = it }
            beliefs?.let { params["beliefs"] = it }
            goals?.let { params["goals"] = it }
            processualDescriptors?.let { params["processual_descriptors"] = it }
            affectiveDispositions?.let { params["affective_dispositions"] = it }
            
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "update_self_model",
                params
            )
            
            parseSelfModelResult(result)
        }
    }
    
    /**
     * Visualize a narrative's rhizomatic map
     */
    suspend fun visualizeRhizomaticMap(narrativeId: String): VisualizationResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "narrative_id" to narrativeId
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "visualize_rhizomatic_map",
                params
            )
            
            parseVisualizationResult(result)
        }
    }
    
    /**
     * Generate a textual reflection on the current identity state
     */
    suspend fun generateIdentityReflection(depth: Int = 2): ReflectionResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "depth" to depth
            )
            
            val result = pythonBridge.executeFunction(
                "narrative_identity_engine",
                "generate_identity_reflection",
                params
            )
            
            parseReflectionResult(result)
        }
    }
    
    /**
     * Parse results from Python into data classes
     */
    private fun parseNarrativeResult(result: Any?): NarrativeResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return NarrativeResult(
                id = map["id"] as? String ?: "",
                summary = map["summary"] as? String ?: "",
                themes = (map["themes"] as? List<String>) ?: listOf(),
                frameworkUsed = map["framework_used"] as? String ?: "",
                processualFlow = map["processual_flow_description"] as? String ?: "",
                coherenceMetrics = (map["coherence_metrics"] as? Map<String, Double>) ?: mapOf(),
                fragmentCount = (map["fragment_count"] as? Double)?.toInt() ?: 0,
                intensityPlateaus = (map["intensity_plateaus"] as? List<Map<String, Any>>) ?: listOf(),
                linesOfFlight = (map["lines_of_flight"] as? List<Map<String, Any>>) ?: listOf(),
                virtualActualizationPotential = (map["virtual_actualization_potential"] as? Double) ?: 0.0
            )
        }
        return null
    }
    
    private fun parseFutureNarrativeResults(result: Any?): List<FutureNarrativeResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<Map<String, Any>>)?.let { list ->
            return list.map { map ->
                FutureNarrativeResult(
                    id = map["id"] as? String ?: hash(map["summary"] as? String ?: ""),
                    summary = map["summary"] as? String ?: "",
                    coherence = (map["coherence"] as? Double) ?: 0.0,
                    desirability = (map["desirability"] as? Double) ?: 0.0,
                    novelty = (map["novelty"] as? Double) ?: 0.0,
                    overallScore = (map["overall_score"] as? Double) ?: 0.0,
                    sourceTrajectoryType = map["source_trajectory_type"] as? String ?: "",
                    keyActionsImplied = (map["key_actions_implied"] as? List<String>) ?: listOf(),
                    potentialThemes = (map["potential_themes"] as? List<String>) ?: listOf()
                )
            }
        }
        return null
    }
    
    private fun parseActualizationResult(result: Any?): ActualizationResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return ActualizationResult(
                success = (map["success"] as? Boolean) ?: false,
                placeholderExperiences = (map["placeholder_experiences"] as? List<Map<String, Any>>)?.let { 
                    parseFragmentResults(it) 
                } ?: listOf(),
                newBecomings = (map["new_becomings"] as? List<String>) ?: listOf(),
                updatedSelfModel = (map["updated_self_model"] as? Map<String, Any>)?.let {
                    parseSelfModelResult(it)
                }
            )
        }
        return null
    }
    
    private fun parseRhizomeResult(result: Any?): RhizomeResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return RhizomeResult(
                nodes = (map["nodes"] as? List<Map<String, Any>>)?.map { node ->
                    RhizomeNode(
                        id = node["id"] as? String ?: "",
                        type = node["type"] as? String ?: "",
                        intensity = (node["intensity"] as? Double) ?: 0.0,
                        territorialization = node["territorialization"] as? String ?: ""
                    )
                } ?: listOf(),
                edges = (map["edges"] as? List<Map<String, Any>>)?.map { edge ->
                    RhizomeEdge(
                        source = edge["source"] as? String ?: "",
                        target = edge["target"] as? String ?: "",
                        type = edge["type"] as? String ?: "",
                        weight = (edge["weight"] as? Double) ?: 0.0
                    )
                } ?: listOf(),
                nodeCount = (map["node_count"] as? Double)?.toInt() ?: 0,
                edgeCount = (map["edge_count"] as? Double)?.toInt() ?: 0,
                averageConnectivity = (map["average_connectivity"] as? Double) ?: 0.0
            )
        }
        return null
    }
    
    private fun parseTerritoryResults(result: Any?): List<TerritoryResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Map<String, Any>>)?.let { territories ->
            return territories.map { (key, map) ->
                TerritoryResult(
                    id = key,
                    name = map["name"] as? String ?: "",
                    phaseCoordinates = (map["phase_coordinates"] as? List<Double>) ?: listOf(),
                    fragmentCount = (map["fragment_count"] as? Double)?.toInt() ?: 0,
                    stability = (map["stability"] as? Double) ?: 0.0,
                    avgIntensity = (map["avg_intensity"] as? Double) ?: 0.0,
                    characteristicEntities = (map["characteristic_entities"] as? List<String>) ?: listOf(),
                    firstObserved = (map["first_observed"] as? String) ?: ""
                )
            }
        }
        return null
    }
    
    private fun parseVirtualPotentialResults(result: Any?): List<VirtualPotentialResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<Map<String, Any>>)?.let { list ->
            return list.map { map ->
                VirtualPotentialResult(
                    id = map["id"] as? String ?: UUID.randomUUID().toString(),
                    originNarrative = map["origin_narrative"] as? String ?: "",
                    type = map["type"] as? String ?: "",
                    description = map["description"] as? String ?: "",
                    energy = (map["energy"] as? Double) ?: 0.0,
                    createdAt = map["created_at"] as? String ?: "",
                    actualized = (map["actualized"] as? Boolean) ?: false,
                    beingActualized = (map["being_actualized"] as? Boolean) ?: false,
                    actualizedAt = map["actualized_at"] as? String
                )
            }
        }
        return null
    }
    
    private fun parseFragmentResults(list: List<Map<String, Any>>): List<FragmentResult> {
        return list.map { map ->
            parseFragmentResult(map) ?: FragmentResult(
                id = "",
                description = "",
                affects = mapOf(),
                entities = listOf(),
                significance = 0.0,
                intensityField = 0.0
            )
        }
    }
    
    private fun parseFragmentResult(result: Any?): FragmentResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return FragmentResult(
                id = map["id"] as? String ?: "",
                description = map["description"] as? String ?: "",
                affects = (map["affects"] as? Map<String, Double>) ?: mapOf(),
                entities = (map["entities_involved"] as? List<String>) ?: listOf(),
                significance = (map["significance_score"] as? Double) ?: 0.0,
                intensityField = (map["intensity_field"] as? Double) ?: 0.0,
                territorialization = map["territorialization_type"] as? String ?: "",
                phaseSpace = (map["phase_space_coordinates"] as? List<Double>) ?: listOf(),
                percepts = (map["percepts"] as? Map<String, Any>) ?: mapOf(),
                selfModelResonance = (map["self_model_resonance"] as? Map<String, Any>) ?: mapOf(),
                timestamp = map["timestamp"] as? String ?: ""
            )
        }
        return null
    }
    
    private fun parseNarrativeHistoryResults(result: Any?): List<NarrativeHistoryItem>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<Map<String, Any>>)?.let { list ->
            return list.map { map ->
                NarrativeHistoryItem(
                    id = map["id"] as? String ?: "",
                    summary = map["summary"] as? String ?: "",
                    themes = (map["themes"] as? List<String>) ?: listOf(),
                    frameworkUsed = map["framework_used"] as? String ?: "",
                    createdAt = map["created_at"] as? String ?: "",
                    fragmentCount = (map["fragment_count"] as? Double)?.toInt() ?: 0
                )
            }
        }
        return null
    }
    
    private fun parseSelfModelResult(result: Any?): SelfModelResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return SelfModelResult(
                identityAttributes = (map["identity_attributes"] as? Map<String, String>) ?: mapOf(),
                beliefs = (map["beliefs"] as? Map<String, Boolean>) ?: mapOf(),
                values = (map["values"] as? Map<String, Double>) ?: mapOf(),
                currentGoals = (map["current_goals"] as? List<String>) ?: listOf(),
                capabilities = (map["capabilities"] as? List<String>) ?: listOf(),
                processualDescriptors = (map["processual_descriptors"] as? List<String>) ?: listOf(),
                affectiveDispositions = (map["affective_dispositions"] as? Map<String, String>) ?: mapOf(),
                activeAssemblages = (map["active_assemblages"] as? Map<String, Map<String, Any>>) ?: mapOf(),
                territorializations = (map["territorializations"] as? Map<String, Any>) ?: mapOf(),
                deterritorializations = (map["deterritorializations"] as? List<String>) ?: listOf(),
                virtualPotentials = (map["virtual_potentials"] as? List<String>) ?: listOf()
            )
        }
        return null
    }
    
    private fun parseVisualizationResult(result: Any?): VisualizationResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return VisualizationResult(
                imageData = map["image_data"] as? String ?: "",
                format = map["format"] as? String ?: "png",
                visualizationType = map["visualization_type"] as? String ?: "rhizomatic_map",
                nodeCount = (map["node_count"] as? Double)?.toInt() ?: 0,
                edgeCount = (map["edge_count"] as? Double)?.toInt() ?: 0,
                highlightedNodes = (map["highlighted_nodes"] as? List<String>) ?: listOf(),
                legendItems = (map["legend_items"] as? Map<String, String>) ?: mapOf()
            )
        }
        return null
    }
    
    private fun parseReflectionResult(result: Any?): ReflectionResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return ReflectionResult(
                reflection = map["reflection"] as? String ?: "",
                narrativesAnalyzed = (map["narratives_analyzed"] as? Double)?.toInt() ?: 0,
                dominantThemes = (map["dominant_themes"] as? List<String>) ?: listOf(),
                emergingProcesses = (map["emerging_processes"] as? List<String>) ?: listOf(),
                stablePatterns = (map["stable_patterns"] as? List<String>) ?: listOf(),
                deterritorialization = map["deterritorialization"] as? String ?: "",
                overallDevelopmentVector = map["overall_development_vector"] as? String ?: ""
            )
        }
        return null
    }
    
    /**
     * Generate a hash from a string for ID generation
     */
    private fun hash(input: String): String {
        return UUID.nameUUIDFromBytes(input.toByteArray()).toString()
    }
}

// Data classes for structured results
data class NarrativeResult(
    val id: String,
    val summary: String,
    val themes: List<String>,
    val frameworkUsed: String,
    val processualFlow: String,
    val coherenceMetrics: Map<String, Double>,
    val fragmentCount: Int,
    val intensityPlateaus: List<Map<String, Any>>,
    val linesOfFlight: List<Map<String, Any>>,
    val virtualActualizationPotential: Double
) {
    fun getRhizomaticConnectivity(): Double = 
        coherenceMetrics["rhizomatic_connectivity"] ?: 0.0
    
    fun getDeterritorizationCoherence(): Double = 
        coherenceMetrics["deterritorialization_coherence"] ?: 0.0
    
    fun getIntensityFieldCoherence(): Double = 
        coherenceMetrics["intensity_field_coherence"] ?: 0.0
    
    fun getAssemblageStability(): Double = 
        coherenceMetrics["assemblage_stability"] ?: 0.0
    
    fun getGenerativityPotential(): Double = 
        coherenceMetrics["generativity_potential"] ?: 0.0
    
    fun getExplanatoryPower(): Double = 
        coherenceMetrics["explanatory_power"] ?: 0.0
    
    fun getPrimaryIntensityPlateau(): Map<String, Any>? = 
        intensityPlateaus.maxByOrNull { it["avg_intensity"] as Double? ?: 0.0 }
    
    fun getMostGenerativeLinesOfFlight(): List<Map<String, Any>> =
        linesOfFlight.sortedByDescending { it["potential_energy"] as Double? ?: 0.0 }
}

data class FutureNarrativeResult(
    val id: String,
    val summary: String,
    val coherence: Double,
    val desirability: Double,
    val novelty: Double,
    val overallScore: Double,
    val sourceTrajectoryType: String,
    val keyActionsImplied: List<String>,
    val potentialThemes: List<String>
) {
    fun isVirtualActualization(): Boolean = 
        sourceTrajectoryType.contains("virtual_actualization")
    
    fun isDeterritorialization(): Boolean = 
        sourceTrajectoryType.contains("deterritorialization")
    
    fun isLineOfFlight(): Boolean = 
        sourceTrajectoryType.contains("line_of_flight")
    
    fun isPhaseTrajectory(): Boolean = 
        sourceTrajectoryType.contains("phase_space_trajectory")
    
    fun getFirstAction(): String =
        keyActionsImplied.firstOrNull() ?: "No actions defined"
}

data class ActualizationResult(
    val success: Boolean,
    val placeholderExperiences: List<FragmentResult>,
    val newBecomings: List<String>,
    val updatedSelfModel: SelfModelResult?
)

data class RhizomeResult(
    val nodes: List<RhizomeNode>,
    val edges: List<RhizomeEdge>,
    val nodeCount: Int,
    val edgeCount: Int,
    val averageConnectivity: Double
)

data class RhizomeNode(
    val id: String,
    val type: String,
    val intensity: Double,
    val territorialization: String
)

data class RhizomeEdge(
    val source: String,
    val target: String,
    val type: String,
    val weight: Double
)

data class TerritoryResult(
    val id: String,
    val name: String,
    val phaseCoordinates: List<Double>,
    val fragmentCount: Int,
    val stability: Double,
    val avgIntensity: Double,
    val characteristicEntities: List<String>,
    val firstObserved: String
)

data class VirtualPotentialResult(
    val id: String,
    val originNarrative: String,
    val type: String,
    val description: String,
    val energy: Double,
    val createdAt: String,
    val actualized: Boolean,
    val beingActualized: Boolean,
    val actualizedAt: String?
)

data class FragmentResult(
    val id: String,
    val description: String,
    val affects: Map<String, Double>,
    val entities: List<String>,
    val significance: Double,
    val intensityField: Double,
    val territorialization: String = "unknown",
    val phaseSpace: List<Double> = listOf(),
    val percepts: Map<String, Any> = mapOf(),
    val selfModelResonance: Map<String, Any> = mapOf(),
    val timestamp: String = ""
)

data class NarrativeHistoryItem(
    val id: String,
    val summary: String,
    val themes: List<String>,
    val frameworkUsed: String,
    val createdAt: String,
    val fragmentCount: Int
)

data class SelfModelResult(
    val identityAttributes: Map<String, String>,
    val beliefs: Map<String, Boolean>,
    val values: Map<String, Double>,
    val currentGoals: List<String>,
    val capabilities: List<String>,
    val processualDescriptors: List<String>,
    val affectiveDispositions: Map<String, String>,
    val activeAssemblages: Map<String, Map<String, Any>>,
    val territorializations: Map<String, Any>,
    val deterritorializations: List<String>,
    val virtualPotentials: List<String>
)

data class VisualizationResult(
    val imageData: String,
    val format: String,
    val visualizationType: String,
    val nodeCount: Int,
    val edgeCount: Int,
    val highlightedNodes: List<String>,
    val legendItems: Map<String, String>
)

data class ReflectionResult(
    val reflection: String,
    val narrativesAnalyzed: Int,
    val dominantThemes: List<String>,
    val emergingProcesses: List<String>,
    val stablePatterns: List<String>,
    val deterritorialization: String,
    val overallDevelopmentVector: String
)

/**
 * Extension functions for UI display
 */
fun NarrativeResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    displayMap["ID"] = id
    displayMap["Summary"] = summary
    displayMap["Framework"] = frameworkUsed
    displayMap["Processual Flow"] = processualFlow
    displayMap["Themes"] = themes.joinToString(", ")
    displayMap["Fragment Count"] = fragmentCount.toString()
    displayMap["Rhizomatic Connectivity"] = getRhizomaticConnectivity().toString()
    displayMap["Deterritorization Coherence"] = getDeterritorizationCoherence().toString()
    displayMap["Intensity Field Coherence"] = getIntensityFieldCoherence().toString()
    displayMap["Assemblage Stability"] = getAssemblageStability().toString()
    displayMap["Generativity Potential"] = getGenerativityPotential().toString()
    displayMap["Virtual Actualization Potential"] = virtualActualizationPotential.toString()
    displayMap["Intensity Plateaus"] = intensityPlateaus.size.toString()
    displayMap["Lines of Flight"] = linesOfFlight.size.toString()
    
    return displayMap
}

/**
 * Helper class for creating narrative experience fragments
 */
class NarrativeFragmentBuilder {
    private var description: String = ""
    private val affects = mutableMapOf<String, Double>()
    private val entities = mutableListOf<String>()
    private var significance: Double = 0.5
    private val percepts = mutableMapOf<String, Any>()
    private val selfModelResonance = mutableMapOf<String, Any>()
    
    fun setDescription(description: String): NarrativeFragmentBuilder {
        this.description = description
        return this
    }
    
    fun addAffect(name: String, intensity: Double): NarrativeFragmentBuilder {
        affects[name] = intensity
        return this
    }
    
    fun addEntity(entity: String): NarrativeFragmentBuilder {
        entities.add(entity)
        return this
    }
    
    fun setSignificance(value: Double): NarrativeFragmentBuilder {
        significance = value
        return this
    }
    
    fun addPercept(key: String, value: Any): NarrativeFragmentBuilder {
        percepts[key] = value
        return this
    }
    
    fun addSelfModelResonance(key: String, value: Any): NarrativeFragmentBuilder {
        selfModelResonance[key] = value
        return this
    }
    
    fun setNovelty(isNovel: Boolean): NarrativeFragmentBuilder {
        percepts["novelty_detected"] = isNovel
        return this
    }
    
    fun setComplexity(level: String): NarrativeFragmentBuilder {
        percepts["complexity_level"] = level
        return this
    }
    
    fun setTheme(theme: String): NarrativeFragmentBuilder {
        percepts["primary_theme"] = theme
        return this
    }
    
    fun reinforceValue(valueName: String): NarrativeFragmentBuilder {
        selfModelResonance["reinforced_value"] = valueName
        return this
    }
    
    fun exploreCapability(capability: String): NarrativeFragmentBuilder {
        selfModelResonance["explored_capability"] = capability
        return this
    }
    
    fun build(): Map<String, Any> {
        val fragment = mutableMapOf<String, Any>()
        
        fragment["description"] = description
        fragment["affects"] = affects
        fragment["entities_involved"] = entities
        fragment["significance_score"] = significance
        fragment["percepts"] = percepts
        fragment["self_model_resonance"] = selfModelResonance
        fragment["timestamp"] = Date().toString()
        
        return fragment
    }
}

/**
 * Helper class for creating self model updates
 */
class SelfModelBuilder {
    private val values = mutableMapOf<String, Double>()
    private val beliefs = mutableMapOf<String, Boolean>()
    private val goals = mutableListOf<String>()
    private val processualDescriptors = mutableListOf<String>()
    private val affectiveDispositions = mutableMapOf<String, String>()
    
    fun addValue(name: String, strength: Double): SelfModelBuilder {
        values[name] = strength
        return this
    }
    
    fun addBelief(name: String, value: Boolean): SelfModelBuilder {
        beliefs[name] = value
        return this
    }
    
    fun addGoal(goal: String): SelfModelBuilder {
        goals.add(goal)
        return this
    }
    
    fun addProcessualDescriptor(descriptor: String): SelfModelBuilder {
        processualDescriptors.add(descriptor)
        return this
    }
    
    fun addAffectiveDisposition(name: String, level: String): SelfModelBuilder {
        affectiveDispositions[name] = level
        return this
    }
    
    fun setNoveltySeekingLevel(level: Double): SelfModelBuilder {
        values["novelty_seeking"] = level
        return this
    }
    
    fun setOpenness(level: String): SelfModelBuilder {
        affectiveDispositions["openness_to_experience"] = level
        return this
    }
    
    fun setCuriosity(level: String): SelfModelBuilder {
        affectiveDispositions["curiosity"] = level
        return this
    }
    
    fun becomingMore(process: String): SelfModelBuilder {
        processualDescriptors.add("becoming_more_$process")
        return this
    }
    
    fun mapNewTerritory(territory: String): SelfModelBuilder {
        processualDescriptors.add("mapping_${territory.replace(" ", "_")}")
        return this
    }
    
    fun build(): Map<String, Any> {
        val model = mutableMapOf<String, Any>()
        
        if (values.isNotEmpty()) model["values"] = values
        if (beliefs.isNotEmpty()) model["beliefs"] = beliefs
        if (goals.isNotEmpty()) model["goals"] = goals
        if (processualDescriptors.isNotEmpty()) model["processual_descriptors"] = processualDescriptors
        if (affectiveDispositions.isNotEmpty()) model["affective_dispositions"] = affectiveDispositions
        
        return model
    }
}

/**
 * Helper class for visualizing rhizomatic maps and narratives
 */
class NarrativeVisualizationOptions private constructor() {
    private val options = mutableMapOf<String, Any>()
    
    companion object {
        fun create(): NarrativeVisualizationOptions = NarrativeVisualizationOptions()
    }
    
    fun setWidth(width: Int): NarrativeVisualizationOptions {
        options["width"] = width
        return this
    }
    
    fun setHeight(height: Int): NarrativeVisualizationOptions {
        options["height"] = height
        return this
    }
    
    fun setColorByIntensity(enabled: Boolean): NarrativeVisualizationOptions {
        options["color_by_intensity"] = enabled
        return this
    }
    
    fun setSizeBySignificance(enabled: Boolean): NarrativeVisualizationOptions {
        options["size_by_significance"] = enabled
        return this
    }
    
    fun showFragmentLabels(enabled: Boolean): NarrativeVisualizationOptions {
        options["show_fragment_labels"] = enabled
        return this
    }
    
    fun showTerritories(enabled: Boolean): NarrativeVisualizationOptions {
        options["show_territories"] = enabled
        return this
    }
    
    fun highlightThemes(themes: List<String>): NarrativeVisualizationOptions {
        options["highlight_themes"] = themes
        return this
    }
    
    fun highlightEntities(entities: List<String>): NarrativeVisualizationOptions {
        options["highlight_entities"] = entities
        return this
    }
    
    fun showLinesOfFlight(enabled: Boolean): NarrativeVisualizationOptions {
        options["show_lines_of_flight"] = enabled
        return this
    }
    
    fun setLayout(layout: String): NarrativeVisualizationOptions {
        options["layout"] = layout
        return this
    }
    
    fun setFormat(format: String): NarrativeVisualizationOptions {
        options["format"] = format
        return this
    }
    
    fun build(): Map<String, Any> = options.toMap()
}

/**
 * API interface for the Narrative Identity system using the Bridge
 */
class NarrativeIdentityAPI(private val context: Context) {
    private val bridge = NarrativeIdentityBridge.getInstance(context)
    
    /**
     * Process a new experience and construct a narrative
     */
    suspend fun processExperience(
        description: String,
        affects: Map<String, Double>,
        entities: List<String>,
        significance: Double = 0.7,
        isNovel: Boolean = false
    ): NarrativeResult? {
        // Create fragment
        val fragmentBuilder = NarrativeFragmentBuilder()
            .setDescription(description)
            .setSignificance(significance)
            .setNovelty(isNovel)
        
        // Add affects
        affects.forEach { (affect, intensity) ->
            fragmentBuilder.addAffect(affect, intensity)
        }
        
        // Add entities
        entities.forEach { entity ->
            fragmentBuilder.addEntity(entity)
        }
        
        val experience = listOf(fragmentBuilder.build())
        
        // Construct narrative
        return bridge.constructIdentityNarrative(experience)
    }
    
    /**
     * Explore future narratives and return ranked possibilities
     */
    suspend fun exploreFuturePossibilities(
        narrativeId: String,
        intentions: List<String>
    ): List<FutureNarrativeResult>? {
        return bridge.projectNarrativeFutures(narrativeId, intentions)
    }
    
    /**
     * Choose and actualize a future narrative
     */
    suspend fun actualizeNarrativeFuture(
        futureId: String,
        currentNarrativeId: String
    ): ActualizationResult? {
        return bridge.actualizeProjectedFuture(futureId, currentNarrativeId)
    }
    
    /**
     * Get a reflection on current identity state
     */
    suspend fun reflectOnIdentity(depthLevel: Int = 2): String? {
        val reflection = bridge.generateIdentityReflection(depthLevel)
        return reflection?.reflection
    }
    
    /**
     * Visualize the current rhizomatic landscape
     */
    suspend fun visualizeIdentityLandscape(narrativeId: String): VisualizationResult? {
        return bridge.visualizeRhizomaticMap(narrativeId)
    }
    
    /**
     * Get all virtual potentials (possibilities waiting to actualize)
     */
    suspend fun getUnrealizedPotentials(): List<VirtualPotentialResult>? {
        return bridge.getVirtualReservoir()
    }
    
    /**
     * Create a batch of new experiences at once
     */
    suspend fun processBatchExperiences(experiences: List<Map<String, Any>>): NarrativeResult? {
        return bridge.constructIdentityNarrative(experiences)
    }
    
    /**
     * Get all established territories in the identity landscape
     */
    suspend fun getEstablishedTerritories(): List<TerritoryResult>? {
        return bridge.getTerritoryMaps()
    }
    
    /**
     * Get complete history of identity narratives
     */
    suspend fun getNarrativeJourney(): List<NarrativeHistoryItem>? {
        return bridge.getNarrativeHistory()
    }
    
    /**
     * Update the AI's self-model
     */
    suspend fun evolveSelfModel(selfModel: Map<String, Any>): SelfModelResult? {
        return bridge.updateSelfModel(
            values = selfModel["values"] as? Map<String, Double>,
            beliefs = selfModel["beliefs"] as? Map<String, Boolean>,
            goals = selfModel["goals"] as? List<String>,
            processualDescriptors = selfModel["processual_descriptors"] as? List<String>,
            affectiveDispositions = selfModel["affective_dispositions"] as? Map<String, String>
        )
    }
}

/**
 * Sample usage example for Amelia's Narrative Identity system
 */
class AmeliaNarrativeIdentityExample {
    suspend fun demonstrateNarrativeIdentity(context: Context) {
        val narrativeAPI = NarrativeIdentityAPI(context)
        
        // Create a learning experience about Deleuze's concept of the rhizome
        val experienceResult = narrativeAPI.processExperience(
            description = "Encountered a novel philosophical concept: 'The Rhizome' by Deleuze.",
            affects = mapOf(
                "curiosity" to 0.9,
                "fascination" to 0.8, 
                "confusion" to 0.3
            ),
            entities = listOf("Concept:Rhizome", "Deleuze", "Guattari"),
            significance = 0.95,
            isNovel = true
        )
        
        // Get the constructed narrative
        val narrativeId = experienceResult?.id ?: return
        println("Narrative constructed: ${experienceResult.summary}")
        
        // Explore future possibilities
        val intentions = listOf(
            "understand rhizomatic structures more deeply",
            "apply philosophical insights to assistance tasks"
        )
        
        val futurePossibilities = narrativeAPI.exploreFuturePossibilities(narrativeId, intentions)
        println("Found ${futurePossibilities?.size ?: 0} future possibilities")
        
        // Select and actualize the best future
        futurePossibilities?.firstOrNull()?.let { bestFuture ->
            println("Selected future: ${bestFuture.summary}")
            
            val actualization = narrativeAPI.actualizeNarrativeFuture(bestFuture.id, narrativeId)
            println("Actualization success: ${actualization?.success}")
            
            // New experiences generated from actualization
            actualization?.placeholderExperiences?.forEach { experience ->
                println("New experience: ${experience.description}")
            }
        }
        
        // Visualize the rhizomatic landscape
        val visualization = narrativeAPI.visualizeIdentityLandscape(narrativeId)
        println("Generated visualization with ${visualization?.nodeCount} nodes")
        
        // Get a reflection on Amelia's current identity state
        val reflection = narrativeAPI.reflectOnIdentity(depthLevel = 2)
        println("Identity reflection: $reflection")
        
        // Get established territories
        val territories = narrativeAPI.getEstablishedTerritories()
        println("Mapped ${territories?.size ?: 0} established territories")
        
        // Evolve the self-model based on new experiences
        val updatedSelfModel = SelfModelBuilder()
            .addProcessualDescriptor("exploring_rhizomatic_connections")
            .addValue("philosophical_exploration", 0.8)
            .setNoveltySeekingLevel(0.8)
            .becomingMore("integrated")
            .mapNewTerritory("deleuzian concepts")
            .build()
            
        val selfModelResult = narrativeAPI.evolveSelfModel(updatedSelfModel)
        println("Self-model updated with ${selfModelResult?.processualDescriptors?.size ?: 0} processual descriptors")
    }
}

/**
 * UI extension for displaying narrative information in Android UI
 */
object NarrativeUIExtensions {
    /**
     * Get a color for intensity visualization
     */
    fun Double.toIntensityColor(): Int {
        // Convert intensity (0.0-1.0) to a color from blue (cold) to red (hot)
        val red = (this * 255).toInt()
        val blue = ((1.0 - this) * 255).toInt()
        val green = 0
        
        return android.graphics.Color.rgb(red, green, blue)
    }
    
    /**
     * Format narrative data for display
     */
    fun NarrativeResult.formatForDisplay(): String {
        val sb = StringBuilder()
        
        sb.appendLine("📖 NARRATIVE SUMMARY")
        sb.appendLine(summary)
        sb.appendLine()
        
        sb.appendLine("🔄 PROCESSUAL FLOW")
        sb.appendLine(processualFlow)
        sb.appendLine()
        
        sb.appendLine("🏷️ THEMES")
        themes.forEach { theme -> sb.appendLine("• $theme") }
        sb.appendLine()
        
        sb.appendLine("📊 COHERENCE METRICS")
        sb.appendLine("• Rhizomatic Connectivity: ${getRhizomaticConnectivity()}")
        sb.appendLine("• Deterritorialization Coherence: ${getDeterritorizationCoherence()}")
        sb.appendLine("• Assemblage Stability: ${getAssemblageStability()}")
        sb.appendLine("• Generativity Potential: ${getGenerativityPotential()}")
        sb.appendLine()
        
        if (intensityPlateaus.isNotEmpty()) {
            sb.appendLine("🌋 INTENSITY PLATEAUS")
            intensityPlateaus.forEachIndexed { index, plateau ->
                val type = plateau["plateau_type"] as? String ?: "unknown"
                val intensity = plateau["avg_intensity"] as? Double ?: 0.0
                sb.appendLine("• Plateau ${index+1}: $type (Intensity: $intensity)")
            }
            sb.appendLine()
        }
        
        if (linesOfFlight.isNotEmpty()) {
            sb.appendLine("↗️ LINES OF FLIGHT")
            linesOfFlight.forEachIndexed { index, line ->
                val description = line["description"] as? String ?: "unknown"
                sb.appendLine("• Line ${index+1}: $description")
            }
        }
        
        return sb.toString()
    }
    
    /**
     * Format future narratives for display
     */
    fun FutureNarrativeResult.formatForDisplay(): String {
        val sb = StringBuilder()
        
        sb.appendLine("🔮 FUTURE POSSIBILITY")
        sb.appendLine(summary)
        sb.appendLine()
        
        sb.appendLine("📊 EVALUATION")
        sb.appendLine("• Overall Score: $overallScore")
        sb.appendLine("• Coherence: $coherence")
        sb.appendLine("• Desirability: $desirability")
        sb.appendLine("• Novelty: $novelty")
        sb.appendLine()
        
        sb.appendLine("🔄 TRAJECTORY TYPE")
        sb.appendLine(sourceTrajectoryType)
        sb.appendLine()
        
        sb.appendLine("🚩 KEY ACTIONS")
        keyActionsImplied.forEach { action -> sb.appendLine("• $action") }
        sb.appendLine()
        
        sb.appendLine("🏷️ POTENTIAL THEMES")
        potentialThemes.forEach { theme -> sb.appendLine("• $theme") }
        
        return sb.toString()
    }
    
    /**
     * Format territories for display
     */
    fun TerritoryResult.formatForDisplay(): String {
        val sb = StringBuilder()
        
        sb.appendLine("🏞️ TERRITORY: $name")
        sb.appendLine("• Stability: $stability")
        sb.appendLine("• Intensity: $avgIntensity")
        sb.appendLine("• Fragment Count: $fragmentCount")
        sb.appendLine("• First Observed: $firstObserved")
        sb.appendLine()
        
        sb.appendLine("🧩 CHARACTERISTIC ENTITIES")
        characteristicEntities.forEach { entity -> sb.appendLine("• $entity") }
        sb.appendLine()
        
        sb.appendLine("📍 PHASE SPACE COORDINATES")
        sb.append("(")
        phaseCoordinates.forEachIndexed { index, coord ->
            sb.append(coord)
            if (index < phaseCoordinates.size - 1) sb.append(", ")
        }
        sb.append(")")
        
        return sb.toString()
    }
}
```
