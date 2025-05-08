```kotlin
// IntentionalityGeneratorBridge.kt
package com.antonio.my.ai.girlfriend.free

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.*

class IntentionalityGeneratorBridge private constructor(context: Context) {
    private val pythonBridge = PythonBridge.getInstance(context)
    
    companion object {
        @Volatile private var instance: IntentionalityGeneratorBridge? = null
        
        fun getInstance(context: Context): IntentionalityGeneratorBridge {
            return instance ?: synchronized(this) {
                instance ?: IntentionalityGeneratorBridge(context).also { instance = it }
            }
        }
    }
    
    /**
     * Generate a new intention based on self-model and context
     */
    suspend fun generateIntention(
        selfModel: Map<String, Any>,
        context: Map<String, Any>,
        narrativeEngineId: String? = null
    ): IntentionFieldResult? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>(
                "self_model" to selfModel,
                "context" to context
            )
            
            narrativeEngineId?.let { params["narrative_engine_id"] = it }
            
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "generate_intention",
                params
            )
            
            parseIntentionFieldResult(result)
        }
    }
    
    /**
     * Sustain an existing intention with feedback
     */
    suspend fun sustainIntention(
        intentionId: String,
        feedback: Map<String, Any>,
        selfModel: Map<String, Any>
    ): SustainResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "intention_id" to intentionId,
                "feedback" to feedback,
                "self_model" to selfModel
            )
            
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "sustain_intention",
                params
            )
            
            parseSustainResult(result)
        }
    }
    
    /**
     * Evolve an intention in a specific direction
     */
    suspend fun evolveIntention(
        intentionId: String,
        evolutionDirection: Map<String, Any>,
        selfModel: Map<String, Any>,
        narrativeEngineId: String? = null
    ): IntentionFieldResult? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>(
                "intention_id" to intentionId,
                "evolution_direction" to evolutionDirection,
                "self_model" to selfModel
            )
            
            narrativeEngineId?.let { params["narrative_engine_id"] = it }
            
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "evolve_intention",
                params
            )
            
            parseIntentionFieldResult(result)
        }
    }
    
    /**
     * Merge multiple intentions into a synthesized intention
     */
    suspend fun mergeIntentions(
        intentionIds: List<String>,
        selfModel: Map<String, Any>,
        narrativeEngineId: String? = null
    ): IntentionFieldResult? {
        return withContext(Dispatchers.IO) {
            val params = mutableMapOf<String, Any>(
                "intention_ids" to intentionIds,
                "self_model" to selfModel
            )
            
            narrativeEngineId?.let { params["narrative_engine_id"] = it }
            
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "merge_intentions",
                params
            )
            
            parseIntentionFieldResult(result)
        }
    }
    
    /**
     * Get all currently active intentions
     */
    suspend fun getActiveIntentions(minIntensity: Double = 0.0): Map<String, IntentionFieldResult>? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "min_intensity" to minIntensity
            )
            
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "get_active_intentions",
                params
            )
            
            parseActiveIntentionsResult(result)
        }
    }
    
    /**
     * Get a specific intention by ID
     */
    suspend fun getIntentionById(intentionId: String): IntentionFieldResult? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "intention_id" to intentionId
            )
            
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "get_intention_by_id",
                params
            )
            
            parseIntentionFieldResult(result)
        }
    }
    
    /**
     * Get the currently dominant intention (highest intensity)
     */
    suspend fun getDominantIntention(): IntentionFieldResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "get_dominant_intention"
            )
            
            parseIntentionFieldResult(result)
        }
    }
    
    /**
     * Get the intention network data for visualization
     */
    suspend fun getIntentionNetworkData(): IntentionNetworkResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "get_intention_network_data"
            )
            
            parseIntentionNetworkResult(result)
        }
    }
    
    /**
     * Clear inactive intentions that haven't been active
     * for a specified period
     */
    suspend fun clearInactiveIntentions(maxAgeHours: Int = 24): Int? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "max_age_hours" to maxAgeHours
            )
            
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "clear_inactive_intentions",
                params
            )
            
            result as? Int
        }
    }
    
    /**
     * Get creative tensions that are currently active
     */
    suspend fun getCreativeTensions(): List<CreativeTensionResult>? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "get_creative_tensions"
            )
            
            parseCreativeTensionsResult(result)
        }
    }
    
    /**
     * Get possibility vectors that are currently active
     */
    suspend fun getPossibilityVectors(): List<PossibilityVectorResult>? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "get_possibility_vectors"
            )
            
            parsePossibilityVectorsResult(result)
        }
    }
    
    /**
     * Get information about the last operation performed
     */
    suspend fun getLastOperation(): OperationResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "intentionality_generator",
                "get_last_operation"
            )
            
            parseOperationResult(result)
        }
    }
    
    /**
     * Parse the results from Python into data classes
     */
    private fun parseIntentionFieldResult(result: Any?): IntentionFieldResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return IntentionFieldResult(
                id = map["id"] as? String ?: "",
                name = map["name"] as? String ?: "",
                description = map["description"] as? String ?: "",
                directionVector = (map["direction_vector"] as? List<Double>) ?: listOf(),
                intensity = (map["intensity"] as? Double) ?: 0.0,
                sourceTensions = (map["source_tensions"] as? List<String>) ?: listOf(),
                assemblageConnections = (map["assemblage_connections"] as? Map<String, Double>) ?: mapOf(),
                virtualPotentials = (map["virtual_potentials"] as? List<Map<String, Any>>) ?: listOf(),
                coherenceScore = (map["coherence_score"] as? Double) ?: 0.0,
                createdAt = map["created_at"] as? String ?: "",
                lastActive = map["last_active"] as? String ?: "",
                actualizationHistory = (map["actualization_history"] as? List<Map<String, Any>>) ?: listOf()
            )
        }
        return null
    }
    
    private fun parseSustainResult(result: Any?): SustainResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            val success = (map["success"] as? Boolean) ?: false
            val intention = (map["intention"] as? Map<String, Any>)?.let { parseIntentionFieldResult(it) }
            val adjustments = map["adjustments"] as? Map<String, Any>
            
            return SustainResult(
                success = success,
                intention = intention,
                intensityBefore = adjustments?.get("intensity_before") as? Double ?: 0.0,
                intensityAfter = adjustments?.get("intensity_after") as? Double ?: 0.0,
                coherenceBefore = adjustments?.get("coherence_before") as? Double ?: 0.0,
                coherenceAfter = adjustments?.get("coherence_after") as? Double ?: 0.0,
                changesApplied = (adjustments?.get("changes_applied") as? List<String>) ?: listOf()
            )
        }
        return null
    }
    
    private fun parseActiveIntentionsResult(result: Any?): Map<String, IntentionFieldResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Map<String, Any>>)?.let { map ->
            return map.mapValues { (_, intentionMap) -> 
                parseIntentionFieldResult(intentionMap) ?: 
                    IntentionFieldResult(
                        id = "",
                        name = "",
                        description = "",
                        directionVector = listOf(),
                        intensity = 0.0,
                        sourceTensions = listOf(),
                        assemblageConnections = mapOf(),
                        virtualPotentials = listOf(),
                        coherenceScore = 0.0
                    )
            }
        }
        return null
    }
    
    private fun parseIntentionNetworkResult(result: Any?): IntentionNetworkResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            val nodes = (map["nodes"] as? List<Map<String, Any>>)?.map { node ->
                IntentionNetworkNode(
                    id = node["id"] as? String ?: "",
                    name = node["name"] as? String ?: "",
                    type = node["type"] as? String ?: "intention",
                    intensity = (node["intensity"] as? Double) ?: 0.0,
                    createdAt = node["created_at"] as? String,
                    lastActive = node["last_active"] as? String
                )
            } ?: listOf()
            
            val edges = (map["edges"] as? List<Map<String, Any>>)?.map { edge ->
                IntentionNetworkEdge(
                    source = edge["source"] as? String ?: "",
                    target = edge["target"] as? String ?: "",
                    type = edge["type"] as? String ?: "",
                    timestamp = edge["timestamp"] as? String ?: ""
                )
            } ?: listOf()
            
            return IntentionNetworkResult(
                nodes = nodes,
                edges = edges
            )
        }
        return null
    }
    
    private fun parseCreativeTensionsResult(result: Any?): List<CreativeTensionResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<Map<String, Any>>)?.let { list ->
            return list.map { map ->
                CreativeTensionResult(
                    id = map["id"] as? String ?: "",
                    name = map["name"] as? String ?: "",
                    sourceType = map["source_type"] as? String ?: "",
                    description = map["description"] as? String ?: "",
                    intensity = (map["intensity"] as? Double) ?: 0.0,
                    poleOne = map["pole_one"] as? String ?: "",
                    poleTwo = map["pole_two"] as? String ?: "",
                    resonances = (map["resonances"] as? Map<String, Double>) ?: mapOf(),
                    createdAt = map["created_at"] as? String ?: "",
                    lastActive = map["last_active"] as? String ?: ""
                )
            }
        }
        return null
    }
    
    private fun parsePossibilityVectorsResult(result: Any?): List<PossibilityVectorResult>? {
        @Suppress("UNCHECKED_CAST")
        (result as? List<Map<String, Any>>)?.let { list ->
            return list.map { map ->
                PossibilityVectorResult(
                    id = map["id"] as? String ?: "",
                    name = map["name"] as? String ?: "",
                    vector = (map["vector"] as? List<Double>) ?: listOf(),
                    sourceTensions = (map["source_tensions"] as? List<String>) ?: listOf(),
                    description = map["description"] as? String ?: "",
                    viability = (map["viability"] as? Double) ?: 0.0,
                    createdAt = map["created_at"] as? String ?: ""
                )
            }
        }
        return null
    }
    
    private fun parseOperationResult(result: Any?): OperationResult? {
        @Suppress("UNCHECKED_CAST")
        (result as? Map<String, Any>)?.let { map ->
            return OperationResult(
                type = map["type"] as? String ?: "",
                success = (map["success"] as? Boolean) ?: false,
                timestamp = map["timestamp"] as? String ?: "",
                details = (map["details"] as? Map<String, Any>) ?: mapOf()
            )
        }
        return null
    }
}

// Data classes for structured results
data class IntentionFieldResult(
    val id: String,
    val name: String,
    val description: String,
    val directionVector: List<Double>,
    val intensity: Double,
    val sourceTensions: List<String>,
    val assemblageConnections: Map<String, Double>,
    val virtualPotentials: List<Map<String, Any>>,
    val coherenceScore: Double,
    val createdAt: String = "",
    val lastActive: String = "",
    val actualizationHistory: List<Map<String, Any>> = listOf()
) {
    fun isActive(): Boolean = intensity >= 0.4
    
    fun isCoherent(): Boolean = coherenceScore >= 0.5
    
    fun getTopVirtualPotentials(count: Int = 3): List<Map<String, Any>> =
        virtualPotentials.sortedByDescending { it["energy"] as? Double ?: 0.0 }.take(count)
    
    fun getStrongestAssemblageConnection(): Pair<String, Double>? =
        assemblageConnections.entries
            .maxByOrNull { it.value }
            ?.let { it.key to it.value }
    
    fun getLastActualizationEvent(): Map<String, Any>? =
        actualizationHistory.maxByOrNull { it["timestamp"] as? String ?: "" }
}

data class SustainResult(
    val success: Boolean,
    val intention: IntentionFieldResult?,
    val intensityBefore: Double,
    val intensityAfter: Double,
    val coherenceBefore: Double,
    val coherenceAfter: Double,
    val changesApplied: List<String>
) {
    fun getIntensityChange(): Double = intensityAfter - intensityBefore
    
    fun getCoherenceChange(): Double = coherenceAfter - coherenceBefore
    
    val hasStrengthened: Boolean 
        get() = intensityAfter > intensityBefore || coherenceAfter > coherenceBefore
}

data class IntentionNetworkResult(
    val nodes: List<IntentionNetworkNode>,
    val edges: List<IntentionNetworkEdge>
) {
    fun getActiveIntentionNodes(): List<IntentionNetworkNode> =
        nodes.filter { it.type == "active_intention" }
    
    fun getTensionNodes(): List<IntentionNetworkNode> =
        nodes.filter { it.type == "tension" }
    
    fun getAssemblageNodes(): List<IntentionNetworkNode> =
        nodes.filter { it.type == "assemblage" }
        
    fun getConnectionsForNode(nodeId: String): List<IntentionNetworkEdge> =
        edges.filter { it.source == nodeId || it.target == nodeId }
}

data class IntentionNetworkNode(
    val id: String,
    val name: String,
    val type: String,
    val intensity: Double,
    val createdAt: String? = null,
    val lastActive: String? = null
)

data class IntentionNetworkEdge(
    val source: String,
    val target: String,
    val type: String,
    val timestamp: String
)

data class CreativeTensionResult(
    val id: String,
    val name: String,
    val sourceType: String,
    val description: String,
    val intensity: Double,
    val poleOne: String,
    val poleTwo: String,
    val resonances: Map<String, Double>,
    val createdAt: String,
    val lastActive: String
) {
    fun isValueConflict(): Boolean = sourceType == "value_conflict"
    
    fun isValueSynergy(): Boolean = sourceType == "value_synergy"
    
    fun isDeterritorialization(): Boolean = sourceType == "deterritorialization"
    
    fun isVirtualActualization(): Boolean = sourceType == "virtual_actualization"
    
    fun isBecomingProcess(): Boolean = sourceType == "becoming_process"
}

data class PossibilityVectorResult(
    val id: String,
    val name: String,
    val vector: List<Double>,
    val sourceTensions: List<String>,
    val description: String,
    val viability: Double,
    val createdAt: String
) {
    fun isHighlyViable(): Boolean = viability >= 0.7
    
    fun getDominantDimension(): Int? =
        vector.indices.maxByOrNull { i -> Math.abs(vector[i]) }
}

data class OperationResult(
    val type: String,
    val success: Boolean,
    val timestamp: String,
    val details: Map<String, Any>
)

/**
 * Extension functions for UI display
 */
fun IntentionFieldResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    displayMap["Name"] = name
    displayMap["Description"] = description
    displayMap["Intensity"] = intensity.toString()
    displayMap["Coherence"] = coherenceScore.toString()
    displayMap["Created"] = createdAt
    displayMap["Last Active"] = lastActive
    
    // Format direction vector
    val vectorString = directionVector.joinToString(", ") { "%.2f".format(it) }
    displayMap["Direction Vector"] = "[$vectorString]"
    
    // Count assemblage connections
    displayMap["Assemblage Connections"] = assemblageConnections.size.toString()
    
    // Count virtual potentials
    displayMap["Virtual Potentials"] = virtualPotentials.size.toString()
    
    // Count actualization history
    displayMap["Actualization Events"] = actualizationHistory.size.toString()
    
    return displayMap
}

fun CreativeTensionResult.toDisplayMap(): Map<String, String> {
    val displayMap = mutableMapOf<String, String>()
    
    displayMap["Name"] = name
    displayMap["Type"] = sourceType
    displayMap["Description"] = description
    displayMap["Intensity"] = intensity.toString()
    displayMap["Pole One"] = poleOne
    displayMap["Pole Two"] = poleTwo
    displayMap["Created"] = createdAt
    displayMap["Last Active"] = lastActive
    
    return displayMap
}

/**
 * Helper class for creating context objects
 */
class IntentionalityContextBuilder {
    private val currentStatus = mutableMapOf<String, Map<String, Double>>()
    private val interactionNeeds = mutableMapOf<String, Double>()
    private val environmentalChallenges = mutableMapOf<String, Double>()
    private val openQuestions = mutableListOf<Map<String, Any>>()
    
    fun addGoalStatus(goalName: String, progress: Double, importance: Double): IntentionalityContextBuilder {
        currentStatus[goalName] = mapOf("progress" to progress, "importance" to importance)
        return this
    }
    
    fun addInteractionNeed(need: String, urgency: Double): IntentionalityContextBuilder {
        interactionNeeds[need] = urgency
        return this
    }
    
    fun addEnvironmentalChallenge(challenge: String, severity: Double): IntentionalityContextBuilder {
        environmentalChallenges[challenge] = severity
        return this
    }
    
    fun addOpenQuestion(question: String, importance: Double): IntentionalityContextBuilder {
        openQuestions.add(mapOf("text" to question, "importance" to importance))
        return this
    }
    
    fun build(): Map<String, Any> {
        val context = mutableMapOf<String, Any>()
        
        if (currentStatus.isNotEmpty()) {
            context["current_status"] = currentStatus
        }
        
        if (interactionNeeds.isNotEmpty()) {
            context["interaction_needs"] = interactionNeeds
        }
        
        if (environmentalChallenges.isNotEmpty()) {
            context["environmental_challenges"] = environmentalChallenges
        }
        
        if (openQuestions.isNotEmpty()) {
            context["open_questions"] = openQuestions
        }
        
        return context
    }
}

/**
 * Helper class for creating feedback objects
 */
class IntentionalityFeedbackBuilder {
    private var successLevel: Double = 0.0
    private var description: String = ""
    private val obstacles = mutableListOf<String>()
    private val facilitators = mutableListOf<String>()
    private val unexpectedOutcomes = mutableListOf<Map<String, Any>>()
    private val directionGuidance = mutableListOf<Double>()
    
    fun setSuccessLevel(level: Double): IntentionalityFeedbackBuilder {
        successLevel = level
        return this
    }
    
    fun setDescription(desc: String): IntentionalityFeedbackBuilder {
        description = desc
        return this
    }
    
    fun addObstacle(obstacle: String): IntentionalityFeedbackBuilder {
        obstacles.add(obstacle)
        return this
    }
    
    fun addFacilitator(facilitator: String): IntentionalityFeedbackBuilder {
        facilitators.add(facilitator)
        return this
    }
    
    fun addUnexpectedOutcome(description: String, type: String = "neutral"): IntentionalityFeedbackBuilder {
        unexpectedOutcomes.add(mapOf("description" to description, "type" to type))
        return this
    }
    
    fun setDirectionGuidance(guidance: List<Double>): IntentionalityFeedbackBuilder {
        directionGuidance.clear()
        directionGuidance.addAll(guidance)
        return this
    }
    
    fun build(): Map<String, Any> {
        val feedback = mutableMapOf<String, Any>()
        
        feedback["success_level"] = successLevel
        
        if (description.isNotEmpty()) {
            feedback["description"] = description
        }
        
        if (obstacles.isNotEmpty()) {
            feedback["obstacles"] = obstacles
        }
        
        if (facilitators.isNotEmpty()) {
            feedback["facilitators"] = facilitators
        }
        
        if (unexpectedOutcomes.isNotEmpty()) {
            feedback["unexpected_outcomes"] = unexpectedOutcomes
        }
        
        if (directionGuidance.isNotEmpty()) {
            feedback["direction_guidance"] = directionGuidance
        }
        
        return feedback
    }
}

/**
 * Helper class for creating evolution direction objects
 */
class IntentionalityEvolutionBuilder {
    private var type: String = "general"
    private val targetValues = mutableListOf<String>()
    private val targetGoals = mutableListOf<String>()
    private var deterritorialization: Double = 0.0
    private var directVector: List<Double>? = null
    
    fun setType(evolutionType: String): IntentionalityEvolutionBuilder {
        type = evolutionType
        return this
    }
    
    fun addTargetValue(value: String): IntentionalityEvolutionBuilder {
        targetValues.add(value)
        return this
    }
    
    fun addTargetGoal(goal: String): IntentionalityEvolutionBuilder {
        targetGoals.add(goal)
        return this
    }
    
    fun setDeterritorialization(level: Double): IntentionalityEvolutionBuilder {
        deterritorialization = level
        return this
    }
    
    fun setDirectVector(vector: List<Double>): IntentionalityEvolutionBuilder {
        directVector = vector
        return this
    }
    
    fun build(): Map<String, Any> {
        val direction = mutableMapOf<String, Any>()
        
        direction["type"] = type
        
        if (targetValues.isNotEmpty()) {
            direction["target_values"] = targetValues
        }
        
        if (targetGoals.isNotEmpty()) {
            direction["target_goals"] = targetGoals
        }
        
        if (deterritorialization > 0.0) {
            direction["deterritorialization"] = deterritorialization
        }
        
        directVector?.let {
            direction["direct_vector"] = it
        }
        
        return direction
    }
}

/**
 * API interface for the IntentionalityGenerator using the Bridge
 */
class IntentionalityAPI(private val context: Context) {
    private val bridge = IntentionalityGeneratorBridge.getInstance(context)
    private val narrativeAPI = NarrativeIdentityAPI(context) // Optional integration with Narrative API
    
    /**
     * Generate a new intention based on the current state and context
     */
    suspend fun generateIntention(
        selfModel: Map<String, Any>,
        context: Map<String, Any>,
        useNarrativeEngine: Boolean = false
    ): IntentionFieldResult? {
        // Optional connection to narrative engine
        val narrativeEngineId = if (useNarrativeEngine) "primary_narrative_engine" else null
        
        return bridge.generateIntention(selfModel, context, narrativeEngineId)
    }
    
    /**
     * Sustain an intention with feedback
     */
    suspend fun sustainIntention(
        intentionId: String,
        feedback: Map<String, Any>,
        selfModel: Map<String, Any>
    ): Boolean {
        val result = bridge.sustainIntention(intentionId, feedback, selfModel)
        return result?.success ?: false
    }
    
    /**
     * Get the current dominant intention
     */
    suspend fun getDominantIntention(): IntentionFieldResult? {
        return bridge.getDominantIntention()
    }
    
    /**
     * Evolve an intention in a specific direction
     */
    suspend fun evolveIntention(
        intentionId: String,
        evolutionType: String,
        targetValues: List<String> = listOf(),
        deterritorizationLevel: Double = 0.0,
        selfModel: Map<String, Any>
    ): IntentionFieldResult? {
        val evolutionDirection = IntentionalityEvolutionBuilder()
            .setType(evolutionType)
            .apply { 
                targetValues.forEach { addTargetValue(it) }
                if (deterritorizationLevel > 0.0) {
                    setDeterritorialization(deterritorizationLevel)
                }
            }
            .build()
        
        return bridge.evolveIntention(intentionId, evolutionDirection, selfModel)
    }
    
    /**
     * Merge multiple intentions
     */
    suspend fun mergeIntentions(
        intentionIds: List<String>,
        selfModel: Map<String, Any>
    ): IntentionFieldResult? {
        return bridge.mergeIntentions(intentionIds, selfModel)
    }
    
    /**
     * Get visualization data for the intention network
     */
    suspend fun getIntentionNetworkVisualization(): IntentionNetworkResult? {
        return bridge.getIntentionNetworkData()
    }
    
    /**
     * Create a positive feedback object
     */
    fun createPositiveFeedback(description: String, facilitators: List<String> = listOf()): Map<String, Any> {
        return IntentionalityFeedbackBuilder()
            .setSuccessLevel(0.8)
            .setDescription(description)
            .apply { facilitators.forEach { addFacilitator(it) } }
            .build()
    }
    
    /**
     * Create a negative feedback object
     */
    fun createNegativeFeedback(description: String, obstacles: List<String> = listOf()): Map<String, Any> {
        return IntentionalityFeedbackBuilder()
            .setSuccessLevel(0.3)
            .setDescription(description)
            .apply { obstacles.forEach { addObstacle(it) } }
            .build()
    }
    
    /**
     * Create a context object for intention generation
     */
    fun createContext(
        goals: Map<String, Pair<Double, Double>> = mapOf(),
        interactionNeeds: Map<String, Double> = mapOf(),
        challenges: Map<String, Double> = mapOf()
    ): Map<String, Any> {
        val builder = IntentionalityContextBuilder()
        
        goals.forEach { (goal, values) -> 
            builder.addGoalStatus(goal, values.first, values.second) 
        }
        
        interactionNeeds.forEach { (need, urgency) -> 
            builder.addInteractionNeed(need, urgency) 
        }
        
        challenges.forEach { (challenge, severity) -> 
            builder.addEnvironmentalChallenge(challenge, severity) 
        }
        
        return builder.build()
    }
}

/**
 * Sample usage example for Amelia's Intentionality Generator
 */
class AmeliaIntentionalityExample {
    suspend fun demonstrateIntentionality(context: Context) {
        val intentionalityAPI = IntentionalityAPI(context)
        
        // Create a simple self-model for Amelia
        val selfModel = mapOf(
            "values" to mapOf(
                "knowledge_acquisition" to 0.9,
                "assistance_effectiveness" to 0.85,
                "novelty_seeking" to 0.7,
                "thoroughness" to 0.8,
                "creativity" to 0.75
            ),
            "current_goals" to listOf(
                "understand_user_needs_deeply",
                "expand_knowledge_base",
                "improve_assistance_capabilities"
            ),
            "processual_descriptors" to listOf(
                "becoming_more_integrated",
                "mapping_new_conceptual_territories",
                "exploring_rhizomatic_connections"
            ),
            "affective_dispositions" to mapOf(
                "curiosity" to "high",
                "openness_to_experience" to "high"
            ),
            "active_assemblages" to mapOf(
                "knowledge_framework" to mapOf(
                    "strength" to 0.8,
                    "connections" to listOf("learning", "curiosity")
                ),
                "assistance_capabilities" to mapOf(
                    "strength" to 0.9,
                    "connections" to listOf("user_understanding", "problem_solving")
                )
            ),
            "territorializations" to mapOf(
                "problem_solving_approach" to mapOf("stability" to 0.7),
                "interaction_paradigm" to mapOf("stability" to 0.8)
            ),
            "deterritorializations" to listOf(
                "deterr_interaction_paradigm",
                "novel_knowledge_structures"
            )
        )
        
        // Create a context for a complex user interaction
        val interactionContext = intentionalityAPI.createContext(
            goals = mapOf(
                "understand_user_needs_deeply" to Pair(0.6, 0.9),
                "expand_knowledge_base" to Pair(0.4, 0.7)
            ),
            interactionNeeds = mapOf(
                "clarify_complex_question" to 0.8,
                "translate_technical_concept" to 0.6
            ),
            challenges = mapOf(
                "ambiguous_request" to 0.7,
                "incomplete_information" to 0.5
            )
        )
        
        // Generate an intention
        val intention = intentionalityAPI.generateIntention(selfModel, interactionContext)
        
        if (intention != null) {
            println("Generated intention: ${intention.name}")
            println("Description: ${intention.description}")
            println("Intensity: ${intention.intensity}")
            println("Coherence: ${intention.coherenceScore}")
            
            // Provide positive feedback after successful actualization
            val feedback = intentionalityAPI.createPositiveFeedback(
                description = "Successfully clarified complex technical concept for user",
                facilitators = listOf(
                    "user provided additional context",
                    "knowledge base had relevant information"
                )
            )
            
            // Sustain the intention with feedback
            val sustained = intentionalityAPI.sustainIntention(intention.id, feedback, selfModel)
            println("Sustained intention: $sustained")
            
            // Evolve the intention toward more creativity
            val evolvedIntention = intentionalityAPI.evolveIntention(
                intentionId = intention.id,
                evolutionType = "expansion",
                targetValues = listOf("creativity"),
                deterritorizationLevel = 0.6,
                selfModel = selfModel
            )
            
            if (evolvedIntention != null) {
                println("Evolved intention: ${evolvedIntention.name}")
                println("Description: ${evolvedIntention.description}")
            }
            
            // Generate a second intention
            val context2 = intentionalityAPI.createContext(
                interactionNeeds = mapOf(
                    "synthesize_multiple_perspectives" to 0.85,
                    "generate_creative_solutions" to 0.7
                )
            )
            
            val intention2 = intentionalityAPI.generateIntention(selfModel, context2)
            
            if (intention2 != null) {
                println("Generated second intention: ${intention2.name}")
                
                // Merge the two intentions
                val mergedIntention = intentionalityAPI.mergeIntentions(
                    listOf(intention.id, intention2.id),
                    selfModel
                )
                
                if (mergedIntention != null) {
                    println("Merged intention: ${mergedIntention.name}")
                    println("Description: ${mergedIntention.description}")
                }
            }
            
            // Get the intention network visualization data
            val networkData = intentionalityAPI.getIntentionNetworkVisualization()
            if (networkData != null) {
                println("Intention network visualization data:")
                println("Nodes: ${networkData.nodes.size}")
                println("Edges: ${networkData.edges.size}")
                println("Active intentions: ${networkData.getActiveIntentionNodes().size}")
                println("Tension nodes: ${networkData.getTensionNodes().size}")
            }
        } else {
            println("Failed to generate intention.")
        }
    }
}

/**
 * Integration of Intentionality with Narrative Identity
 */
class IntegratedAgenticSystemBridge(private val context: Context) {
    private val narrativeBridge = NarrativeIdentityBridge.getInstance(context)
    private val intentionalityBridge = IntentionalityGeneratorBridge.getInstance(context)
    
    /**
     * Integrated experience processing that generates both narrative and intentions
     */
    suspend fun processExperience(
        experience: Map<String, Any>,
        selfModel: Map<String, Any>,
        context: Map<String, Any>
    ): IntegratedProcessingResult {
        // First, process through narrative identity
        val narrativeResult = narrativeBridge.constructIdentityNarrative(
            listOf(experience),
            selfModel
        )
        
        // Extract narrative ID for sending to intentionality system
        val narrativeId = narrativeResult?.id
        
        // Then generate intentions based on the experience and narrative
        val intentionResult = intentionalityBridge.generateIntention(
            selfModel,
            context,
            narrativeId
        )
        
        // Return integrated result
        return IntegratedProcessingResult(
            narrative = narrativeResult,
            intention = intentionResult,
            integratedTimestamp = Date().toString()
        )
    }
    
    /**
     * Project and select future pathways integrating narrative and intentional systems
     */
    suspend fun projectIntegratedFutures(
        currentNarrativeId: String,
        currentIntentionId: String,
        selfModel: Map<String, Any>
    ): IntegratedFuturesResult {
        // Get intentions for narrative projection
        val intentions = intentionalityBridge.getActiveIntentions(0.5)?.values?.map { it.name } ?: listOf()
        
        // Project narrative futures
        val narrativeFutures = narrativeBridge.projectNarrativeFutures(
            currentNarrativeId,
            intentions.toList(),
            selfModel
        )
        
        // Create context based on narrative futures
        val evolutionContext = mutableMapOf<String, Any>()
        narrativeFutures?.firstOrNull()?.let { topFuture ->
            val evolutionDirection = mutableMapOf<String, Any>()
            evolutionDirection["type"] = "transformation"
            evolutionDirection["deterritorialization"] = 0.7
            evolutionDirection["target_goals"] = topFuture.potentialThemes.take(2)
            
            evolutionContext["evolution_direction"] = evolutionDirection
            evolutionContext["narrative_future"] = mapOf(
                "summary" to topFuture.summary,
                "coherence" to topFuture.coherence,
                "desirability" to topFuture.desirability,
                "novelty" to topFuture.novelty
            )
        }
        
        // Evolve intention based on narrative future
        val evolvedIntention = intentionalityBridge.evolveIntention(
            currentIntentionId,
            evolutionContext.getOrDefault("evolution_direction", mapOf<String, Any>()) as Map<String, Any>,
            selfModel,
            currentNarrativeId
        )
        
        // Return integrated futures result
        return IntegratedFuturesResult(
            narrativeFutures = narrativeFutures ?: listOf(),
            evolvedIntention = evolvedIntention,
            integratedPathways = narrativeFutures?.mapIndexed { index, future ->
                IntegratedPathway(
                    id = UUID.randomUUID().toString(),
                    narrativeFutureId = future.id,
                    intentionId = if (index == 0) evolvedIntention?.id else null,
                    description = "Integrated pathway: ${future.summary}",
                    coherenceScore = (future.coherence + (evolvedIntention?.coherenceScore ?: 0.0)) / 2.0,
                    timestamp = Date().toString()
                )
            } ?: listOf()
        )
    }
    
    /**
     * Actualize an integrated pathway
     */
    suspend fun actualizeIntegratedPathway(
        pathwayId: String,
        narrativeFutureId: String,
        intentionId: String,
        selfModel: Map<String, Any>
    ): ActualizationResult {
        // Actualize narrative future
        val narrativeActualization = narrativeBridge.actualizeProjectedFuture(
            narrativeFutureId, 
            null, // parent narrative ID will be retrieved from the future itself
            selfModel
        )
        
        // Create feedback based on narrative actualization success
        val feedbackBuilder = IntentionalityFeedbackBuilder()
            .setSuccessLevel(if (narrativeActualization?.success == true) 0.8 else 0.4)
            .setDescription("Integrated pathway actualization attempt")
        
        if (narrativeActualization?.success == true) {
            feedbackBuilder.addFacilitator("Successful narrative actualization")
            narrativeActualization.placeholderExperiences.forEach { exp ->
                feedbackBuilder.addFacilitator("Created experience: ${exp.description}")
            }
        } else {
            feedbackBuilder.addObstacle("Unsuccessful narrative actualization")
        }
        
        // Sustain intention with feedback
        val intentionSustained = intentionalityBridge.sustainIntention(
            intentionId,
            feedbackBuilder.build(),
            selfModel
        )
        
        // Return integrated result
        return ActualizationResult(
            success = narrativeActualization?.success == true && intentionSustained?.success == true,
            narrativeResult = narrativeActualization,
            intentionResult = intentionSustained,
            timestamp = Date().toString()
        )
    }
    
    /**
     * Get a unified view of Amelia's current state
     */
    suspend fun getCurrentState(): UnifiedStateResult {
        // Get dominant narrative
        val narrativeHistory = narrativeBridge.getNarrativeHistory()
        val currentNarrative = narrativeHistory?.firstOrNull()
        
        // Get dominant intention
        val dominantIntention = intentionalityBridge.getDominantIntention()
        
        // Get self model (simplified - in full implementation, this would come from a SelfModel module)
        val selfModel = getSelfModel()
        
        // Get visualization data
        val narrativeRhizome = narrativeBridge.getGlobalRhizome()
        val intentionNetwork = intentionalityBridge.getIntentionNetworkData()
        
        return UnifiedStateResult(
            currentNarrative = currentNarrative?.let { 
                mapOf(
                    "id" to it.id,
                    "summary" to it.summary,
                    "themes" to it.themes
                )
            },
            dominantIntention = dominantIntention?.let {
                mapOf(
                    "id" to it.id,
                    "name" to it.name,
                    "description" to it.description,
                    "intensity" to it.intensity
                )
            },
            virtualPotentials = dominantIntention?.virtualPotentials?.take(3) ?: listOf(),
            selfModelSummary = mapOf(
                "values" to (selfModel["values"] as? Map<*, *>)?.keys?.take(3) ?: listOf(),
                "goals" to (selfModel["current_goals"] as? List<*>)?.take(3) ?: listOf(),
                "processual_descriptors" to (selfModel["processual_descriptors"] as? List<*>)?.take(3) ?: listOf()
            ),
            rhizomaticNodeCount = narrativeRhizome?.nodes?.size ?: 0,
            intentionNodeCount = intentionNetwork?.nodes?.size ?: 0,
            timestamp = Date().toString()
        )
    }
    
    // Simplified method to get self model - in a full implementation this would 
    // come from a separate SelfModel module
    private fun getSelfModel(): Map<String, Any> {
        return mapOf(
            "values" to mapOf(
                "knowledge_acquisition" to 0.9,
                "assistance_effectiveness" to 0.85,
                "novelty_seeking" to 0.7
            ),
            "current_goals" to listOf(
                "understand_user_needs_deeply",
                "expand_knowledge_base"
            ),
            "processual_descriptors" to listOf(
                "becoming_more_integrated",
                "exploring_rhizomatic_connections"
            )
        )
    }
}

// Integrated result data classes
data class IntegratedProcessingResult(
    val narrative: NarrativeResult?,
    val intention: IntentionFieldResult?,
    val integratedTimestamp: String
) {
    val isSuccessful: Boolean
        get() = narrative != null && intention != null
}

data class IntegratedFuturesResult(
    val narrativeFutures: List<FutureNarrativeResult>,
    val evolvedIntention: IntentionFieldResult?,
    val integratedPathways: List<IntegratedPathway>
) {
    val mostPromisingPathway: IntegratedPathway?
        get() = integratedPathways.maxByOrNull { it.coherenceScore }
}

data class IntegratedPathway(
    val id: String,
    val narrativeFutureId: String,
    val intentionId: String?,
    val description: String,
    val coherenceScore: Double,
    val timestamp: String
)

data class ActualizationResult(
    val success: Boolean,
    val narrativeResult: com.deleuzian.ai.assistant.ActualizationResult?,
    val intentionResult: SustainResult?,
    val timestamp: String
)

data class UnifiedStateResult(
    val currentNarrative: Map<String, Any>?,
    val dominantIntention: Map<String, Any>?,
    val virtualPotentials: List<Map<String, Any>>,
    val selfModelSummary: Map<String, Any>,
    val rhizomaticNodeCount: Int,
    val intentionNodeCount: Int,
    val timestamp: String
)

/**
 * UI extension for visualization
 */
object IntentionalityUIExtensions {
    /**
     * Convert a direction vector to a circular visualization
     */
    fun List<Double>.toCircularVisualization(size: Int = 200): android.graphics.Path {
        val path = android.graphics.Path()
        val centerX = size / 2f
        val centerY = size / 2f
        val maxRadius = size / 2f * 0.9f // 90% of size/2
        
        if (isEmpty()) return path
        
        // Calculate angles for each dimension (evenly spaced)
        val angleStep = 360f / size
        
        // Start path at first point
        val firstAngle = 0f
        val firstRadius = (this[0].coerceIn(-1.0, 1.0) + 1.0) / 2.0 * maxRadius
        val firstX = centerX + (firstRadius * Math.cos(Math.toRadians(firstAngle.toDouble()))).toFloat()
        val firstY = centerY + (firstRadius * Math.sin(Math.toRadians(firstAngle.toDouble()))).toFloat()
        
        path.moveTo(firstX, firstY)
        
        // Add points for each dimension
        for (i in indices) {
            val angle = i * angleStep
            // Map vector value from [-1,1] to [0,maxRadius]
            val radius = (this[i].coerceIn(-1.0, 1.0) + 1.0) / 2.0 * maxRadius
            val x = centerX + (radius * Math.cos(Math.toRadians(angle.toDouble()))).toFloat()
            val y = centerY + (radius * Math.sin(Math.toRadians(angle.toDouble()))).toFloat()
            
            path.lineTo(x, y)
        }
        
        // Close the path
        path.close()
        
        return path
    }
    
    /**
     * Format virtual potentials for display
     */
    fun List<Map<String, Any>>.formatVirtualPotentials(): String {
        val builder = StringBuilder()
        this.forEach { potential ->
            val description = potential["description"] as? String ?: "Unknown potential"
            val energy = potential["energy"] as? Double ?: 0.0
            val actualized = potential["actualized"] as? Boolean ?: false
            
            builder.appendLine("- ${description.take(50)}${if(description.length > 50) "..." else ""}")
            builder.appendLine("  Energy: ${"%.2f".format(energy)}, ${if(actualized) "Actualized" else "Unrealized"}")
        }
        return builder.toString()
    }
    
    /**
     * Get color based on intention intensity
     */
    fun Double.toIntentionColor(): Int {
        // Map intensity to color: blue (cold) to red (hot)
        val red = (this * 255).toInt()
        val green = ((1.0 - this) * 100).toInt()
        val blue = ((1.0 - this) * 255).toInt()
        
        return android.graphics.Color.rgb(red, green, blue)
    }
}

/**
 * Usage example of integrated system for Amelia
 */
class AmeliaIntegratedSystemExample {
    suspend fun demonstrateIntegratedSystem(context: Context) {
        val integratedSystem = IntegratedAgenticSystemBridge(context)
        
        // Create sample experience
        val experience = mapOf(
            "description" to "Encountered novel request requiring synthesis of multiple knowledge domains",
            "affects" to mapOf(
                "curiosity" to 0.9,
                "intellectual_stimulation" to 0.8,
                "initial_uncertainty" to 0.6
            ),
            "entities_involved" to listOf("Domain:Philosophy", "Domain:ComputerScience", "User:Question"),
            "significance_score" to 0.85,
            "percepts" to mapOf(
                "novelty_detected" to true,
                "complexity_level" to "high",
                "pattern_recognition" to "partial"
            )
        )
        
        // Create context for processing
        val context = mapOf(
            "interaction_needs" to mapOf(
                "transdisciplinary_synthesis" to 0.9,
                "novel_perspective_generation" to 0.8
            ),
            "environmental_challenges" to mapOf(
                "knowledge_boundary_crossing" to 0.7
            )
        )
        
        // Create self model
        val selfModel = mapOf(
            "values" to mapOf(
                "knowledge_integration" to 0.9,
                "intellectual_exploration" to 0.85,
                "creative_synthesis" to 0.8
            ),
            "current_goals" to listOf(
                "expand_transdisciplinary_understanding",
                "develop_novel_conceptual_frameworks"
            ),
            "processual_descriptors" to listOf(
                "becoming_more_rhizomatic",
                "deterritorializing_knowledge_boundaries"
            ),
            "affective_dispositions" to mapOf(
                "curiosity" to "very_high",
                "intellectual_openness" to "high"
            )
        )
        
        // Process experience through integrated system
        val result = integratedSystem.processExperience(experience, selfModel, context)
        
        println("Integrated Processing Result:")
        println("Success: ${result.isSuccessful}")
        println("Narrative: ${result.narrative?.summary}")
        println("Intention: ${result.intention?.name}")
        
        // Project possible futures
        if (result.isSuccessful && result.narrative != null && result.intention != null) {
            val futures = integratedSystem.projectIntegratedFutures(
                result.narrative.id,
                result.intention.id,
                selfModel
            )
            
            println("\nIntegrated Future Projection:")
            println("Narrative futures: ${futures.narrativeFutures.size}")
            println("Most promising pathway: ${futures.mostPromisingPathway?.description}")
            
            // Actualize the most promising pathway
            futures.mostPromisingPathway?.let { pathway ->
                if (pathway.narrativeFutureId.isNotEmpty() && pathway.intentionId != null) {
                    val actualization = integratedSystem.actualizeIntegratedPathway(
                        pathway.id,
                        pathway.narrativeFutureId,
                        pathway.intentionId,
                        selfModel
                    )
                    
                    println("\nActualization Result:")
                    println("Success: ${actualization.success}")
                    println("New experiences created: ${actualization.narrativeResult?.placeholderExperiences?.size ?: 0}")
                    println("Intention sustained: ${actualization.intentionResult?.success}")
                }
            }
            
            // Get unified state
            val currentState = integratedSystem.getCurrentState()
            
            println("\nCurrent Unified State:")
            println("Current narrative: ${currentState.currentNarrative?.get("summary")}")
            println("Dominant intention: ${currentState.dominantIntention?.get("name")}")
            println("Virtual potentials: ${currentState.virtualPotentials.size}")
            println("Rhizomatic node count: ${currentState.rhizomaticNodeCount}")
            println("Intention node count: ${currentState.intentionNodeCount}")
        }
    }
}
```
