package com.antonio.my.ai.girlfriend.free.amelia.android.models

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import java.util.*

/**
 * Enhanced data models for neuro-symbolic dream analysis
 */

// Additional Enums

enum class NarrativeStyle {
    MYTHIC,
    FRAGMENTED,
    VISIONARY
}

enum class NarrativeLength {
    SHORT,
    MEDIUM,
    LONG
}

enum class PatternType {
    RECURSIVE,
    EMERGENT,
    HIERARCHICAL,
    CYCLIC,
    TRANSFORMATIVE,
    NETWORK,
    TEMPORAL,
    FREQUENCY
}

enum class ConnectionType {
    SEMANTIC,
    ASSOCIATIVE,
    ARCHETYPAL,
    CULTURAL,
    PERSONAL,
    SYNTACTIC,
    TRANSFORMATIONAL,
    TEMPORAL,
    SPATIAL,
    CAUSAL
}

enum class DeterritorializedVector {
    BECOMING_ANIMAL,
    BECOMING_MINERAL,
    BECOMING_PLANT,
    BECOMING_MACHINE,
    BECOMING_COSMIC,
    BECOMING_ANCESTRAL,
    MULTIPLICITY,
    NOMADISM,
    METAMORPHOSIS,
    ASSEMBLAGE
}

// Enhanced Data Classes

@Parcelize
data class SymbolicPattern(
    val id: String = UUID.randomUUID().toString(),
    val patternType: PatternType,
    val elements: List<String>,
    val coherenceScore: Float,
    val emergenceProbability: Float,
    val complexityMeasure: Float,
    val archetypalBasis: List<String> = emptyList(),
    val temporalSignature: List<Float> = emptyList(),
    val culturalResonance: Map<String, Float> = emptyMap(),
    val transformationVectors: List<String> = emptyList(),
    val detectedAt: Long = System.currentTimeMillis()
) : Parcelable

@Parcelize
data class SymbolicConnection(
    val id: String = UUID.randomUUID().toString(),
    val symbol1: String,
    val symbol2: String,
    val strength: Float,
    val connectionType: ConnectionType,
    val semanticSimilarity: Float = 0.0f,
    val archetypalSimilarity: Float = 0.0f,
    val culturalOverlap: List<String> = emptyList(),
    val transformationOverlap: List<String> = emptyList(),
    val createdAt: Long = System.currentTimeMillis()
) : Parcelable

@Parcelize
data class FieldCoherenceMetrics(
    val overallCoherence: Float = 0.0f,
    val symbolCoherence: Float = 0.0f,
    val patternCoherence: Float = 0.0f,
    val networkCoherence: Float = 0.0f,
    val archetypalCoherence: Float = 0.0f,
    val entropyMeasure: Float = 0.0f,
    val complexityMeasure: Float = 0.0f,
    val luminosity: Float = 0.0f,
    val temporalFlux: Float = 0.0f,
    val dimensionalDepth: Float = 0.0f
) : Parcelable

@Parcelize
data class FieldIntensity(
    val coherence: Float = 0.5f,
    val entropy: Float = 0.5f,
    val luminosity: Float = 0.5f,
    val temporalFlux: Float = 0.5f,
    val dimensionalDepth: Float = 0.5f,
    val archetypalResonance: Float = 0.5f
) : Parcelable {
    
    fun calculateFieldVector(): FloatArray {
        return floatArrayOf(
            coherence, entropy, luminosity,
            temporalFlux, dimensionalDepth, archetypalResonance
        )
    }
    
    fun getCoherenceLevel(): CoherenceLevel {
        return when {
            coherence > 0.8f -> CoherenceLevel.CRYSTALLINE
            coherence > 0.4f -> CoherenceLevel.LIMINAL
            else -> CoherenceLevel.CHAOTIC
        }
    }
}

enum class CoherenceLevel {
    CHAOTIC,
    LIMINAL,
    CRYSTALLINE
}

@Parcelize
data class NeuralSymbolicInsights(
    val dominantArchetypes: List<String> = emptyList(),
    val symbolicDensity: Int = 0,
    val patternEmergence: Int = 0,
    val personalResonanceStrength: Float = 0.0f,
    val transformationReadiness: Map<DeterritorializedVector, Float> = emptyMap(),
    val culturalInfluence: Map<String, Float> = emptyMap(),
    val complexityScore: Float = 0.0f,
    val noveltyIndex: Float = 0.0f,
    val archetypalDiversity: Float = 0.0f
) : Parcelable

@Parcelize
data class TransformationAnalysis(
    val activationStrength: Float,
    val triggerMatches: Int,
    val participatingSymbols: List<String>,
    val transformationTypes: List<String>,
    val vectorField: List<Float> = emptyList(),
    val coherenceImpact: Float = 0.0f,
    val entropyGeneration: Float = 0.0f,
    val temporalDistortion: Float = 0.0f,
    val spatialFluidity: Float = 0.0f,
    val consciousnessExpansion: Float = 0.0f
) : Parcelable

@Parcelize
data class TransformationScenario(
    val id: String = UUID.randomUUID().toString(),
    val vector: DeterritorializedVector,
    val activationStrength: Float,
    val participatingSymbols: List<String>,
    val narrative: String,
    val probability: Float = 0.0f,
    val intensity: Float = 0.0f,
    val fieldEffects: Map<String, Float> = emptyMap(),
    val patternIntegration: Map<String, Float> = emptyMap(),
    val emergenceTime: Long = System.currentTimeMillis()
) : Parcelable

@Parcelize
data class SymbolAnalogy(
    val symbol: String,
    val similarity: Float,
    val analogyType: String = "semantic",
    val confidence: Float = 0.0f
) : Parcelable

@Parcelize
data class DreamNarrative(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val style: NarrativeStyle,
    val fieldIntensity: FieldIntensity,
    val transformationScenarios: List<TransformationScenario> = emptyList(),
    val narrativeNodes: Int = 0,
    val symbolicDensity: Float = 0.0f,
    val archetypalThemes: List<String> = emptyList(),
    val deterritorializationVectors: List<String> = emptyList(),
    val coherenceReport: String = "",
    val generatedAt: Long = System.currentTimeMillis(),
    val isError: Boolean = false,
    val errorMessage: String = ""
) : Parcelable {
    
    companion object {
        fun error(message: String): DreamNarrative {
            return DreamNarrative(
                text = "",
                style = NarrativeStyle.MYTHIC,
                fieldIntensity = FieldIntensity(),
                isError = true,
                errorMessage = message
            )
        }
    }
    
    fun isValid(): Boolean = !isError && text.isNotBlank()
    
    fun getComplexityLevel(): String {
        return when {
            symbolicDensity > 0.8f -> "High"
            symbolicDensity > 0.4f -> "Medium"
            else -> "Low"
        }
    }
    
    fun getDominantVector(): DeterritorializedVector? {
        return if (deterritorializationVectors.isNotEmpty()) {
            try {
                DeterritorializedVector.valueOf(deterritorializationVectors.first().uppercase())
            } catch (e: Exception) {
                null
            }
        } else null
    }
}

@Parcelize
data class DreamAnalysisResult(
    val id: String = UUID.randomUUID().toString(),
    val symbols: List<SymbolMapping> = emptyList(),
    val patterns: List<SymbolicPattern> = emptyList(),
    val connections: List<SymbolicConnection> = emptyList(),
    val fieldCoherence: FieldCoherenceMetrics = FieldCoherenceMetrics(),
    val neuralInsights: NeuralSymbolicInsights = NeuralSymbolicInsights(),
    val transformationAnalysis: Map<DeterritorializedVector, TransformationAnalysis> = emptyMap(),
    val transformationScenarios: List<TransformationScenario> = emptyList(),
    val complexityMeasure: Float = 0.0f,
    val culturalContext: List<String> = emptyList(),
    val analysisDepth: AnalysisDepth = AnalysisDepth.MODERATE,
    val processingTime: Long = 0L,
    val timestamp: Long = System.currentTimeMillis(),
    val isError: Boolean = false,
    val errorMessage: String = ""
) : Parcelable {
    
    companion object {
        fun success(
            symbols: List<SymbolMapping>,
            patterns: List<SymbolicPattern>,
            connections: List<SymbolicConnection>,
            fieldCoherence: FieldCoherenceMetrics,
            neuralInsights: NeuralSymbolicInsights,
            complexityMeasure: Float
        ): DreamAnalysisResult {
            return DreamAnalysisResult(
                symbols = symbols,
                patterns = patterns,
                connections = connections,
                fieldCoherence = fieldCoherence,
                neuralInsights = neuralInsights,
                complexityMeasure = complexityMeasure
            )
        }
        
        fun error(message: String): DreamAnalysisResult {
            return DreamAnalysisResult(
                isError = true,
                errorMessage = message
            )
        }
    }
    
    fun isValid(): Boolean = !isError && symbols.isNotEmpty()
    
    fun getAnalysisQuality(): AnalysisQuality {
        return when {
            complexityMeasure > 0.8f && fieldCoherence.overallCoherence > 0.7f -> AnalysisQuality.EXCELLENT
            complexityMeasure > 0.6f && fieldCoherence.overallCoherence > 0.5f -> AnalysisQuality.GOOD
            complexityMeasure > 0.4f && fieldCoherence.overallCoherence > 0.3f -> AnalysisQuality.MODERATE
            else -> AnalysisQuality.BASIC
        }
    }
    
    fun getDominantArchetypes(): List<String> {
        return neuralInsights.dominantArchetypes
    }
    
    fun getActiveTransformationVectors(): List<DeterritorializedVector> {
        return transformationAnalysis.filter { it.value.activationStrength > 0.5f }.keys.toList()
    }
    
    fun getHighCoherencePatterns(): List<SymbolicPattern> {
        return patterns.filter { it.coherenceScore > 0.7f }
    }
    
    fun getStrongConnections(): List<SymbolicConnection> {
        return connections.filter { it.strength > 0.6f }
    }
}

enum class AnalysisQuality {
    BASIC,
    MODERATE,
    GOOD,
    EXCELLENT
}

@Parcelize
data class PatternEvolution(
    val id: String = UUID.randomUUID().toString(),
    val patternId: String,
    val evolutionSteps: List<PatternEvolutionStep>,
    val emergenceHistory: List<Long>,
    val stabilityMeasure: Float = 0.0f,
    val adaptationRate: Float = 0.0f,
    val lastUpdate: Long = System.currentTimeMillis()
) : Parcelable

@Parcelize
data class PatternEvolutionStep(
    val timestamp: Long,
    val coherenceScore: Float,
    val complexityMeasure: Float,
    val elementCount: Int,
    val feedback: Float = 0.0f,
    val contextualFactors: List<String> = emptyList()
) : Parcelable

@Parcelize
data class SymbolNetworkNode(
    val symbol: String,
    val centrality: Float,
    val clusteringCoefficient: Float,
    val connections: List<String>,
    val archetypalStrength: Map<String, Float> = emptyMap(),
    val culturalContext: Map<String, Float> = emptyMap(),
    val temporalSignature: List<Float> = emptyList()
) : Parcelable

@Parcelize
data class SymbolNetworkMetrics(
    val nodes: List<SymbolNetworkNode>,
    val density: Float,
    val averagePathLength: Float,
    val clusteringCoefficient: Float,
    val smallWorldness: Float,
    val communities: List<List<String>>,
    val hubNodes: List<String>,
    val bridgeNodes: List<String>
) : Parcelable

@Parcelize
data class TransformationField(
    val id: String = UUID.randomUUID().toString(),
    val activeVectors: Map<DeterritorializedVector, Float>,
    val fieldCoherence: Float,
    val transformationPotential: Float,
    val spatialConfiguration: List<Float>,
    val temporalDynamics: List<Float>,
    val resonanceFrequencies: Map<String, Float>,
    val interferencePatterns: List<InterferencePattern>,
    val stabilityIndex: Float = 0.0f
) : Parcelable

@Parcelize
data class InterferencePattern(
    val id: String = UUID.randomUUID().toString(),
    val interferingVectors: List<DeterritorializedVector>,
    val interferenceType: InterferenceType,
    val amplitude: Float,
    val frequency: Float,
    val phase: Float,
    val resultantField: List<Float> = emptyList()
) : Parcelable

enum class InterferenceType {
    CONSTRUCTIVE,
    DESTRUCTIVE,
    RESONANT,
    CHAOTIC
}

@Parcelize
data class ConsciousnessMapping(
    val id: String = UUID.randomUUID().toString(),
    val consciousnessLevels: Map<String, Float>,
    val awarenessGradients: List<Float>,
    val attentionFoci: List<String>,
    val memoryActivations: Map<String, Float>,
    val emotionalResonance: Map<EmotionalTone, Float>,
    val cognitiveComplexity: Float,
    val introspectionDepth: Float,
    val lucidityMarkers: List<String> = emptyList()
) : Parcelable

@Parcelize
data class DreamFieldDynamics(
    val id: String = UUID.randomUUID().toString(),
    val fieldIntensity: FieldIntensity,
    val transformationField: TransformationField,
    val consciousnessMapping: ConsciousnessMapping,
    val coherenceMetrics: FieldCoherenceMetrics,
    val narrativeFlows: List<NarrativeFlow>,
    val symbolicDensityMap: Map<String, Float>,
    val temporalDistortions: List<TemporalDistortion>,
    val dimensionalResonance: DimensionalResonance
) : Parcelable

@Parcelize
data class NarrativeFlow(
    val id: String = UUID.randomUUID().toString(),
    val flowDirection: String,
    val velocity: Float,
    val coherence: Float,
    val narrativeElements: List<String>,
    val transformationPoints: List<String>,
    val bifurcationNodes: List<String> = emptyList()
) : Parcelable

@Parcelize
data class TemporalDistortion(
    val id: String = UUID.randomUUID().toString(),
    val distortionType: TemporalDistortionType,
    val magnitude: Float,
    val affectedRegions: List<String>,
    val causativeFactors: List<String>,
    val stabilityIndex: Float = 0.0f
) : Parcelable

enum class TemporalDistortionType {
    COMPRESSION,
    DILATION,
    RECURSION,
    FRAGMENTATION,
    SYNCHRONICITY
}

@Parcelize
data class DimensionalResonance(
    val primaryDimension: String,
    val resonanceFrequency: Float,
    val harmonics: List<Float>,
    val stabilityIndex: Float,
    val multidimensionalLinks: List<String>,
    val phaseCoherence: Float = 0.0f,
    val dimensionalDepth: Int = 3
) : Parcelable

// Analysis Context and Configuration

@Parcelize
data class AnalysisConfiguration(
    val analysisDepth: AnalysisDepth = AnalysisDepth.MODERATE,
    val culturalContexts: List<String> = listOf("western"),
    val interpretationStyle: InterpretationStyle = InterpretationStyle.INTEGRATIVE,
    val enablePatternLearning: Boolean = true,
    val enableTransformationAnalysis: Boolean = true,
    val enableNarrativeGeneration: Boolean = true,
    val transformationSensitivity: Float = 0.5f,
    val coherenceThreshold: Float = 0.4f,
    val patternEmergenceThreshold: Float = 0.3f,
    val connectionStrengthThreshold: Float = 0.5f
) : Parcelable

@Parcelize
data class AnalysisMetadata(
    val analysisId: String = UUID.randomUUID().toString(),
    val sessionId: String,
    val userId: String,
    val dreamText: String,
    val configuration: AnalysisConfiguration,
    val startTime: Long,
    val endTime: Long = 0L,
    val processingSteps: List<ProcessingStep> = emptyList(),
    val qualityMetrics: QualityMetrics = QualityMetrics(),
    val resourceUsage: ResourceUsage = ResourceUsage()
) : Parcelable

@Parcelize
data class ProcessingStep(
    val stepName: String,
    val startTime: Long,
    val endTime: Long,
    val status: StepStatus,
    val resultSize: Int = 0,
    val errorMessage: String = ""
) : Parcelable

enum class StepStatus {
    PENDING,
    RUNNING,
    COMPLETED,
    FAILED,
    SKIPPED
}

@Parcelize
data class QualityMetrics(
    val symbolAccuracy: Float = 0.0f,
    val patternReliability: Float = 0.0f,
    val connectionValidity: Float = 0.0f,
    val narrativeCoherence: Float = 0.0f,
    val overallQuality: Float = 0.0f,
    val userSatisfaction: Float = 0.0f,
    val expertValidation: Float = 0.0f
) : Parcelable

@Parcelize
data class ResourceUsage(
    val memoryUsage: Long = 0L,
    val cpuTime: Long = 0L,
    val pythonMemory: Long = 0L,
    val networkRequests: Int = 0,
    val cacheHits: Int = 0,
    val cacheMisses: Int = 0
) : Parcelable

// Result aggregation and comparison

@Parcelize
data class AnalysisComparison(
    val id: String = UUID.randomUUID().toString(),
    val analysis1: DreamAnalysisResult,
    val analysis2: DreamAnalysisResult,
    val symbolSimilarity: Float,
    val patternSimilarity: Float,
    val narrativeSimilarity: Float,
    val overallSimilarity: Float,
    val keyDifferences: List<String>,
    val sharedElements: List<String>,
    val evolutionIndicators: List<String>
) : Parcelable

@Parcelize
data class AnalysisEvolution(
    val userId: String,
    val timeRange: Pair<Long, Long>,
    val analysisCount: Int,
    val evolutionTrends: Map<String, List<Float>>,
    val emergingPatterns: List<SymbolicPattern>,
    val stabilizingSymbols: List<String>,
    val transformationProgression: Map<DeterritorializedVector, List<Float>>,
    val consciousnessEvolution: List<Float>,
    val complexityProgression: List<Float>
) : Parcelable

// Export and sharing models

@Parcelize
data class AnalysisExport(
    val exportId: String = UUID.randomUUID().toString(),
    val analysis: DreamAnalysisResult,
    val narrative: DreamNarrative?,
    val exportFormat: ExportFormat,
    val includePersonalData: Boolean,
    val includeMetadata: Boolean,
    val compressionLevel: Int = 5,
    val exportTime: Long = System.currentTimeMillis(),
    val fileSize: Long = 0L,
    val checksum: String = ""
) : Parcelable

enum class ExportFormat {
    JSON,
    XML,
    CSV,
    PDF,
    HTML,
    MARKDOWN
}
