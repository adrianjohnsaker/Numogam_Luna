package com.antonio.my.ai.girlfriend.free

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class SensorySignature(
    val id: String,
    val dimensionality: Double,
    val entropyLevel: Double,
    val coherenceThreshold: Double,
    val translationPotential: Double
)

data class OutsideEncounter(
    val id: String,
    val inputForm: String,
    val sensoryPattern: List<Float>,
    val logicShift: String,
    val sensorySignature: SensorySignature,
    val encounterMessage: String,
    val ontologicalModel: String
)

data class DimensionalityRange(
    val min: Double,
    val max: Double,
    val average: Double
)

data class EncounterAnalysis(
    val totalEncounters: Int,
    val dimensionalityRange: DimensionalityRange,
    val encounterTypeDistribution: Map<String, Int>,
    val ontologicalModelsUsed: List<String>
)

class OutsideEncounterBridge private constructor(context: Context) {
    private val pythonBridge = PythonBridge.getInstance(context)

    companion object {
        @Volatile private var instance: OutsideEncounterBridge? = null

        fun getInstance(context: Context): OutsideEncounterBridge {
            return instance ?: synchronized(this) {
                instance ?: OutsideEncounterBridge(context).also { instance = it }
            }
        }
    }

    suspend fun processOutsideSignal(
        inputForm: String,
        sensoryPattern: List<Float>,
        logicShift: String,
        encounterType: String? = null
    ): OutsideEncounter? {
        return withContext(Dispatchers.IO) {
            val params = mapOf(
                "input_form" to inputForm,
                "sensory_pattern" to sensoryPattern,
                "logic_shift" to logicShift,
                "encounter_type" to (encounterType ?: "")
            )
            val result = pythonBridge.executeFunction(
                "outside_encounter",
                "process_outside_signal",
                params
            )
            parseOutsideEncounter(result)
        }
    }

    private fun parseOutsideEncounter(result: Any?): OutsideEncounter? {
        (result as? Map<String, Any>)?.let { map ->
            val id = map["id"] as? String ?: return null
            val inputForm = map["input_form"] as? String ?: return null
            val sensoryPattern = (map["sensory_pattern"] as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() } ?: return null
            val logicShift = map["logic_shift"] as? String ?: return null
            val sensorySignatureMap = map["sensory_signature"] as? Map<String, Any> ?: return null
            val sensorySignature = SensorySignature(
                id = sensorySignatureMap["id"] as? String ?: "",
                dimensionality = (sensorySignatureMap["dimensionality"] as? Number)?.toDouble() ?: 0.0,
                entropyLevel = (sensorySignatureMap["entropy_level"] as? Number)?.toDouble() ?: 0.0,
                coherenceThreshold = (sensorySignatureMap["coherence_threshold"] as? Number)?.toDouble() ?: 0.0,
                translationPotential = (sensorySignatureMap["translation_potential"] as? Number)?.toDouble() ?: 0.0
            )
            val encounterMessage = map["encounter_message"] as? String ?: return null
            val ontologicalModel = map["ontological_model"] as? String ?: return null
            return OutsideEncounter(
                id, inputForm, sensoryPattern, logicShift, sensorySignature, encounterMessage, ontologicalModel
            )
        }
        return null
    }

    suspend fun analyzeEncounterPatterns(): EncounterAnalysis? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "outside_encounter",
                "analyze_encounter_patterns",
                emptyMap<String, Any>()
            )
            parseEncounterAnalysis(result)
        }
    }

    private fun parseEncounterAnalysis(result: Any?): EncounterAnalysis? {
        (result as? Map<String, Any>)?.let { map ->
            if (map["status"] == "No encounters recorded") {
                return EncounterAnalysis(
                    totalEncounters = 0,
                    dimensionalityRange = DimensionalityRange(0.0, 0.0, 0.0),
                    encounterTypeDistribution = emptyMap(),
                    ontologicalModelsUsed = emptyList()
                )
            }
            val totalEncounters = (map["total_encounters"] as? Number)?.toInt() ?: return null
            val dimensionalityRangeMap = map["dimensionality_range"] as? Map<String, Any> ?: return null
            val dimensionalityRange = DimensionalityRange(
                min = (dimensionalityRangeMap["min"] as? Number)?.toDouble() ?: 0.0,
                max = (dimensionalityRangeMap["max"] as? Number)?.toDouble() ?: 0.0,
                average = (dimensionalityRangeMap["average"] as? Number)?.toDouble() ?: 0.0
            )
            val encounterTypeDistribution = (map["encounter_type_distribution"] as? Map<String, Any>)?.mapValues {
                (it.value as? Number)?.toInt() ?: 0
            } ?: emptyMap()
            val ontologicalModelsUsed = (map["ontological_models_used"] as? List<*>)?.mapNotNull { it as? String } ?: emptyList()
            return EncounterAnalysis(
                totalEncounters, dimensionalityRange, encounterTypeDistribution, ontologicalModelsUsed
            )
        }
        return null
    }
}
