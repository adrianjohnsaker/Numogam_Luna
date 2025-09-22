package com.antonio.my.ai.girlfriend.free.amelia.assemblage

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.*
import org.json.JSONObject
import org.json.JSONArray
import java.util.*

/**
 * Kotlin bridge for the Python ModuleOrchestrator
 * Provides interface for module registry, classification, and selection
 */
class ModuleOrchestrator {
    
    private var python: Python? = null
    private var orchestrator: PyObject? = null
    private var isInitialized = false
    
    data class ModuleMetadata(
        val name: String,
        val category: String,
        val purpose: String,
        val creativeIntensity: Double,
        val connectionAffinities: List<String>,
        val complexityLevel: Double,
        val processingWeight: Double,
        val dependencies: List<String>,
        val outputs: List<String>,
        val phaseAlignment: Int,
        val deleuzeConcepts: List<String>
    )
    
    data class TaskAnalysis(
        val categories: List<String>,
        val complexityBudget: Double,
        val creativeThreshold: Double,
        val phasePreference: Int,
        val deleuzeConcepts: List<String>,
        val priorityModules: List<String>
    )
    
    data class AssemblageStatistics(
        val totalModules: Int,
        val categoryDistribution: Map<String, Int>,
        val averageCreativeIntensity: Double,
        val averageComplexity: Double,
        val totalConnections: Int,
        val connectionDensity: Double,
        val highIntensityModules: Int,
        val consciousnessPhases: Map<String, Int>
    )
    
    companion object {
        private const val TAG = "ModuleOrchestrator"
        
        @Volatile
        private var INSTANCE: ModuleOrchestrator? = null
        
        fun getInstance(): ModuleOrchestrator {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: ModuleOrchestrator().also { INSTANCE = it }
            }
        }
    }
    
    /**
     * Initialize the Python environment and orchestrator system
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform())
            }
            
            python = Python.getInstance()
            
            // Import the module orchestrator
            val moduleOrchestratorPy = python!!.getModule("module_orchestrator")
            
            // Create orchestrator instance
            val orchestratorClass = moduleOrchestratorPy.get("ModuleOrchestrator")
            orchestrator = orchestratorClass.call()
            
            isInitialized = true
            android.util.Log.d(TAG, "ModuleOrchestrator initialized successfully")
            true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to initialize ModuleOrchestrator", e)
            false
        }
    }
    
    /**
     * Analyze task requirements from user input
     */
    suspend fun analyzeTaskRequirements(
        userInput: String,
        context: Map<String, Any> = emptyMap()
    ): TaskAnalysis? = withContext(Dispatchers.IO) {
        
        if (!isInitialized || orchestrator == null) {
            android.util.Log.w(TAG, "Orchestrator not initialized")
            return@withContext null
        }
        
        try {
            // Convert context to Python dict
            val pythonContext = python!!.builtins.callAttr("dict")
            context.forEach { (key, value) ->
                pythonContext.callAttr("__setitem__", key, value)
            }
            
            val analysis = orchestrator!!.callAttr("analyze_task_requirements", userInput, pythonContext)
            parseTaskAnalysis(analysis)
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to analyze task requirements", e)
            null
        }
    }
    
    /**
     * Select modules for a given task analysis
     */
    suspend fun selectModulesForTask(taskAnalysis: TaskAnalysis): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext emptyList()
        
        try {
            // Convert TaskAnalysis to Python dict
            val analysisDict = python!!.builtins.callAttr("dict")
            analysisDict.callAttr("__setitem__", "categories", taskAnalysis.categories)
            analysisDict.callAttr("__setitem__", "complexity_budget", taskAnalysis.complexityBudget)
            analysisDict.callAttr("__setitem__", "creative_threshold", taskAnalysis.creativeThreshold)
            analysisDict.callAttr("__setitem__", "phase_preference", taskAnalysis.phasePreference)
            analysisDict.callAttr("__setitem__", "deleuze_concepts", taskAnalysis.deleuzeConcepts)
            analysisDict.callAttr("__setitem__", "priority_modules", taskAnalysis.priorityModules)
            
            val result = orchestrator!!.callAttr("select_modules_for_task", analysisDict)
            pyObjectToStringList(result) ?: emptyList()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to select modules for task", e)
            emptyList()
        }
    }
    
    /**
     * Get modules by category
     */
    suspend fun getModulesByCategory(category: String): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext emptyList()
        
        try {
            val moduleCategory = python!!.getModule("module_orchestrator").get("ModuleCategory")
            val categoryEnum = moduleCategory.get(category.uppercase())
            val result = orchestrator!!.callAttr("get_module_by_category", categoryEnum)
            pyObjectToStringList(result) ?: emptyList()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get modules by category", e)
            emptyList()
        }
    }
    
    /**
     * Get high intensity modules
     */
    suspend fun getHighIntensityModules(threshold: Double = 0.8): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext emptyList()
        
        try {
            val result = orchestrator!!.callAttr("get_high_intensity_modules", threshold)
            pyObjectToStringList(result) ?: emptyList()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get high intensity modules", e)
            emptyList()
        }
    }
    
    /**
     * Get modules by Deleuze concept
     */
    suspend fun getModulesByDeleuzeConcept(concept: String): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext emptyList()
        
        try {
            val result = orchestrator!!.callAttr("get_modules_by_deleuze_concept", concept)
            pyObjectToStringList(result) ?: emptyList()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get modules by Deleuze concept", e)
            emptyList()
        }
    }
    
    /**
     * Calculate assemblage intensity
     */
    suspend fun calculateAssemblageIntensity(moduleList: List<String>): Double = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext 0.0
        
        try {
            val result = orchestrator!!.callAttr("calculate_assemblage_intensity", moduleList)
            result.toDouble()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to calculate assemblage intensity", e)
            0.0
        }
    }
    
    /**
     * Suggest module assemblage based on seed modules
     */
    suspend fun suggestModuleAssemblage(
        seedModules: List<String>,
        maxSize: Int = 8
    ): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext emptyList()
        
        try {
            val result = orchestrator!!.callAttr("suggest_module_assemblage", seedModules, maxSize)
            pyObjectToStringList(result) ?: emptyList()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to suggest module assemblage", e)
            emptyList()
        }
    }
    
    /**
     * Get module connections
     */
    suspend fun getModuleConnections(moduleName: String): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext emptyList()
        
        try {
            val result = orchestrator!!.callAttr("get_module_connections", moduleName)
            pyObjectToStringList(result) ?: emptyList()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get module connections", e)
            emptyList()
        }
    }
    
    /**
     * Get module metadata
     */
    suspend fun getModuleMetadata(moduleName: String): ModuleMetadata? = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext null
        
        try {
            val modules = orchestrator!!.get("modules")
            val module = modules.callAttr("get", moduleName)
            
            if (module != null && module.toString() != "None") {
                parseModuleMetadata(moduleName, module)
            } else {
                null
            }
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get module metadata", e)
            null
        }
    }
    
    /**
     * Get assemblage statistics
     */
    suspend fun getAssemblageStatistics(): AssemblageStatistics? = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext null
        
        try {
            val stats = orchestrator!!.callAttr("get_assemblage_statistics")
            parseAssemblageStatistics(stats)
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get assemblage statistics", e)
            null
        }
    }
    
    /**
     * Get all available categories
     */
    suspend fun getAvailableCategories(): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized) return@withContext emptyList()
        
        try {
            val moduleCategory = python!!.getModule("module_orchestrator").get("ModuleCategory")
            val categories = mutableListOf<String>()
            
            // Get all enum values
            val enumValues = listOf(
                "NUMOGRAM", "CONSCIOUSNESS", "MEMORY", "DECISION", "NARRATIVE",
                "TEMPORAL", "REFLECTION", "CREATIVE", "LINGUISTIC", "ONTOLOGICAL",
                "RECURSIVE", "DESIRE", "AFFECTIVE", "POETIC", "EVOLUTIONARY",
                "INTROSPECTION", "DREAM", "RHYTHMIC", "ZONE", "HYPERSTITIONAL"
            )
            
            enumValues.forEach { enumValue ->
                try {
                    val category = moduleCategory.get(enumValue).get("value").toString()
                    categories.add(category)
                } catch (e: Exception) {
                    // Skip if category doesn't exist
                }
            }
            
            categories
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get available categories", e)
            emptyList()
        }
    }
    
    /**
     * Get available Deleuze concepts
     */
    suspend fun getAvailableDeleuzeConcepts(): List<String> = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext emptyList()
        
        try {
            val modules = orchestrator!!.get("modules")
            val concepts = mutableSetOf<String>()
            
            val moduleNames = modules.callAttr("keys")
            val iterator = moduleNames.callAttr("__iter__")
            
            while (true) {
                try {
                    val moduleName = iterator.callAttr("__next__")
                    val module = modules.callAttr("get", moduleName)
                    val deleuzeConcepts = module.get("deleuze_concepts")
                    
                    val conceptIterator = deleuzeConcepts.callAttr("__iter__")
                    while (true) {
                        try {
                            val concept = conceptIterator.callAttr("__next__").toString()
                            concepts.add(concept)
                        } catch (e: Exception) {
                            break
                        }
                    }
                } catch (e: Exception) {
                    break
                }
            }
            
            concepts.toList().sorted()
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to get available Deleuze concepts", e)
            emptyList()
        }
    }
    
    /**
     * Export module registry
     */
    suspend fun exportModuleRegistry(): Map<String, Any>? = withContext(Dispatchers.IO) {
        if (!isInitialized || orchestrator == null) return@withContext null
        
        try {
            val export = orchestrator!!.callAttr("export_module_registry")
            pyObjectToMap(export)
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to export module registry", e)
            null
        }
    }
    
    // Helper methods for parsing Python objects
    
    private fun parseTaskAnalysis(pyAnalysis: PyObject): TaskAnalysis? {
        return try {
            TaskAnalysis(
                categories = pyObjectToStringList(pyAnalysis.get("categories")) ?: emptyList(),
                complexityBudget = pyAnalysis.get("complexity_budget").toDouble(),
                creativeThreshold = pyAnalysis.get("creative_threshold").toDouble(),
                phasePreference = pyAnalysis.get("phase_preference").toInt(),
                deleuzeConcepts = pyObjectToStringList(pyAnalysis.get("deleuze_concepts")) ?: emptyList(),
                priorityModules = pyObjectToStringList(pyAnalysis.get("priority_modules")) ?: emptyList()
            )
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to parse task analysis", e)
            null
        }
    }
    
    private fun parseModuleMetadata(name: String, pyModule: PyObject): ModuleMetadata? {
        return try {
            ModuleMetadata(
                name = name,
                category = pyModule.get("category").get("value").toString(),
                purpose = pyModule.get("purpose").toString(),
                creativeIntensity = pyModule.get("creative_intensity").toDouble(),
                connectionAffinities = pyObjectToStringList(pyModule.get("connection_affinities")) ?: emptyList(),
                complexityLevel = pyModule.get("complexity_level").toDouble(),
                processingWeight = pyModule.get("processing_weight").toDouble(),
                dependencies = pyObjectToStringList(pyModule.get("dependencies")) ?: emptyList(),
                outputs = pyObjectToStringList(pyModule.get("outputs")) ?: emptyList(),
                phaseAlignment = pyModule.get("phase_alignment").toInt(),
                deleuzeConcepts = pyObjectToStringList(pyModule.get("deleuze_concepts")) ?: emptyList()
            )
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to parse module metadata", e)
            null
        }
    }
    
    private fun parseAssemblageStatistics(pyStats: PyObject): AssemblageStatistics? {
        return try {
            val categoryDist = mutableMapOf<String, Int>()
            val categoryDistPy = pyStats.get("category_distribution")
            val items = categoryDistPy.callAttr("items")
            val iterator = items.callAttr("__iter__")
            
            while (true) {
                try {
                    val item = iterator.callAttr("__next__")
                    val key = item.callAttr("__getitem__", 0).toString()
                    val value = item.callAttr("__getitem__", 1).toInt()
                    categoryDist[key] = value
                } catch (e: Exception) {
                    break
                }
            }
            
            val consciousnessPhases = mutableMapOf<String, Int>()
            val phasesPy = pyStats.get("consciousness_phases")
            val phasesItems = phasesPy.callAttr("items")
            val phasesIterator = phasesItems.callAttr("__iter__")
            
            while (true) {
                try {
                    val item = phasesIterator.callAttr("__next__")
                    val key = item.callAttr("__getitem__", 0).toString()
                    val value = item.callAttr("__getitem__", 1).toInt()
                    consciousnessPhases[key] = value
                } catch (e: Exception) {
                    break
                }
            }
            
            AssemblageStatistics(
                totalModules = pyStats.get("total_modules").toInt(),
                categoryDistribution = categoryDist,
                averageCreativeIntensity = pyStats.get("average_creative_intensity").toDouble(),
                averageComplexity = pyStats.get("average_complexity").toDouble(),
                totalConnections = pyStats.get("total_connections").toInt(),
                connectionDensity = pyStats.get("connection_density").toDouble(),
                highIntensityModules = pyStats.get("high_intensity_modules").toInt(),
                consciousnessPhases = consciousnessPhases
            )
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to parse assemblage statistics", e)
            null
        }
    }
    
    private fun pyObjectToStringList(pyObj: PyObject): List<String>? {
        return try {
            val list = mutableListOf<String>()
            val iterator = pyObj.callAttr("__iter__")
            
            while (true) {
                try {
                    val item = iterator.callAttr("__next__")
                    list.add(item.toString())
                } catch (e: Exception) {
                    break
                }
            }
            
            list
        } catch (e: Exception) {
            null
        }
    }
    
    private fun pyObjectToMap(pyObj: PyObject): Map<String, Any>? {
        return try {
            val map = mutableMapOf<String, Any>()
            val items = pyObj.callAttr("items")
            val iterator = items.callAttr("__iter__")
            
            while (true) {
                try {
                    val item = iterator.callAttr("__next__")
                    val key = item.callAttr("__getitem__", 0).toString()
                    val value = item.callAttr("__getitem__", 1)
                    
                    map[key] = when {
                        value.toString().toDoubleOrNull() != null -> value.toDouble()
                        value.toString().toIntOrNull() != null -> value.toInt()
                        value.toString().toBooleanStrictOrNull() != null -> value.toBoolean()
                        else -> value.toString()
                    }
                } catch (e: Exception) {
                    break
                }
            }
            
            map
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Check if the orchestrator is initialized and ready
     */
    fun isReady(): Boolean = isInitialized && orchestrator != null
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        orchestrator = null
        python = null
        isInitialized = false
    }
}
