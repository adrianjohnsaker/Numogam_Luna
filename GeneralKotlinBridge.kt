package com.antonio.my.ai.girlfriend.free.bridge

import android.content.Context
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Python Module Bridge for Amelia AI Girlfriend Android App
 *
 * Provides a robust interface between Kotlin code and Python modules with:
 * - Proper error handling and logging
 * - Asynchronous operations with coroutines
 * - Type-safe responses
 * - Module lifecycle management
 */
class PythonModuleBridge private constructor(context: Context) {

    companion object {
        private const val TAG = "PythonModuleBridge"
        
        @Volatile
        private var INSTANCE: PythonModuleBridge? = null

        fun getInstance(context: Context): PythonModuleBridge {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: PythonModuleBridge(context.applicationContext).also {
                    INSTANCE = it
                    it.initialize()
                }
            }
        }
    }

    private val context: Context = context.applicationContext
    private var pythonInstance: Python? = null
    private var isInitialized = false

    /**
     * Python module configuration data class
     */
    private data class PythonModuleConfig(
        val moduleName: String,
        val initMethod: String = "initialize",
        val requiresAssets: Boolean = false
    )

    /**
     * Supported Python modules
     */
    private val pythonModules = listOf(
        PythonModuleConfig(
            moduleName = "explanation_depth_controller",
            requiresAssets = true
        ),
        PythonModuleConfig(
            moduleName = "module_documentation_repository",
            requiresAssets = true
        ),
        PythonModuleConfig(
            moduleName = "personality_engine"
        ),
        PythonModuleConfig(
            moduleName = "conversation_processor"
        )
    )

    /**
     * Initialize the Python environment and configured modules
     */
    private fun initialize() {
        if (isInitialized) return

        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
                pythonInstance = Python.getInstance()
                Log.i(TAG, "Python environment initialized successfully")
            }

            // Initialize all configured modules
            pythonModules.forEach { config ->
                initializeModule(config)
            }

            isInitialized = true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python environment", e)
            throw PythonBridgeException("Python initialization failed", e)
        }
    }

    /**
     * Initialize a specific Python module with configuration
     */
    private fun initializeModule(config: PythonModuleConfig) {
        try {
            if (config.requiresAssets) {
                copyModuleAssets(config.moduleName)
            }

            val module = pythonInstance?.getModule(config.moduleName)
            module?.callAttr(config.initMethod)
            Log.i(TAG, "${config.moduleName} initialized successfully")
        } catch (e: PyException) {
            Log.e(TAG, "Error initializing ${config.moduleName}", e)
            throw PythonBridgeException("Module ${config.moduleName} initialization failed", e)
        }
    }

    /**
     * Copy necessary assets for Python modules
     */
    private fun copyModuleAssets(moduleName: String) {
        try {
            val assetsDir = File(context.filesDir, "python")
            if (!assetsDir.exists()) {
                assetsDir.mkdirs()
            }

            val moduleAssets = context.assets.list(moduleName) ?: return

            moduleAssets.forEach { asset ->
                val destFile = File(assetsDir, "$moduleName/$asset")
                if (!destFile.exists()) {
                    destFile.parentFile?.mkdirs()
                    context.assets.open("$moduleName/$asset").use { input ->
                        destFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.d(TAG, "Copied asset: $moduleName/$asset")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error copying assets for $moduleName", e)
            throw PythonBridgeException("Asset copy failed for $moduleName", e)
        }
    }

    /**
     * Explanation Service - Handles all explanation-related operations
     */
    object ExplanationService {
        suspend fun getExplanation(
            topic: String,
            depth: Int,
            userId: String? = null,
            context: Map<String, Any>? = null
        ): ExplanationResult {
            return withContext(Dispatchers.IO) {
                try {
                    val module = PythonModuleBridge.getInstance(context).pythonInstance
                        ?.getModule("explanation_depth_controller")
                        ?: throw PythonBridgeException("Python module not initialized")

                    val kwargs = mutableMapOf<String, Any>(
                        "topic" to topic,
                        "depth_level" to depth
                    )

                    userId?.let { kwargs["user_id"] = it }
                    context?.let { kwargs["context"] = JSONObject(it).toString() }

                    val result = module.callAttr("get_explanation", *kwargs.entries
                        .flatMap { listOf(it.key, it.value) }
                        .toTypedArray())

                    val explanation = result?.toString() ?: ""
                    val metadata = if (result?.hasAttr("metadata") == true) {
                        result.getAttr("metadata").toString()
                    } else null

                    ExplanationResult.Success(
                        explanation = explanation,
                        metadata = metadata
                    )
                } catch (e: PyException) {
                    Log.e(TAG, "Error getting explanation", e)
                    ExplanationResult.Error(
                        message = e.message ?: "Explanation generation failed",
                        errorCode = PythonErrorCode.EXPLANATION_ERROR
                    )
                }
            }
        }

        suspend fun updateUserPreference(
            userId: String,
            preference: ExplanationPreference
        ): OperationResult {
            return withContext(Dispatchers.IO) {
                try {
                    val module = PythonModuleBridge.getInstance(context).pythonInstance
                        ?.getModule("explanation_depth_controller")
                        ?: throw PythonBridgeException("Python module not initialized")

                    module.callAttr(
                        "update_user_preference",
                        userId,
                        JSONObject(preference.toMap()).toString()
                    )

                    OperationResult.Success
                } catch (e: PyException) {
                    Log.e(TAG, "Error updating user preference", e)
                    OperationResult.Error(
                        message = e.message ?: "Preference update failed",
                        errorCode = PythonErrorCode.PREFERENCE_UPDATE_ERROR
                    )
                }
            }
        }
    }

    /**
     * Documentation Service - Handles all documentation-related operations
     */
    object DocumentationService {
        suspend fun storeDocumentation(
            moduleName: String,
            content: DocumentationContent,
            overwrite: Boolean = false
        ): OperationResult {
            return withContext(Dispatchers.IO) {
                try {
                    val module = PythonModuleBridge.getInstance(context).pythonInstance
                        ?.getModule("module_documentation_repository")
                        ?: throw PythonBridgeException("Python module not initialized")

                    val result = module.callAttr(
                        "store_documentation",
                        moduleName,
                        JSONObject(content.toMap()).toString(),
                        overwrite
                    )

                    when (result?.toString()?.toBoolean()) {
                        true -> OperationResult.Success
                        false -> OperationResult.Error("Documentation storage failed")
                        null -> OperationResult.Error("Invalid response from Python module")
                    }
                } catch (e: PyException) {
                    Log.e(TAG, "Error storing documentation", e)
                    OperationResult.Error(
                        message = e.message ?: "Documentation storage failed",
                        errorCode = PythonErrorCode.DOCUMENTATION_STORAGE_ERROR
                    )
                }
            }
        }

        suspend fun retrieveDocumentation(
            moduleName: String,
            version: String? = null
        ): DocumentationResult {
            return withContext(Dispatchers.IO) {
                try {
                    val module = PythonModuleBridge.getInstance(context).pythonInstance
                        ?.getModule("module_documentation_repository")
                        ?: throw PythonBridgeException("Python module not initialized")

                    val result = if (version != null) {
                        module.callAttr("retrieve_documentation_version", moduleName, version)
                    } else {
                        module.callAttr("retrieve_documentation", moduleName)
                    }

                    val content = result?.toString() ?: ""
                    DocumentationResult.Success(
                        content = DocumentationContent.fromJson(content),
                        version = if (result?.hasAttr("version") == true) {
                            result.getAttr("version").toString()
                        } else null
                    )
                } catch (e: PyException) {
                    Log.e(TAG, "Error retrieving documentation", e)
                    DocumentationResult.Error(
                        message = e.message ?: "Documentation retrieval failed",
                        errorCode = PythonErrorCode.DOCUMENTATION_RETRIEVAL_ERROR
                    )
                }
            }
        }

        suspend fun getDocumentationHistory(
            moduleName: String,
            limit: Int = 5
        ): DocumentationHistoryResult {
            return withContext(Dispatchers.IO) {
                try {
                    val module = PythonModuleBridge.getInstance(context).pythonInstance
                        ?.getModule("module_documentation_repository")
                        ?: throw PythonBridgeException("Python module not initialized")

                    val result = module.callAttr("get_documentation_history", moduleName, limit)
                    val jsonArray = JSONArray(result.toString())

                    val history = mutableListOf<DocumentationVersionInfo>()
                    for (i in 0 until jsonArray.length()) {
                        val item = jsonArray.getJSONObject(i)
                        history.add(
                            DocumentationVersionInfo(
                                version = item.getString("version"),
                                timestamp = item.getString("timestamp"),
                                author = item.optString("author", null),
                                changes = if (item.has("changes")) {
                                    val changesArray = item.getJSONArray("changes")
                                    List(changesArray.length()) { changesArray.getString(it) }
                                } else emptyList()
                            )
                        )
                    }

                    DocumentationHistoryResult.Success(history)
                } catch (e: PyException) {
                    Log.e(TAG, "Error getting documentation history", e)
                    DocumentationHistoryResult.Error(
                        message = e.message ?: "History retrieval failed",
                        errorCode = PythonErrorCode.HISTORY_RETRIEVAL_ERROR
                    )
                }
            }
        }
    }

    /**
     * Result Types
     */
    sealed class OperationResult {
        object Success : OperationResult()
        data class Error(
            val message: String,
            val errorCode: PythonErrorCode? = null
        ) : OperationResult()
    }

    sealed class ExplanationResult {
        data class Success(
            val explanation: String,
            val metadata: String? = null
        ) : ExplanationResult()

        data class Error(
            val message: String,
            val errorCode: PythonErrorCode
        ) : ExplanationResult()
    }

    sealed class DocumentationResult {
        data class Success(
            val content: DocumentationContent,
            val version: String? = null
        ) : DocumentationResult()

        data class Error(
            val message: String,
            val errorCode: PythonErrorCode
        ) : DocumentationResult()
    }

    sealed class DocumentationHistoryResult {
        data class Success(
            val history: List<DocumentationVersionInfo>
        ) : DocumentationHistoryResult()

        data class Error(
            val message: String,
            val errorCode: PythonErrorCode
        ) : DocumentationHistoryResult()
    }

    /**
     * Data Classes
     */
    data class DocumentationContent(
        val text: String,
        val metadata: Map<String, Any> = emptyMap(),
        val categories: List<String> = emptyList(),
        val technicalConcepts: List<String> = emptyList()
    ) {
        fun toMap(): Map<String, Any> = mapOf(
            "text" to text,
            "metadata" to metadata,
            "categories" to categories,
            "technical_concepts" to technicalConcepts
        )

        companion object {
            fun fromJson(json: String): DocumentationContent {
                val obj = JSONObject(json)
                return DocumentationContent(
                    text = obj.getString("text"),
                    metadata = if (obj.has("metadata")) {
                        obj.getJSONObject("metadata").toMap()
                    } else emptyMap(),
                    categories = if (obj.has("categories")) {
                        obj.getJSONArray("categories").toList()
                    } else emptyList(),
                    technicalConcepts = if (obj.has("technical_concepts")) {
                        obj.getJSONArray("technical_concepts").toList()
                    } else emptyList()
                )
            }
        }
    }

    data class DocumentationVersionInfo(
        val version: String,
        val timestamp: String,
        val author: String?,
        val changes: List<String>
    )

    data class ExplanationPreference(
        val preferredDepth: Int,
        val technicalTerminology: Boolean,
        val examples: Boolean,
        val analogies: Boolean
    ) {
        fun toMap(): Map<String, Any> = mapOf(
            "preferred_depth" to preferredDepth,
            "technical_terminology" to technicalTerminology,
            "examples" to examples,
            "analogies" to analogies
        )
    }

    /**
     * Error Handling
     */
    enum class PythonErrorCode {
        EXPLANATION_ERROR,
        PREFERENCE_UPDATE_ERROR,
        DOCUMENTATION_STORAGE_ERROR,
        DOCUMENTATION_RETRIEVAL_ERROR,
        HISTORY_RETRIEVAL_ERROR,
        MODULE_INITIALIZATION_ERROR
    }

    class PythonBridgeException(message: String, cause: Throwable? = null) :
        Exception(message, cause)
}

/**
 * Extension functions for JSON conversion
 */
private fun JSONObject.toMap(): Map<String, Any> {
    val map = mutableMapOf<String, Any>()
    val keys = this.keys()
    while (keys.hasNext()) {
        val key = keys.next()
        map[key] = this.get(key)
    }
    return map
}

private fun JSONArray.toList(): List<String> {
    val list = mutableListOf<String>()
    for (i in 0 until this.length()) {
        list.add(this.getString(i))
    }
    return list
}

/**
 * Example ViewModel using the bridge
 */
class AmeliaViewModel(private val context: Context) {
    private val pythonBridge by lazy { PythonModuleBridge.getInstance(context) }

    suspend fun getExplanation(
        topic: String,
        depth: Int,
        userId: String? = null
    ): String {
        return when (val result = PythonModuleBridge.ExplanationService.getExplanation(topic, depth, userId)) {
            is PythonModuleBridge.ExplanationResult.Success -> result.explanation
            is PythonModuleBridge.ExplanationResult.Error -> "Error: ${result.message}"
        }
    }

    suspend fun storeDocumentation(
        moduleName: String,
        content: PythonModuleBridge.DocumentationContent
    ): Boolean {
        return when (PythonModuleBridge.DocumentationService.storeDocumentation(moduleName, content)) {
            PythonModuleBridge.OperationResult.Success -> true
            is PythonModuleBridge.OperationResult.Error -> false
        }
    }

    suspend fun getDocumentationHistory(moduleName: String): List<PythonModuleBridge.DocumentationVersionInfo> {
        return when (val result = PythonModuleBridge.DocumentationService.getDocumentationHistory(moduleName)) {
            is PythonModuleBridge.DocumentationHistoryResult.Success -> result.history
            is PythonModuleBridge.DocumentationHistoryResult.Error -> emptyList()
        }
    }
}
