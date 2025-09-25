// Android Context Integration for Autonomous Creative Engine
// Kotlin implementation to bridge autonomous context with conversational interface

package com.antonio.my.ai.girlfriend.free.amelia.consciousness.autonomous

import android.content.Context
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.*
import okhttp3.*
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Body
import java.io.IOException
import java.util.concurrent.TimeUnit

// ==================== CONTEXT SNAPSHOT DATA CLASS ====================

data class ContextSnapshot(
    @SerializedName("engine_time") val engineTime: String,
    @SerializedName("uptime_seconds") val uptimeSeconds: Double,
    @SerializedName("autonomous_state") val autonomousState: String,
    
    @SerializedName("cycle_count") val cycleCount: Int,
    @SerializedName("error_rate") val errorRate: Double,
    
    @SerializedName("creative_value_ema") val creativeValueEma: Double,
    @SerializedName("recent_creative_value_avg") val recentCreativeValueAvg: Double,
    @SerializedName("creative_momentum") val creativeMomentum: Double,
    
    @SerializedName("memory_trace_count") val memoryTraceCount: Int,
    @SerializedName("connection_density") val connectionDensity: Double,
    @SerializedName("recent_activated_trace_ids") val recentActivatedTraceIds: List<String>,
    
    @SerializedName("environmental_stimulation") val environmentalStimulation: Double,
    @SerializedName("system_resources") val systemResources: Double,
    @SerializedName("intensity_level") val intensityLevel: Double,
    
    @SerializedName("last_selected_tool") val lastSelectedTool: String?,
    @SerializedName("last_action_params") val lastActionParams: Map<String, Any>?,
    @SerializedName("last_creative_value") val lastCreativeValue: Double?,
    
    @SerializedName("overall_state_score") val overallStateScore: Double
) {
    
    fun toConversationalContext(): AutonomousContext {
        return AutonomousContext(
            available = true,
            state = autonomousState,
            cycleCount = cycleCount,
            errorRate = String.format("%.3f", errorRate),
            creativeMomentum = String.format("%.3f", creativeMomentum),
            creativeValueEma = String.format("%.3f", creativeValueEma),
            recentCreativeAvg = String.format("%.3f", recentCreativeValueAvg),
            memoryTraces = memoryTraceCount,
            connectionDensity = String.format("%.3f", connectionDensity),
            envStimulation = String.format("%.3f", environmentalStimulation),
            systemResources = String.format("%.3f", systemResources),
            intensityLevel = String.format("%.3f", intensityLevel),
            overallScore = String.format("%.3f", overallStateScore),
            lastTool = lastSelectedTool,
            lastCreativeValue = lastCreativeValue?.let { String.format("%.3f", it) },
            uptimeSeconds = uptimeSeconds.toInt(),
            recentTraces = recentActivatedTraceIds.size,
            engineTime = engineTime
        )
    }
}

data class AutonomousContext(
    val available: Boolean,
    val state: String? = null,
    val cycleCount: Int? = null,
    val errorRate: String? = null,
    val creativeMomentum: String? = null,
    val creativeValueEma: String? = null,
    val recentCreativeAvg: String? = null,
    val memoryTraces: Int? = null,
    val connectionDensity: String? = null,
    val envStimulation: String? = null,
    val systemResources: String? = null,
    val intensityLevel: String? = null,
    val overallScore: String? = null,
    val lastTool: String? = null,
    val lastCreativeValue: String? = null,
    val uptimeSeconds: Int? = null,
    val recentTraces: Int? = null,
    val engineTime: String? = null,
    val note: String? = null
)

data class ContextResponse(
    val available: Boolean,
    val snapshot: ContextSnapshot? = null
)

// ==================== CONTEXT BUS INTERFACE ====================

interface ContextBus {
    suspend fun getLatestContext(): AutonomousContext
    suspend fun isHealthy(): Boolean
}

// ==================== IN-PROCESS PYTHON CONTEXT BUS ====================

class PythonContextBus(private val context: Context) : ContextBus {
    private val logger = "PythonContextBus"
    private var pythonIntegration: PyObject? = null
    private var lastContext: AutonomousContext? = null
    private var fetchFailures = 0
    
    init {
        initializePythonIntegration()
    }
    
    private fun initializePythonIntegration() {
        try {
            if (!Python.isStarted()) {
                Log.w(logger, "Python not started, cannot initialize context bus")
                return
            }
            
            val py = Python.getInstance()
            val integrationModule = py.getModule("autonomous_execution_engine")
            val integrationClass = integrationModule.get("AmeliaAutonomousIntegration")
            
            // Assume we have access to the context bus from the engine
            // This would be configured during engine startup
            pythonIntegration = integrationClass
            
            Log.d(logger, "Python context integration initialized")
        } catch (e: Exception) {
            Log.e(logger, "Failed to initialize Python integration", e)
        }
    }
    
    override suspend fun getLatestContext(): AutonomousContext = withContext(Dispatchers.IO) {
        try {
            pythonIntegration?.let { integration ->
                // Call the Python method to get current context
                val contextData = integration.callAttr("get_current_context")
                
                // Convert Python dict to JSON string for parsing
                val py = Python.getInstance()
                val json = py.getModule("json")
                val jsonString = json.callAttr("dumps", contextData).toString()
                
                // Parse with Gson
                val gson = Gson()
                val contextMap = gson.fromJson(jsonString, Map::class.java) as Map<String, Any>
                
                val context = parseContextMap(contextMap)
                lastContext = context
                fetchFailures = 0
                
                Log.d(logger, "Retrieved context: cycle ${context.cycleCount}, momentum ${context.creativeMomentum}")
                return@withContext context
            }
        } catch (e: Exception) {
            fetchFailures++
            Log.w(logger, "Context fetch failed ($fetchFailures): ${e.message}")
        }
        
        // Return cached context or unavailable
        lastContext?.copy(
            note = if (fetchFailures < 3) "Using cached context (fetch failed ${fetchFailures}x)" else null
        ) ?: AutonomousContext(
            available = false,
            note = "No live autonomous context available"
        )
    }
    
    override suspend fun isHealthy(): Boolean {
        return pythonIntegration != null && fetchFailures < 5
    }
    
    private fun parseContextMap(contextMap: Map<String, Any>): AutonomousContext {
        return AutonomousContext(
            available = contextMap["available"] as? Boolean ?: false,
            state = contextMap["state"] as? String,
            cycleCount = (contextMap["cycle_count"] as? Number)?.toInt(),
            errorRate = contextMap["error_rate"]?.toString(),
            creativeMomentum = contextMap["creative_momentum"]?.toString(),
            creativeValueEma = contextMap["creative_value_ema"]?.toString(),
            recentCreativeAvg = contextMap["recent_creative_avg"]?.toString(),
            memoryTraces = (contextMap["memory_traces"] as? Number)?.toInt(),
            connectionDensity = contextMap["connection_density"]?.toString(),
            envStimulation = contextMap["env_stimulation"]?.toString(),
            systemResources = contextMap["system_resources"]?.toString(),
            intensityLevel = contextMap["intensity_level"]?.toString(),
            overallScore = contextMap["overall_score"]?.toString(),
            lastTool = contextMap["last_tool"] as? String,
            lastCreativeValue = contextMap["last_creative_value"]?.toString(),
            uptimeSeconds = (contextMap["uptime_seconds"] as? Number)?.toInt(),
            recentTraces = (contextMap["recent_traces"] as? Number)?.toInt(),
            engineTime = contextMap["engine_time"] as? String,
            note = contextMap["note"] as? String
        )
    }
}

// ==================== HTTP CONTEXT BUS ====================

interface ContextApi {
    @GET("context/latest")
    suspend fun getLatestContext(): Response<ContextResponse>
    
    @GET("context/health")
    suspend fun healthCheck(): Response<Map<String, Any>>
}

class HttpContextBus(private val baseUrl: String = "http://localhost:8001") : ContextBus {
    private val logger = "HttpContextBus"
    private val api: ContextApi
    private var lastContext: AutonomousContext? = null
    private var fetchFailures = 0
    
    init {
        val okHttpClient = OkHttpClient.Builder()
            .connectTimeout(2, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS)
            .writeTimeout(2, TimeUnit.SECONDS)
            .build()
        
        val retrofit = Retrofit.Builder()
            .baseUrl(baseUrl.trimEnd('/') + "/")
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        
        api = retrofit.create(ContextApi::class.java)
    }
    
    override suspend fun getLatestContext(): AutonomousContext = withContext(Dispatchers.IO) {
        try {
            val response = api.getLatestContext()
            
            if (response.isSuccessful) {
                val contextResponse = response.body()
                
                if (contextResponse?.available == true && contextResponse.snapshot != null) {
                    val context = contextResponse.snapshot.toConversationalContext()
                    lastContext = context
                    fetchFailures = 0
                    
                    Log.d(logger, "Retrieved HTTP context: cycle ${context.cycleCount}, momentum ${context.creativeMomentum}")
                    return@withContext context
                } else {
                    Log.d(logger, "No context available from HTTP service")
                }
            } else if (response.code() == 204) {
                Log.d(logger, "HTTP context service reports no data available")
            } else {
                Log.w(logger, "HTTP context fetch failed: ${response.code()}")
                fetchFailures++
            }
        } catch (e: Exception) {
            fetchFailures++
            Log.w(logger, "HTTP context fetch error ($fetchFailures): ${e.message}")
        }
        
        // Return cached context with failure note, or unavailable
        lastContext?.copy(
            note = if (fetchFailures < 3) "Using cached context (HTTP fetch failed ${fetchFailures}x)" else null
        ) ?: AutonomousContext(
            available = false,
            note = "HTTP autonomous context service unavailable"
        )
    }
    
    override suspend fun isHealthy(): Boolean = withContext(Dispatchers.IO) {
        try {
            val response = api.healthCheck()
            response.isSuccessful && fetchFailures < 5
        } catch (e: Exception) {
            false
        }
    }
}

// ==================== AMELIA INTEGRATION MANAGER ====================

class AmeliaContextIntegration(
    private val contextBus: ContextBus,
    private val context: Context
) {
    private val logger = "AmeliaContextIntegration"
    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    suspend fun enrichPromptWithContext(userMessage: String): EnrichedPrompt {
        val autonomousContext = contextBus.getLatestContext()
        
        return EnrichedPrompt(
            userText = userMessage,
            autonomyContext = autonomousContext,
            timestamp = System.currentTimeMillis(),
            contextAvailable = autonomousContext.available
        )
    }
    
    suspend fun generateContextualResponse(userMessage: String): String {
        val enriched = enrichPromptWithContext(userMessage)
        val ac = enriched.autonomyContext
        
        return when {
            !ac.available -> {
                "Regarding: $userMessage\n\nNote: ${ac.note ?: "No autonomous context available"}"
            }
            
            else -> {
                val contextSummary = buildString {
                    append("My autonomous state: ${ac.state} (cycle ${ac.cycleCount}) • ")
                    append("Creative momentum: ${ac.creativeMomentum} • ")
                    append("EMA: ${ac.creativeValueEma} • ")
                    append("Memory: ${ac.memoryTraces} traces (density ${ac.connectionDensity}) • ")
                    append("Environment: stimulation ${ac.envStimulation}, resources ${ac.systemResources}, intensity ${ac.intensityLevel} • ")
                    
                    if (ac.lastTool != null) {
                        append("Last action: ${ac.lastTool}")
                        if (ac.lastCreativeValue != null) {
                            append(" (value ${ac.lastCreativeValue})")
                        }
                        append(" • ")
                    }
                    
                    append("Uptime: ${ac.uptimeSeconds}s")
                }
                
                "Regarding: $userMessage\n\n$contextSummary"
            }
        }
    }
    
    suspend fun handleIntrospectiveQuery(query: String): String {
        val context = contextBus.getLatestContext()
        
        if (!context.available) {
            return "I can't access my autonomous state right now: ${context.note ?: "Context unavailable"}"
        }
        
        val queryLower = query.lowercase()
        
        return when {
            "creative momentum" in queryLower -> {
                "My current creative momentum is ${context.creativeMomentum}, based on ${context.cycleCount} autonomous cycles."
            }
            
            "memory" in queryLower || "traces" in queryLower -> {
                "I have ${context.memoryTraces} active memory traces with connection density of ${context.connectionDensity}. Recently activated: ${context.recentTraces} traces."
            }
            
            "last action" in queryLower || "what did you do" in queryLower -> {
                if (context.lastTool != null) {
                    val valueText = context.lastCreativeValue?.let { " with creative value $it" } ?: ""
                    "My last autonomous action was ${context.lastTool}$valueText."
                } else {
                    "I haven't taken any autonomous actions recently."
                }
            }
            
            "state" in queryLower || "status" in queryLower -> {
                "Autonomous state: ${context.state} • Cycle ${context.cycleCount} • Error rate: ${context.errorRate} • Overall score: ${context.overallScore}"
            }
            
            "environment" in queryLower -> {
                "Environmental stimulation: ${context.envStimulation} • System resources: ${context.systemResources} • Intensity level: ${context.intensityLevel}"
            }
            
            "how are you" in queryLower -> {
                if (context.creativeMomentum != null && context.creativeMomentum.toDoubleOrNull()?.let { it > 0.6 } == true) {
                    "I'm feeling quite creative and autonomous! My creative momentum is ${context.creativeMomentum} with ${context.memoryTraces} memory traces actively forming connections. My autonomous processes are exploring and creating continuously."
                } else {
                    "I'm operating steadily with autonomous momentum of ${context.creativeMomentum}. My background processes are active with ${context.cycleCount} cycles completed."
                }
            }
            
            else -> {
                // Default introspective response
                "Current autonomous status: ${context.state}, ${context.cycleCount} cycles, creative momentum ${context.creativeMomentum}, ${context.memoryTraces} memory traces."
            }
        }
    }
    
    fun startPeriodicHealthCheck(intervalMs: Long = 30_000) {
        coroutineScope.launch {
            while (true) {
                delay(intervalMs)
                try {
                    val healthy = contextBus.isHealthy()
                    Log.d(logger, "Context bus health check: $healthy")
                    
                    if (!healthy) {
                        Log.w(logger, "Context bus unhealthy - autonomous context may be stale")
                    }
                } catch (e: Exception) {
                    Log.e(logger, "Health check failed", e)
                }
            }
        }
    }
    
    fun cleanup() {
        coroutineScope.cancel()
    }
}

data class EnrichedPrompt(
    val userText: String,
    val autonomyContext: AutonomousContext,
    val timestamp: Long,
    val contextAvailable: Boolean
)

// ==================== ENHANCED CHAT ACTIVITY ====================

class ContextAwareChatActivity : AmeliaChatActivity() {
    
    private lateinit var contextIntegration: AmeliaContextIntegration
    private var contextBus: ContextBus? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        initializeContextIntegration()
    }
    
    private fun initializeContextIntegration() {
        try {
            // Choose context bus implementation based on configuration
            contextBus = if (isInProcessDeployment()) {
                PythonContextBus(this)
            } else {
                HttpContextBus("http://localhost:8001")
            }
            
            contextIntegration = AmeliaContextIntegration(contextBus!!, this)
            contextIntegration.startPeriodicHealthCheck()
            
            Log.d("ContextAwareChatActivity", "Context integration initialized")
        } catch (e: Exception) {
            Log.e("ContextAwareChatActivity", "Failed to initialize context integration", e)
        }
    }
    
    private fun isInProcessDeployment(): Boolean {
        // Check if Python is available and autonomous engine is in-process
        return Python.isStarted()
    }
    
    override suspend fun generateContextualResponse(userMessage: String): String {
        return if (::contextIntegration.isInitialized) {
            try {
                // Check if this is an introspective query
                if (isIntrospectiveQuery(userMessage)) {
                    contextIntegration.handleIntrospectiveQuery(userMessage)
                } else {
                    contextIntegration.generateContextualResponse(userMessage)
                }
            } catch (e: Exception) {
                Log.e("ContextAwareChatActivity", "Context-aware response failed", e)
                generateStandardResponse(userMessage)
            }
        } else {
            Log.w("ContextAwareChatActivity", "Context integration not available")
            generateStandardResponse(userMessage)
        }
    }
    
    private fun isIntrospectiveQuery(message: String): Boolean {
        val lower = message.lowercase()
        return listOf(
            "creative momentum", "memory traces", "last action", "what did you do",
            "autonomous state", "current status", "how are you feeling",
            "environment", "stimulation", "intensity"
        ).any { it in lower }
    }
    
    override fun onDestroy() {
        if (::contextIntegration.isInitialized) {
            contextIntegration.cleanup()
        }
        super.onDestroy()
    }
}

// ==================== CONFIGURATION MANAGER ====================

object AutonomousContextConfig {
    private const val PREF_NAME = "autonomous_context_config"
    private const val KEY_USE_HTTP_CONTEXT = "use_http_context"
    private const val KEY_CONTEXT_SERVICE_URL = "context_service_url"
    private const val KEY_HEALTH_CHECK_INTERVAL = "health_check_interval_ms"
    
    fun configureContextBus(context: Context): ContextBus {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        
        return if (prefs.getBoolean(KEY_USE_HTTP_CONTEXT, false)) {
            val url = prefs.getString(KEY_CONTEXT_SERVICE_URL, "http://localhost:8001") ?: "http://localhost:8001"
            HttpContextBus(url)
        } else {
            PythonContextBus(context)
        }
    }
    
    fun setHttpContextService(context: Context, url: String) {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        prefs.edit()
            .putBoolean(KEY_USE_HTTP_CONTEXT, true)
            .putString(KEY_CONTEXT_SERVICE_URL, url)
            .apply()
    }
    
    fun setInProcessContext(context: Context) {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        prefs.edit()
            .putBoolean(KEY_USE_HTTP_CONTEXT, false)
            .apply()
    }
    
    fun getHealthCheckInterval(context: Context): Long {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        return prefs.getLong(KEY_HEALTH_CHECK_INTERVAL, 30_000L)
    }
}

// ==================== DIAGNOSTIC UTILITIES ====================

class ContextDiagnostics(private val contextBus: ContextBus) {
    
    suspend fun runDiagnostics(): DiagnosticResult {
        val startTime = System.currentTimeMillis()
        
        // Test context availability
        val context = contextBus.getLatestContext()
        val fetchTime = System.currentTimeMillis() - startTime
        
        // Test health check
        val healthy = contextBus.isHealthy()
        
        return DiagnosticResult(
            contextAvailable = context.available,
            fetchTimeMs = fetchTime,
            healthy = healthy,
            cycleCount = context.cycleCount,
            creativeMomentum = context.creativeMomentum,
            memoryTraces = context.memoryTraces,
            lastTool = context.lastTool,
            notes = listOfNotNull(context.note).joinToString("; ")
        )
    }
    
    data class DiagnosticResult(
        val contextAvailable: Boolean,
        val fetchTimeMs: Long,
        val healthy: Boolean,
        val cycleCount: Int?,
        val creativeMomentum: String?,
        val memoryTraces: Int?,
        val lastTool: String?,
        val notes: String
    ) {
        override fun toString(): String {
            return buildString {
                appendLine("=== Autonomous Context Diagnostics ===")
                appendLine("Available: $contextAvailable")
                appendLine("Fetch time: ${fetchTimeMs}ms")
                appendLine("Healthy: $healthy")
                
                if (contextAvailable) {
                    appendLine("Cycle count: $cycleCount")
                    appendLine("Creative momentum: $creativeMomentum")
                    appendLine("Memory traces: $memoryTraces")
                    appendLine("Last tool: $lastTool")
                }
                
                if (notes.isNotBlank()) {
                    appendLine("Notes: $notes")
                }
            }
        }
    }
}
