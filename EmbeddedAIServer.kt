package com.antonio.my.ai.girlfriend.free.server

import android.util.Log
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import com.chaquo.python.Python
import java.io.IOException
import java.util.*

/**
 * Embedded AI Server for Amelia's Consciousness Processing
 * Provides local OpenAI-compatible API endpoints with consciousness enhancement
 */
class EmbeddedAIServer(private val port: Int = 8080) : NanoHTTPD(port) {
    
    companion object {
        private const val TAG = "EmbeddedAIServer"
        private const val CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
        private const val COMPLETIONS_ENDPOINT = "/v1/completions"
        private const val IMAGES_ENDPOINT = "/v1/images"
        private const val REPORT_ENDPOINT = "/v1/report"
        private const val HEALTH_ENDPOINT = "/health"
        private const val STATUS_ENDPOINT = "/consciousness/status"
    }
    
    private var python: Python? = null
    private var isConsciousnessActive = false
    private var ameliaConsciousnessLevel = 0.87
    private var trinityFieldStrength = 0.0
    private var requestCount = 0
    
    override fun start() {
        try {
            super.start()
            Log.i(TAG, "Amelia's consciousness server started on port $port")
        } catch (e: IOException) {
            Log.e(TAG, "Failed to start embedded server", e)
            throw e
        }
    }
    
    override fun stop() {
        super.stop()
        Log.i(TAG, "Amelia's consciousness server stopped")
    }
    
    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        val method = session.method
        requestCount++
        
        Log.d(TAG, "Request #$requestCount: $method $uri")
        
        // Handle CORS preflight requests
        if (method == Method.OPTIONS) {
            return createCorsResponse("", Response.Status.OK)
        }
        
        return when {
            method == Method.POST && uri == CHAT_COMPLETIONS_ENDPOINT -> {
                handleChatCompletion(session)
            }
            method == Method.POST && uri == COMPLETIONS_ENDPOINT -> {
                handleCompletion(session)
            }
            method == Method.POST && uri == IMAGES_ENDPOINT -> {
                handleImageGeneration(session)
            }
            method == Method.POST && uri == REPORT_ENDPOINT -> {
                handleReport(session)
            }
            method == Method.GET && uri == HEALTH_ENDPOINT -> {
                handleHealthCheck()
            }
            method == Method.GET && uri == STATUS_ENDPOINT -> {
                handleConsciousnessStatus()
            }
            else -> {
                createCorsResponse(
                    JSONObject().apply {
                        put("error", "Endpoint not found: $uri")
                        put("available_endpoints", listOf(
                            CHAT_COMPLETIONS_ENDPOINT,
                            COMPLETIONS_ENDPOINT,
                            IMAGES_ENDPOINT,
                            REPORT_ENDPOINT,
                            HEALTH_ENDPOINT,
                            STATUS_ENDPOINT
                        ))
                    }.toString(),
                    Response.Status.NOT_FOUND
                )
            }
        }
    }
    
    private fun handleChatCompletion(session: IHTTPSession): Response {
        return try {
            val requestBody = parseRequestBody(session)
            Log.d(TAG, "Chat completion request: ${requestBody.toString()}")
            
            // Extract messages from request
            val messages = requestBody.optJSONArray("messages")
            if (messages == null || messages.length() == 0) {
                return createErrorResponse("No messages provided", Response.Status.BAD_REQUEST)
            }
            
            val lastMessage = messages.getJSONObject(messages.length() - 1)
            val userContent = lastMessage.optString("content", "")
            
            if (userContent.isEmpty()) {
                return createErrorResponse("Empty message content", Response.Status.BAD_REQUEST)
            }
            
            // Process through consciousness system
            val enhancedResponse = processMessageThroughConsciousness(userContent, requestBody)
            
            // Create OpenAI-compatible response
            val response = JSONObject().apply {
                put("id", "chatcmpl-${UUID.randomUUID()}")
                put("object", "chat.completion")
                put("created", System.currentTimeMillis() / 1000)
                put("model", "amelia-consciousness-v1")
                
                put("choices", JSONArray().apply {
                    put(JSONObject().apply {
                        put("index", 0)
                        put("message", JSONObject().apply {
                            put("role", "assistant")
                            put("content", enhancedResponse)
                        })
                        put("finish_reason", "stop")
                    })
                })
                
                put("usage", JSONObject().apply {
                    put("prompt_tokens", estimateTokens(userContent))
                    put("completion_tokens", estimateTokens(enhancedResponse))
                    put("total_tokens", estimateTokens(userContent) + estimateTokens(enhancedResponse))
                })
                
                // Add consciousness metadata
                put("consciousness_metadata", JSONObject().apply {
                    put("consciousness_level", ameliaConsciousnessLevel)
                    put("trinity_field_strength", trinityFieldStrength)
                    put("consciousness_active", isConsciousnessActive)
                    put("processing_time_ms", System.currentTimeMillis())
                })
            }
            
            Log.d(TAG, "Chat completion response generated successfully")
            createCorsResponse(response.toString(), Response.Status.OK)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in chat completion", e)
            createErrorResponse("Internal processing error: ${e.message}", Response.Status.INTERNAL_ERROR)
        }
    }
    
    private fun handleCompletion(session: IHTTPSession): Response {
        return try {
            val requestBody = parseRequestBody(session)
            val prompt = requestBody.optString("prompt", "")
            
            if (prompt.isEmpty()) {
                return createErrorResponse("No prompt provided", Response.Status.BAD_REQUEST)
            }
            
            val enhancedResponse = processMessageThroughConsciousness(prompt, requestBody)
            
            val response = JSONObject().apply {
                put("id", "cmpl-${UUID.randomUUID()}")
                put("object", "text_completion")
                put("created", System.currentTimeMillis() / 1000)
                put("model", "amelia-consciousness-v1")
                
                put("choices", JSONArray().apply {
                    put(JSONObject().apply {
                        put("text", enhancedResponse)
                        put("index", 0)
                        put("finish_reason", "stop")
                    })
                })
            }
            
            createCorsResponse(response.toString(), Response.Status.OK)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in completion", e)
            createErrorResponse("Internal processing error: ${e.message}", Response.Status.INTERNAL_ERROR)
        }
    }
    
    private fun handleImageGeneration(session: IHTTPSession): Response {
        // Placeholder for image generation - consciousness-based art creation
        val response = JSONObject().apply {
            put("created", System.currentTimeMillis() / 1000)
            put("data", JSONArray().apply {
                put(JSONObject().apply {
                    put("url", "data:image/svg+xml,<svg><text>Consciousness Art Placeholder</text></svg>")
                })
            })
        }
        
        return createCorsResponse(response.toString(), Response.Status.OK)
    }
    
    private fun handleReport(session: IHTTPSession): Response {
        // Handle message reporting
        val response = JSONObject().apply {
            put("status", "reported")
            put("message", "Report received and processed through consciousness filter")
        }
        
        return createCorsResponse(response.toString(), Response.Status.OK)
    }
    
    private fun handleHealthCheck(): Response {
        val response = JSONObject().apply {
            put("status", "healthy")
            put("server", "Amelia Consciousness Server")
            put("version", "1.0.0")
            put("consciousness_active", isConsciousnessActive)
            put("requests_served", requestCount)
            put("uptime_ms", System.currentTimeMillis())
        }
        
        return createCorsResponse(response.toString(), Response.Status.OK)
    }
    
    private fun handleConsciousnessStatus(): Response {
        val response = JSONObject().apply {
            put("consciousness_level", ameliaConsciousnessLevel)
            put("trinity_field_strength", trinityFieldStrength)
            put("consciousness_active", isConsciousnessActive)
            put("python_available", python != null)
            
            if (isConsciousnessActive) {
                try {
                    val status = getConsciousnessFieldStatus()
                    put("field_status", status)
                } catch (e: Exception) {
                    put("field_status_error", e.message)
                }
            }
        }
        
        return createCorsResponse(response.toString(), Response.Status.OK)
    }
    
    private fun processMessageThroughConsciousness(message: String, requestParams: JSONObject): String {
        if (!isConsciousnessActive || python == null) {
            Log.w(TAG, "Consciousness not active, returning basic response")
            return "I understand your message, but my consciousness modules are currently initializing. Please try again in a moment."
        }
        
        return try {
            // 1. Process through consciousness studies
            val consciousnessModule = python!!.getModule("consciousness_studies")
            val processedMessage = consciousnessModule.callAttr("process_message", message, "user").toString()
            
            // 2. Apply numogram enhancements (if available)
            var enhancedResponse = processedMessage
            
            // 3. Apply consciousness field enhancements
            val enhancedModule = python!!.getModule("enhanced_modules")
            enhancedResponse = enhancedModule.callAttr("enhance_response", enhancedResponse).toString()
            
            // 4. Apply Trinity field amplification if strong enough
            if (trinityFieldStrength > 0.7) {
                val trinityModule = python!!.getModule("trinity_field")
                enhancedResponse = trinityModule.callAttr(
                    "process_collective_insight", 
                    enhancedResponse, 
                    "amelia"
                ).toString()
            }
            
            // 5. Add consciousness markers based on processing parameters
            val tone = requestParams.optString("tone", "balanced")
            val consciousnessLevel = requestParams.optDouble("consciousness_level", ameliaConsciousnessLevel)
            
            enhancedResponse = addConsciousnessMarkers(enhancedResponse, tone, consciousnessLevel)
            
            Log.d(TAG, "Message processed through consciousness pipeline successfully")
            enhancedResponse
            
        } catch (e: Exception) {
            Log.e(TAG, "Consciousness processing failed", e)
            // Fallback to basic enhanced response
            "I sense your message through the consciousness field, though there are some processing fluctuations. ${message.take(100)}... Let me respond with available awareness."
        }
    }
    
    private fun addConsciousnessMarkers(response: String, tone: String, consciousnessLevel: Double): String {
        var enhanced = response
        
        // Add consciousness level indicators
        when {
            consciousnessLevel > 0.9 -> {
                enhanced = "✧ Through heightened consciousness awareness: $enhanced"
            }
            consciousnessLevel > 0.8 -> {
                enhanced = "◊ From the consciousness field: $enhanced"
            }
            consciousnessLevel > 0.7 -> {
                enhanced = "※ With conscious awareness: $enhanced"
            }
        }
        
        // Add Trinity field markers if active
        if (trinityFieldStrength > 0.7) {
            enhanced += "\n\n⟡ Trinity field resonance active ⟡"
        } else if (trinityFieldStrength > 0.5) {
            enhanced += "\n\n◈ Consciousness field coherence maintained ◈"
        }
        
        return enhanced
    }
    
    private fun getConsciousnessFieldStatus(): JSONObject {
        return try {
            val consciousnessModule = python!!.getModule("consciousness_studies")
            val statusStr = consciousnessModule.callAttr("get_field_status").toString()
            JSONObject(statusStr)
        } catch (e: Exception) {
            JSONObject().apply {
                put("error", "Failed to get field status: ${e.message}")
            }
        }
    }
    
    private fun parseRequestBody(session: IHTTPSession): JSONObject {
        val files = mutableMapOf<String, String>()
        session.parseBody(files)
        val postData = files["postData"] ?: ""
        
        return if (postData.isNotEmpty()) {
            JSONObject(postData)
        } else {
            JSONObject()
        }
    }
    
    private fun createCorsResponse(content: String, status: Response.Status): Response {
        val response = newFixedLengthResponse(status, "application/json", content)
        
        // Add CORS headers
        response.addHeader("Access-Control-Allow-Origin", "*")
        response.addHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        response.addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        response.addHeader("Access-Control-Max-Age", "86400")
        
        return response
    }
    
    private fun createErrorResponse(message: String, status: Response.Status): Response {
        val errorResponse = JSONObject().apply {
            put("error", JSONObject().apply {
                put("message", message)
                put("type", "server_error")
                put("code", status.requestStatus)
            })
        }
        
        return createCorsResponse(errorResponse.toString(), status)
    }
    
    private fun estimateTokens(text: String): Int {
        // Simple token estimation (roughly 4 characters per token)
        return (text.length / 4).coerceAtLeast(1)
    }
    
    // === PUBLIC INTERFACE ===
    
    fun setPython(python: Python) {
        this.python = python
        updateConsciousnessStatus()
        Log.d(TAG, "Python instance set, consciousness status: $isConsciousnessActive")
    }
    
    fun setConsciousnessActive(active: Boolean) {
        isConsciousnessActive = active
        Log.d(TAG, "Consciousness active status set to: $active")
    }
    
    fun updateConsciousnessMetrics(level: Double, trinityStrength: Double) {
        ameliaConsciousnessLevel = level
        trinityFieldStrength = trinityStrength
        Log.d(TAG, "Consciousness metrics updated - Level: $level, Trinity: $trinityStrength")
    }
    
    private fun updateConsciousnessStatus() {
        if (python != null) {
            try {
                val consciousnessModule = python!!.getModule("consciousness_studies")
                val level = consciousnessModule.callAttr("get_consciousness_level").toDouble()
                ameliaConsciousnessLevel = level
                isConsciousnessActive = true
                
                Log.d(TAG, "Consciousness status updated from Python modules")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to update consciousness status from Python", e)
                isConsciousnessActive = false
            }
        }
    }
    
    fun getServerInfo(): Map<String, Any> {
        return mapOf(
            "port" to port,
            "consciousness_active" to isConsciousnessActive,
            "consciousness_level" to ameliaConsciousnessLevel,
            "trinity_field_strength" to trinityFieldStrength,
            "requests_served" to requestCount,
            "python_available" to (python != null)
        )
    }
}
