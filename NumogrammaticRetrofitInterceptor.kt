package com.antonio.my.ai.girlfriend.free.interceptor

import com.antonio.my.ai.girlfriend.free.amelia.ai.processor.AmeliaResponseProcessor
import com.chaquo.python.Python
import kotlinx.coroutines.runBlocking
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody.Companion.toResponseBody
import org.json.JSONObject
import java.io.IOException

/**
 * Intercepts Retrofit API calls to enhance Amelia's responses with Numogrammatic memory
 */
class NumogrammaticRetrofitInterceptor : Interceptor {
    
    companion object {
        private const val TAG = "NumogramInterceptor"
        private var sessionId: String = "session_${System.currentTimeMillis()}"
    }
    
    private val responseProcessor: AmeliaResponseProcessor by lazy {
        // Initialize with a dummy repository for now
        // In production, pass the actual repository instance
        AmeliaResponseProcessor(createDummyRepository())
    }
    
    private var initialized = false
    
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        
        // Only intercept chat completion requests
        if (!request.url.toString().contains("chat/completions")) {
            return chain.proceed(request)
        }
        
        // Initialize if needed
        if (!initialized) {
            runBlocking {
                initialized = responseProcessor.initializeNumogrammatic()
            }
        }
        
        // Get the original request body (user's message)
        val userMessage = extractUserMessage(request)
        
        // Proceed with the original request
        val originalResponse = chain.proceed(request)
        
        // If successful, enhance the response
        return if (originalResponse.isSuccessful) {
            enhanceResponse(originalResponse, userMessage)
        } else {
            originalResponse
        }
    }
    
    private fun extractUserMessage(request: Request): String {
        return try {
            val buffer = okio.Buffer()
            request.body?.writeTo(buffer)
            val requestBody = buffer.readUtf8()
            
            // Parse the JSON to get the user's message
            val json = JSONObject(requestBody)
            val messages = json.optJSONArray("messages")
            
            // Find the last user message
            var userMessage = ""
            if (messages != null) {
                for (i in 0 until messages.length()) {
                    val message = messages.getJSONObject(i)
                    if (message.getString("role") == "user") {
                        userMessage = message.getString("content")
                    }
                }
            }
            
            userMessage
        } catch (e: Exception) {
            ""
        }
    }
    
    private fun enhanceResponse(originalResponse: Response, userMessage: String): Response {
        return try {
            // Read the original response body
            val originalBody = originalResponse.body?.string() ?: ""
            val json = JSONObject(originalBody)
            
            // Extract Amelia's response
            val choices = json.optJSONArray("choices")
            if (choices != null && choices.length() > 0) {
                val choice = choices.getJSONObject(0)
                val message = choice.optJSONObject("message")
                val originalContent = message?.optString("content") ?: ""
                
                // Enhance with numogrammatic memory
                val enhancedResponse = runBlocking {
                    responseProcessor.processResponse(userMessage, originalContent)
                }
                
                // Update the JSON with enhanced response
                message?.put("content", enhancedResponse.enhancedResponse)
                
                // Add metadata as custom headers or in the response
                message?.put("numogram_metadata", JSONObject().apply {
                    put("zone", enhancedResponse.memoryZone)
                    put("circuits", enhancedResponse.circuitsActivated.joinToString(","))
                    put("contagions", enhancedResponse.contagionsDetected.joinToString(","))
                    put("enhanced", true)
                })
            }
            
            // Create new response with enhanced content
            val enhancedBody = json.toString().toResponseBody("application/json".toMediaType())
            
            originalResponse.newBuilder()
                .body(enhancedBody)
                .header("X-Numogram-Enhanced", "true")
                .header("X-Numogram-Zone", enhancedResponse.memoryZone.toString())
                .header("X-Numogram-Session", sessionId)
                .build()
            
        } catch (e: Exception) {
            e.printStackTrace()
            // If enhancement fails, return original response
            originalResponse
        }
    }
    
    // Temporary function - replace with actual repository
    private fun createDummyRepository(): com.amelia.memory.MainActivityRepository {
        // You'll need to provide the actual repository instance here
        // This is just a placeholder
        return com.amelia.memory.MainActivityRepository(
            com.chaquo.python.Python.getPlatform()?.getApplication()!!
        )
    }
    
    fun setSessionId(newSessionId: String) {
        sessionId = newSessionId
    }
}

/**
 * Extension function to add the interceptor to Retrofit
 */
fun OkHttpClient.Builder.addNumogrammaticInterceptor(): OkHttpClient.Builder {
    return this.addInterceptor(NumogrammaticRetrofitInterceptor())
}
