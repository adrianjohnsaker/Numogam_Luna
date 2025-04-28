package com.antonio.my.ai.girlfriend.free

import android.content.Context

/**
 * Enhancer for Amelia's responses using NumogramAI
 */
object AmeliaNumogramEnhancer {
    private var bridge: NumogramBridge? = null
    
    /**
     * Initialize the NumogramBridge
     */
    @JvmStatic
    fun initialize(context: Context) {
        if (bridge == null) {
            bridge = NumogramBridge(context)
        }
    }
    
    /**
     * Enhance Amelia's response using NumogramAI
     * @param context App context
     * @param query User's query
     * @param originalResponse Amelia's original response
     * @return Enhanced response or original if enhancement fails
     */
    @JvmStatic
    fun enhanceResponse(context: Context, query: String, originalResponse: String): String {
        if (bridge == null) {
            initialize(context)
        }
        
        try {
            // Process through NumogramAI
            val numogramResponse = bridge?.processInput(query) ?: return originalResponse
            
            // Get the current zone
            val currentZone = bridge?.getCurrentZone() ?: "unknown"
            
            // Get personality
            val personality = bridge?.getPersonality()
            val dominantTrait = personality?.entries?.maxByOrNull { it.value }?.key ?: "balanced"
            
            // Create an enhanced response that combines Amelia's original response
            // with the numogram-based response
            return when {
                // If original response is very short, just use numogram response
                originalResponse.length <
                20 -> numogramResponse
                
                // Otherwise, combine them
                else -> {
                    "${originalResponse.trim()}\n\n[Zone $currentZone | $dominantTrait]"
                }
            }
        } catch (e: Exception) {
            // If anything goes wrong, return the original response
            return originalResponse
        }
    }
    
    /**
     * Get the current state of the NumogramAI
     */
    @JvmStatic
    fun getState(context: Context): Map<String, Any> {
        if (bridge == null) {
            initialize(context)
        }
        
        return bridge?.getState() ?: mapOf("error" to "Bridge not initialized")
    }
}
