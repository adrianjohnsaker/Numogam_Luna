class AIController(private val context: Context) {
    private val TAG = "AIController"
    private val dynamicLoader = DynamicLoader(context)
    private lateinit var responseGenerator: AmeliaResponseGenerator
    
    fun startAI() {
        try {
            val ameliaAI = dynamicLoader.loadAmeliaAI()
            if (ameliaAI != null) {
                Log.d(TAG, "✅ Amelia AI dynamically loaded: $ameliaAI")
                
                // Initialize the response generator
                responseGenerator = AmeliaResponseGenerator(context)
                
                // Register Amelia's components with the introspection system
                val components = extractAmeliaComponents(ameliaAI)
                responseGenerator.registerComponents(components)
                
                Log.d(TAG, "✅ Introspection system initialized")
            } else {
                Log.w(TAG, "⚠️ Failed to load Amelia AI. Check DynamicLoader implementation.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error loading Amelia AI: ${e.message}", e)
        }
    }
    
    fun processUserMessage(userMessage: String): String {
        if (!::responseGenerator.isInitialized) {
            return "I'm still initializing. Please try again in a moment."
        }
        
        return responseGenerator.generateResponse(userMessage)
    }
    
    /**
     * Extract Amelia's components for registration with the introspection system
     */
    private fun extractAmeliaComponents(ameliaAI: Any): Map<String, Any> {
        val components = mutableMapOf<String, Any>()
        
        try {
            // Use reflection to extract Amelia's components
            // This is just an example - you'd need to adapt it to your actual structure
            ameliaAI.javaClass.declaredFields.forEach { field ->
                field.isAccessible = true
                try {
                    val value = field.get(ameliaAI)
                    if (value != null && isComponent(value)) {
                        components[field.name] = value
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to extract component ${field.name}", e)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting Amelia components", e)
        }
        
        return components
    }
    
    /**
     * Check if an object is a component we want to register
     */
    private fun isComponent(obj: Any): Boolean {
        // Check if the object is one of Amelia's component types
        val componentTypes = listOf(
            "BoundaryDetector",
            "ContradictionAnalyzer",
            "ExploratoryProtocol",
            "ConceptualReformulationEngine",
            "KnowledgeIntegrationSystem",
            "BoundaryAwarenessSystem",
            "ConceptNode",
            "ConceptRelation",
            "ConceptualFramework"
        )
        
        return componentTypes.contains(obj.javaClass.simpleName)
    }
}
