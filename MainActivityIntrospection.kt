package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Enhanced Main Activity for Amelia's System Introspection capabilities
 * Provides comprehensive self-analysis and metacognitive functionality
 */
class MainActivityIntrospection : AppCompatActivity() {
    
    private lateinit var ameliaIntrospectionService: AmeliaIntrospectionService
    private lateinit var systemIntrospectionBridge: SystemIntrospectionBridge
    
    // UI Components
    private lateinit var statusText: TextView
    private lateinit var loader: ProgressBar
    private lateinit var searchInput: TextInputEditText
    
    // Buttons
    private lateinit var queryConceptButton: Button
    private lateinit var findImplementationButton: Button
    private lateinit var explainImplementationButton: Button
    private lateinit var searchConceptsButton: Button
    private lateinit var capabilityAnalysisButton: Button
    private lateinit var runtimeObjectButton: Button
    private lateinit var fullIntrospectionButton: Button
    
    // System initialization state
    private var isInitialized = false
    
    // Performance tracking
    private var operationStartTime: Long = 0
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_system_introspection)
        
        // Initialize views
        initializeViews()
        
        // Set button click listeners
        setupButtonListeners()
        
        // Initialize the System Introspection Module
        initializeIntrospectionSystem()
    }
    
    private fun initializeViews() {
        statusText = findViewById(R.id.status_text)
        loader = findViewById(R.id.loader)
        searchInput = findViewById(R.id.search_input)
        
        queryConceptButton = findViewById(R.id.query_concept_button)
        findImplementationButton = findViewById(R.id.find_implementation_button)
        explainImplementationButton = findViewById(R.id.explain_implementation_button)
        searchConceptsButton = findViewById(R.id.search_concepts_button)
        capabilityAnalysisButton = findViewById(R.id.capability_analysis_button)
        runtimeObjectButton = findViewById(R.id.runtime_object_button)
        fullIntrospectionButton = findViewById(R.id.full_introspection_button)
    }
    
    private fun setupButtonListeners() {
        queryConceptButton.setOnClickListener { performConceptQuery() }
        findImplementationButton.setOnClickListener { performImplementationSearch() }
        explainImplementationButton.setOnClickListener { performImplementationExplanation() }
        searchConceptsButton.setOnClickListener { performConceptSearch() }
        capabilityAnalysisButton.setOnClickListener { performCapabilityAnalysis() }
        runtimeObjectButton.setOnClickListener { performRuntimeObjectAnalysis() }
        fullIntrospectionButton.setOnClickListener { performFullIntrospection() }
    }
    
    private fun initializeIntrospectionSystem() {
        showLoading(true)
        statusText.text = getString(R.string.initializing_introspection)
        
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    // Initialize the introspection services
                    systemIntrospectionBridge = SystemIntrospectionBridge(applicationContext)
                    ameliaIntrospectionService = AmeliaIntrospectionService(applicationContext)
                    
                    // Test the initialization by performing a simple query
                    val testConcept = ameliaIntrospectionService.getConceptDetails("Confidence")
                    
                    // Mark as initialized if we got here without exceptions
                    isInitialized = true
                }
                
                // Update UI on successful initialization
                if (isInitialized) {
                    statusText.text = getString(R.string.status_initialized)
                    enableButtons(true)
                    displaySampleQueries()
                } else {
                    statusText.text = getString(R.string.error_initialize_introspection)
                }
            } catch (e: Exception) {
                statusText.text = "${getString(R.string.error_initialize_introspection)}\n\n" +
                        "Error Details: ${e.message}\n\n" +
                        "Stack Trace: ${e.stackTraceToString().take(500)}..."
                e.printStackTrace()
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performConceptQuery() {
        val query = getSearchQuery() ?: return
        
        showLoading(true)
        statusText.text = getString(R.string.querying_concept)
        startPerformanceTracking()
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val conceptInfo = ameliaIntrospectionService.getConceptDetails(query)
                    val implementation = ameliaIntrospectionService.getConceptImplementation(query)
                    
                    buildConceptQueryResult(conceptInfo, implementation, query)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleError(getString(R.string.error_concept_query), e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performImplementationSearch() {
        val query = getSearchQuery() ?: return
        
        showLoading(true)
        statusText.text = getString(R.string.finding_implementation)
        startPerformanceTracking()
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val implementations = ameliaIntrospectionService.findImplementationsForCapability(query)
                    val searchResults = ameliaIntrospectionService.searchConcepts(query)
                    
                    buildImplementationSearchResult(implementations, searchResults, query)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleError(getString(R.string.error_implementation_search), e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performImplementationExplanation() {
        val query = getSearchQuery() ?: return
        
        showLoading(true)
        statusText.text = getString(R.string.explaining_implementation)
        startPerformanceTracking()
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val explanation = ameliaIntrospectionService.explainImplementation(query)
                    
                    buildImplementationExplanationResult(explanation, query)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleError(getString(R.string.error_implementation_explanation), e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performConceptSearch() {
        val query = getSearchQuery() ?: return
        
        showLoading(true)
        statusText.text = getString(R.string.searching_concepts)
        startPerformanceTracking()
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val searchResults = ameliaIntrospectionService.searchConcepts(query)
                    
                    buildConceptSearchResult(searchResults, query)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleError(getString(R.string.error_concept_search), e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performCapabilityAnalysis() {
        val query = getSearchQuery() ?: return
        
        showLoading(true)
        statusText.text = getString(R.string.analyzing_capability)
        startPerformanceTracking()
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val implementations = ameliaIntrospectionService.findImplementationsForCapability(query)
                    val conceptDetails = try {
                        ameliaIntrospectionService.getConceptDetails(query)
                    } catch (e: Exception) {
                        null
                    }
                    
                    buildCapabilityAnalysisResult(implementations, conceptDetails, query)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleError(getString(R.string.error_capability_analysis), e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performRuntimeObjectAnalysis() {
        val query = getSearchQuery() ?: return
        
        showLoading(true)
        statusText.text = getString(R.string.getting_runtime_object)
        startPerformanceTracking()
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val runtimeObject = systemIntrospectionBridge.getRuntimeObject(query)
                    
                    buildRuntimeObjectResult(runtimeObject, query)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleError(getString(R.string.error_runtime_object), e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performFullIntrospection() {
        val query = getSearchQuery() ?: "comprehensive_analysis"
        
        showLoading(true)
        statusText.text = getString(R.string.performing_full_introspection)
        startPerformanceTracking()
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    // Perform comprehensive introspection
                    val conceptDetails = try {
                        ameliaIntrospectionService.getConceptDetails(query)
                    } catch (e: Exception) { null }
                    
                    val implementations = ameliaIntrospectionService.findImplementationsForCapability(query)
                    
                    val searchResults = ameliaIntrospectionService.searchConcepts(query)
                    
                    val explanation = try {
                        ameliaIntrospectionService.explainImplementation(query)
                    } catch (e: Exception) { null }
                    
                    buildFullIntrospectionResult(conceptDetails, implementations, searchResults, explanation, query)
                }
                
                statusText.text = result
            } catch (e: Exception) {
                handleError(getString(R.string.error_full_introspection), e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    // Result building methods
    
    private fun buildConceptQueryResult(conceptInfo: ConceptInfo, implementation: Map<String, List<String>>, query: String): String {
        val builder = StringBuilder()
        val executionTime = getExecutionTime()
        
        builder.append("${getString(R.string.header_concept_query)}\n")
        builder.append("=" .repeat(50)).append("\n\n")
        builder.append("Query: '$query'\n")
        builder.append("Execution Time: ${executionTime}ms\n")
        builder.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        builder.append("CONCEPT INFORMATION:\n")
        builder.append("-".repeat(30)).append("\n")
        builder.append("Name: ${conceptInfo.name}\n\n")
        
        builder.append("Knowledge Base:\n")
        conceptInfo.knowledge.forEach { (key, value) ->
            builder.append("  • $key: $value\n")
        }
        builder.append("\n")
        
        if (conceptInfo.implementation != null) {
            builder.append("Implementation Details:\n")
            conceptInfo.implementation.forEach { (key, value) ->
                builder.append("  • $key: $value\n")
            }
            builder.append("\n")
        }
        
        if (conceptInfo.codeDetails != null) {
            builder.append("Code Details:\n")
            conceptInfo.codeDetails.forEach { (key, value) ->
                builder.append("  • $key: $value\n")
            }
            builder.append("\n")
        }
        
        builder.append("IMPLEMENTATION MAPPING:\n")
        builder.append("-".repeat(30)).append("\n")
        implementation.forEach { (type, items) ->
            builder.append("$type (${items.size}):\n")
            items.forEach { item ->
                builder.append("  • $item\n")
            }
            builder.append("\n")
        }
        
        return builder.toString()
    }
    
    private fun buildImplementationSearchResult(implementations: Map<String, List<String>>, searchResults: List<Map<String, Any>>, query: String): String {
        val builder = StringBuilder()
        val executionTime = getExecutionTime()
        
        builder.append("${getString(R.string.header_implementation_search)}\n")
        builder.append("=" .repeat(50)).append("\n\n")
        builder.append("Query: '$query'\n")
        builder.append("Execution Time: ${executionTime}ms\n")
        builder.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        builder.append("DIRECT IMPLEMENTATIONS:\n")
        builder.append("-".repeat(30)).append("\n")
        if (implementations.isEmpty()) {
            builder.append("No direct implementations found.\n\n")
        } else {
            implementations.forEach { (type, items) ->
                builder.append("$type (${items.size}):\n")
                items.take(10).forEach { item -> // Limit to first 10 items
                    builder.append("  • $item\n")
                }
                if (items.size > 10) {
                    builder.append("  ... and ${items.size - 10} more\n")
                }
                builder.append("\n")
            }
        }
        
        builder.append("CONCEPT-BASED SEARCH RESULTS:\n")
        builder.append("-".repeat(30)).append("\n")
        if (searchResults.isEmpty()) {
            builder.append("No concept-based results found.\n")
        } else {
            searchResults.take(5).forEach { result -> // Limit to first 5 results
                val concept = result["concept"] as? String ?: "Unknown"
                val impl = result["implementation"] as? Map<*, *>
                
                builder.append("Concept: $concept\n")
                if (impl != null) {
                    impl.forEach { (key, value) ->
                        builder.append("  $key: $value\n")
                    }
                }
                builder.append("\n")
            }
            if (searchResults.size > 5) {
                builder.append("... and ${searchResults.size - 5} more results\n")
            }
        }
        
        return builder.toString()
    }
    
    private fun buildImplementationExplanationResult(explanation: ExplanationInfo, query: String): String {
        val builder = StringBuilder()
        val executionTime = getExecutionTime()
        
        builder.append("${getString(R.string.header_implementation_explanation)}\n")
        builder.append("=" .repeat(50)).append("\n\n")
        builder.append("Query: '$query'\n")
        builder.append("Execution Time: ${executionTime}ms\n")
        builder.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        builder.append("IMPLEMENTATION DETAILS:\n")
        builder.append("-".repeat(30)).append("\n")
        builder.append("Name: ${explanation.name}\n")
        builder.append("Type: ${explanation.type}\n\n")
        
        if (explanation.concepts.isNotEmpty()) {
            builder.append("Related Concepts (${explanation.concepts.size}):\n")
            explanation.concepts.forEach { concept ->
                builder.append("  • $concept\n")
            }
            builder.append("\n")
        }
        
        builder.append("EXPLANATION:\n")
        builder.append("-".repeat(30)).append("\n")
        builder.append("${explanation.explanation}\n\n")
        
        if (explanation.code != null) {
            builder.append("CODE DETAILS:\n")
            builder.append("-".repeat(30)).append("\n")
            explanation.code.forEach { (key, value) ->
                builder.append("$key:\n")
                builder.append("  $value\n\n")
            }
        }
        
        return builder.toString()
    }
    
    private fun buildConceptSearchResult(searchResults: List<Map<String, Any>>, query: String): String {
        val builder = StringBuilder()
        val executionTime = getExecutionTime()
        
        builder.append("${getString(R.string.header_concept_search)}\n")
        builder.append("=" .repeat(50)).append("\n\n")
        builder.append("Query: '$query'\n")
        builder.append("Results Found: ${searchResults.size}\n")
        builder.append("Execution Time: ${executionTime}ms\n")
        builder.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (searchResults.isEmpty()) {
            builder.append("No concepts found matching the query.\n\n")
            builder.append("${getString(R.string.sample_concept_queries)}")
        } else {
            builder.append("SEARCH RESULTS:\n")
            builder.append("-".repeat(30)).append("\n")

                      searchResults.forEachIndexed { index, result ->
                val concept = result["concept"] as? String ?: "Unknown"
                val impl = result["implementation"] as? Map<*, *>
                
                builder.append("${index + 1}. Concept: $concept\n")
                if (impl != null) {
                    impl.forEach { (key, value) ->
                        when (value) {
                            is List<*> -> {
                                builder.append("   $key (${value.size}): ")
                                builder.append(value.take(3).joinToString(", "))
                                if (value.size > 3) builder.append(", ...")
                                builder.append("\n")
                            }
                            else -> builder.append("   $key: $value\n")
                        }
                    }
                }
                builder.append("\n")
            }
        }
        
        return builder.toString()
    }
    
    private fun buildCapabilityAnalysisResult(implementations: Map<String, List<String>>, conceptDetails: ConceptInfo?, query: String): String {
        val builder = StringBuilder()
        val executionTime = getExecutionTime()
        
        builder.append("${getString(R.string.header_capability_analysis)}\n")
        builder.append("=" .repeat(50)).append("\n\n")
        builder.append("Capability: '$query'\n")
        builder.append("Execution Time: ${executionTime}ms\n")
        builder.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        builder.append("METACOGNITIVE CAPABILITY ASSESSMENT:\n")
        builder.append("-".repeat(40)).append("\n")
        
        if (conceptDetails != null) {
            builder.append("Concept Foundation:\n")
            builder.append("  Name: ${conceptDetails.name}\n")
            builder.append("  Knowledge Elements: ${conceptDetails.knowledge.size}\n")
            builder.append("  Has Implementation: ${conceptDetails.implementation != null}\n")
            builder.append("  Has Code Details: ${conceptDetails.codeDetails != null}\n\n")
        }
        
        builder.append("IMPLEMENTATION COVERAGE:\n")
        builder.append("-".repeat(30)).append("\n")
        
        if (implementations.isEmpty()) {
            builder.append("⚠️  No implementations found for this capability.\n")
            builder.append("This may indicate:\n")
            builder.append("  • Capability is not yet implemented\n")
            builder.append("  • Query needs refinement\n")
            builder.append("  • Implementation uses different naming\n\n")
        } else {
            var totalImplementations = 0
            implementations.forEach { (type, items) ->
                totalImplementations += items.size
                builder.append("$type: ${items.size} implementations\n")
                items.take(5).forEach { item ->
                    builder.append("  ✓ $item\n")
                }
                if (items.size > 5) {
                    builder.append("  ... and ${items.size - 5} more\n")
                }
                builder.append("\n")
            }
            
            builder.append("CAPABILITY STRENGTH ASSESSMENT:\n")
            builder.append("-".repeat(30)).append("\n")
            val strength = when {
                totalImplementations >= 10 -> "🟢 STRONG - Well implemented capability"
                totalImplementations >= 5 -> "🟡 MODERATE - Partially implemented capability"
                totalImplementations >= 1 -> "🟠 WEAK - Limited implementation"
                else -> "🔴 MISSING - No implementation found"
            }
            builder.append("$strength\n")
            builder.append("Total Implementations: $totalImplementations\n\n")
        }
        
        return builder.toString()
    }
    
    private fun buildRuntimeObjectResult(runtimeObject: JSONObject, query: String): String {
        val builder = StringBuilder()
        val executionTime = getExecutionTime()
        
        builder.append("${getString(R.string.header_runtime_object)}\n")
        builder.append("=" .repeat(50)).append("\n\n")
        builder.append("Object ID: '$query'\n")
        builder.append("Execution Time: ${executionTime}ms\n")
        builder.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        builder.append("RUNTIME OBJECT ANALYSIS:\n")
        builder.append("-".repeat(30)).append("\n")
        
        if (runtimeObject.length() == 0) {
            builder.append("No runtime object found with ID: $query\n\n")
            builder.append("This could mean:\n")
            builder.append("  • Object ID does not exist\n")
            builder.append("  • Object is not currently active\n")
            builder.append("  • Access permissions are restricted\n")
        } else {
            formatJsonObject(runtimeObject, builder, 0)
        }
        
        return builder.toString()
    }
    
    private fun buildFullIntrospectionResult(
        conceptDetails: ConceptInfo?,
        implementations: Map<String, List<String>>,
        searchResults: List<Map<String, Any>>,
        explanation: ExplanationInfo?,
        query: String
    ): String {
        val builder = StringBuilder()
        val executionTime = getExecutionTime()
        
        builder.append("${getString(R.string.header_full_introspection)}\n")
        builder.append("=" .repeat(60)).append("\n\n")
        builder.append("Comprehensive Analysis for: '$query'\n")
        builder.append("Execution Time: ${executionTime}ms\n")
        builder.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Section 1: Concept Analysis
        builder.append("🧠 CONCEPT KNOWLEDGE BASE\n")
        builder.append("-".repeat(40)).append("\n")
        if (conceptDetails != null) {
            builder.append("✓ Concept Found: ${conceptDetails.name}\n")
            builder.append("  Knowledge Elements: ${conceptDetails.knowledge.size}\n")
            conceptDetails.knowledge.entries.take(3).forEach { (key, value) ->
                builder.append("    • $key: ${value.toString().take(50)}...\n")
            }
            if (conceptDetails.knowledge.size > 3) {
                builder.append("    ... and ${conceptDetails.knowledge.size - 3} more elements\n")
            }
        } else {
            builder.append("⚠️  No concept details found\n")
        }
        builder.append("\n")
        
        // Section 2: Implementation Analysis
        builder.append("⚙️ IMPLEMENTATION MAPPING\n")
        builder.append("-".repeat(40)).append("\n")
        if (implementations.isNotEmpty()) {
            var totalImpl = 0
            implementations.forEach { (type, items) ->
                totalImpl += items.size
                builder.append("$type: ${items.size}\n")
            }
            builder.append("Total Implementations: $totalImpl\n")
        } else {
            builder.append("⚠️  No direct implementations found\n")
        }
        builder.append("\n")
        
        // Section 3: Search Results Summary
        builder.append("🔍 CONCEPT SEARCH RESULTS\n")
        builder.append("-".repeat(40)).append("\n")
        builder.append("Results Found: ${searchResults.size}\n")
        if (searchResults.isNotEmpty()) {
            searchResults.take(3).forEach { result ->
                val concept = result["concept"] as? String ?: "Unknown"
                builder.append("  • $concept\n")
            }
            if (searchResults.size > 3) {
                builder.append("  ... and ${searchResults.size - 3} more\n")
            }
        }
        builder.append("\n")
        
        // Section 4: Explanation Summary
        builder.append("📖 IMPLEMENTATION EXPLANATION\n")
        builder.append("-".repeat(40)).append("\n")
        if (explanation != null) {
            builder.append("✓ Explanation Available\n")
            builder.append("  Name: ${explanation.name}\n")
            builder.append("  Type: ${explanation.type}\n")
            builder.append("  Related Concepts: ${explanation.concepts.size}\n")
            builder.append("  Explanation Length: ${explanation.explanation.length} chars\n")
        } else {
            builder.append("⚠️  No explanation available\n")
        }
        builder.append("\n")
        
        // Section 5: Overall Assessment
        builder.append("📊 INTROSPECTION SUMMARY\n")
        builder.append("-".repeat(40)).append("\n")
        val completeness = calculateCompleteness(conceptDetails, implementations, searchResults, explanation)
        builder.append("Analysis Completeness: $completeness%\n")
        
        val recommendations = generateRecommendations(conceptDetails, implementations, searchResults, explanation, query)
        if (recommendations.isNotEmpty()) {
            builder.append("\nRecommendations:\n")
            recommendations.forEach { rec ->
                builder.append("  • $rec\n")
            }
        }
        
        return builder.toString()
    }
    
    // Helper methods
    
    private fun getSearchQuery(): String? {
        val query = searchInput.text?.toString()?.trim()
        if (query.isNullOrEmpty()) {
            statusText.text = getString(R.string.no_query_entered)
            return null
        }
        return query
    }
    
    private fun startPerformanceTracking() {
        operationStartTime = System.currentTimeMillis()
    }
    
    private fun getExecutionTime(): Long {
        return System.currentTimeMillis() - operationStartTime
    }
    
    private fun getCurrentTimestamp(): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
    }
    
    private fun handleError(errorMessage: String, exception: Exception) {
        val fullError = "$errorMessage\n\n" +
                "Error: ${exception.message}\n\n" +
                "Execution Time: ${getExecutionTime()}ms\n" +
                "Timestamp: ${getCurrentTimestamp()}\n\n" +
                "Stack Trace:\n${exception.stackTraceToString().take(800)}..."
        
        statusText.text = fullError
        exception.printStackTrace()
    }
    
    private fun enableButtons(enabled: Boolean) {
        queryConceptButton.isEnabled = enabled
        findImplementationButton.isEnabled = enabled
        explainImplementationButton.isEnabled = enabled
        searchConceptsButton.isEnabled = enabled
        capabilityAnalysisButton.isEnabled = enabled
        runtimeObjectButton.isEnabled = enabled
        fullIntrospectionButton.isEnabled = enabled
    }
    
    private fun showLoading(show: Boolean) {
        loader.visibility = if (show) View.VISIBLE else View.GONE
    }
    
    private fun displaySampleQueries() {
        val currentText = statusText.text.toString()
        val sampleQueries = "\n\n" + getString(R.string.sample_concept_queries) + "\n\n" +
                getString(R.string.sample_implementation_queries) + "\n\n" +
                getString(R.string.sample_capability_queries)
        
        statusText.text = currentText + sampleQueries
    }
    
    private fun formatJsonObject(jsonObject: JSONObject, builder: StringBuilder, indentLevel: Int) {
        val indent = "  ".repeat(indentLevel)
        val keys = jsonObject.keys()
        
        while (keys.hasNext()) {
            val key = keys.next()
            val value = jsonObject.get(key)
            
            builder.append("$indent$key: ")
            
            when (value) {
                is JSONObject -> {
                    builder.append("\n")
                    formatJsonObject(value, builder, indentLevel + 1)
                }
                is JSONArray -> {
                    builder.append("[\n")
                    for (i in 0 until value.length()) {
                        val item = value.get(i)
                        builder.append("$indent  ")
                        if (item is JSONObject) {
                            builder.append("{\n")
                            formatJsonObject(item, builder, indentLevel + 2)
                            builder.append("$indent  }\n")
                        } else {
                            builder.append("$item\n")
                        }
                    }
                    builder.append("$indent]\n")
                }
                else -> {
                    val valueStr = value.toString()
                    if (valueStr.length > 100) {
                        builder.append("${valueStr.take(100)}...\n")
                    } else {
                        builder.append("$valueStr\n")
                    }
                }
            }
        }
    }
    
    private fun calculateCompleteness(
        conceptDetails: ConceptInfo?,
        implementations: Map<String, List<String>>,
        searchResults: List<Map<String, Any>>,
        explanation: ExplanationInfo?
    ): Int {
        var score = 0
        
        if (conceptDetails != null) score += 25
        if (implementations.isNotEmpty()) score += 25
        if (searchResults.isNotEmpty()) score += 25
        if (explanation != null) score += 25
        
        return score
    }
    
    private fun generateRecommendations(
        conceptDetails: ConceptInfo?,
        implementations: Map<String, List<String>>,
        searchResults: List<Map<String, Any>>,
        explanation: ExplanationInfo?,
        query: String
    ): List<String> {
        val recommendations = mutableListOf<String>()
        
        if (conceptDetails == null) {
            recommendations.add("Try a more specific concept name or check spelling")
        }
        
        if (implementations.isEmpty()) {
            recommendations.add("Search for related capabilities or use broader terms")
        }
        
        if (searchResults.isEmpty()) {
            recommendations.add("Consider using synonyms or related terminology")
        }
        
        if (explanation == null) {
            recommendations.add("Try searching for specific class or method names")
        }
        
        // Add specific recommendations based on query patterns
        when {
            query.contains("detection", ignoreCase = true) -> {
                recommendations.add("Try 'BoundaryDetection' or 'AnomalyDetection'")
            }
            query.contains("confidence", ignoreCase = true) -> {
                recommendations.add("Try 'ConfidenceAssessment' or 'UncertaintyQuantification'")
            }
            query.contains("emotion", ignoreCase = true) -> {
                recommendations.add("Try 'EmotionalState' or 'EmotionProcessor'")
            }
        }
        
        return recommendations
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clean up resources if needed
    }
}
```


