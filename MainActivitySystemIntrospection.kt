package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main Activity for the System Introspection Module
 * Provides comprehensive interface to code analysis, concept mapping, and self-reference operations
 */
class MainActivitySystemIntrospection : AppCompatActivity() {
    
    private lateinit var systemIntrospectionBridge: SystemIntrospectionBridge
    private lateinit var systemIntrospectionService: SystemIntrospectionService
    
    // UI Components
    private lateinit var statusText: TextView
    private lateinit var loader: ProgressBar
    private lateinit var inputText: TextInputEditText
    private lateinit var operationChips: ChipGroup
    
    // Operation Buttons
    private lateinit var queryConceptButton: Button
    private lateinit var queryImplementationButton: Button
    private lateinit var explainImplementationButton: Button
    private lateinit var searchConceptsButton: Button
    private lateinit var analyzeCodeButton: Button
    private lateinit var memoryAccessButton: Button
    private lateinit var runDiagnosticsButton: Button
    private lateinit var systemStatusButton: Button
    private lateinit var comprehensiveAnalysisButton: Button
    private lateinit var generateSamplesButton: Button
    
    // System state
    private var isInitialized = false
    private var operationStartTime: Long = 0
    private var operationHistory = mutableListOf<String>()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_system_introspection)
        
        initializeViews()
        setupButtonListeners()
        setupChipGroupListener()
        initializeIntrospectionSystem()
    }
    
    private fun initializeViews() {
        statusText = findViewById(R.id.status_text)
        loader = findViewById(R.id.loader)
        inputText = findViewById(R.id.input_text)
        operationChips = findViewById(R.id.operation_chips)
        
        queryConceptButton = findViewById(R.id.query_concept_button)
        queryImplementationButton = findViewById(R.id.query_implementation_button)
        explainImplementationButton = findViewById(R.id.explain_implementation_button)
        searchConceptsButton = findViewById(R.id.search_concepts_button)
        analyzeCodeButton = findViewById(R.id.analyze_code_button)
        memoryAccessButton = findViewById(R.id.memory_access_button)
        runDiagnosticsButton = findViewById(R.id.run_diagnostics_button)
        systemStatusButton = findViewById(R.id.system_status_button)
        comprehensiveAnalysisButton = findViewById(R.id.comprehensive_analysis_button)
        generateSamplesButton = findViewById(R.id.generate_samples_button)
    }
    
    private fun setupButtonListeners() {
        queryConceptButton.setOnClickListener { performConceptQuery() }
        queryImplementationButton.setOnClickListener { performImplementationQuery() }
        explainImplementationButton.setOnClickListener { performImplementationExplanation() }
        searchConceptsButton.setOnClickListener { performConceptSearch() }
        analyzeCodeButton.setOnClickListener { performCodeAnalysis() }
        memoryAccessButton.setOnClickListener { performMemoryAccess() }
        runDiagnosticsButton.setOnClickListener { performDiagnostics() }
        systemStatusButton.setOnClickListener { displaySystemStatus() }
        comprehensiveAnalysisButton.setOnClickListener { performComprehensiveAnalysis() }
        generateSamplesButton.setOnClickListener { generateSampleQueries() }
    }
    
    private fun setupChipGroupListener() {
        // Add operation type chips
        val operations = listOf(
            "Concepts" to "concepts",
            "Implementation" to "implementation", 
            "Code Analysis" to "code",
            "Memory" to "memory",
            "Diagnostics" to "diagnostics",
            "System" to "system"
        )
        
        operations.forEach { (display, value) ->
            val chip = Chip(this).apply {
                text = display
                tag = value
                isCheckable = true
            }
            operationChips.addView(chip)
        }
        
        operationChips.setOnCheckedStateChangeListener { group, checkedIds ->
            if (checkedIds.isNotEmpty()) {
                val selectedChip = findViewById<Chip>(checkedIds.first())
                val operation = selectedChip.tag.toString()
                highlightRelevantOperations(operation)
            }
        }
    }
    
    private fun initializeIntrospectionSystem() {
        showLoading(true)
        statusText.text = "Initializing System Introspection Module..."
        
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    systemIntrospectionBridge = SystemIntrospectionBridge.getInstance(applicationContext)
                    systemIntrospectionService = SystemIntrospectionService(applicationContext)
                    
                    // Test system initialization
                    val systemStatus = systemIntrospectionBridge.getSystemStatus()
                    if (!systemStatus.has("error")) {
                        isInitialized = true
                    }
                }
                
                if (isInitialized) {
                    displayInitializationSuccess()
                    enableButtons(true)
                } else {
                    statusText.text = "❌ Failed to initialize System Introspection Module"
                }
            } catch (e: Exception) {
                statusText.text = "❌ Initialization Error: ${e.message}\n\n" +
                        "Stack Trace:\n${e.stackTraceToString().take(500)}..."
                e.printStackTrace()
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun displayInitializationSuccess() {
        val welcomeText = """
            🔍 SYSTEM INTROSPECTION MODULE INITIALIZED 🔍
            
            ═══════════════════════════════════════════════
            
            📋 CONCEPT MAPPING: Ready
               • Abstract concept definitions
               • Implementation mappings
               • Knowledge comparison framework
            
            🔧 CODE ANALYSIS: Ready
               • Python file parsing
               • Class and method extraction
               • Function definition analysis
               • Complexity calculation
            
            🧠 SELF-REFERENCE FRAMEWORK: Ready
               • General vs specific knowledge
               • Implementation alignment analysis
               • Knowledge gap detection
            
            💾 MEMORY ACCESS LAYER: Ready
               • Runtime object registration
               • Attribute querying
               • Method invocation
               • State management
            
            🏥 DIAGNOSTIC FRAMEWORK: Ready
               • Component health monitoring
               • Performance metrics
               • Error detection and reporting
               • System validation
            
            ═══════════════════════════════════════════════
            
            Enter concept names, implementation identifiers, or
            code elements to begin introspective analysis.
            
            The system knows itself...
        """.trimIndent()
        
        statusText.text = welcomeText
    }
    
    private fun performConceptQuery() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔍 Querying concept information and knowledge mappings..."
        startOperation("Concept Query")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val conceptName = inputText.text.toString().trim()
                    
                    // Query the concept
                    val conceptInfo = systemIntrospectionBridge.queryConcept(conceptName)
                    
                    // Get additional knowledge if available
                    val knowledge = systemIntrospectionBridge.getConceptKnowledge(conceptName)
                    
                    // Compare knowledge if possible
                    val comparison = systemIntrospectionBridge.compareKnowledge(conceptName)
                    
                    buildConceptQueryReport(conceptInfo, knowledge, comparison, conceptName)
                }
                
                statusText.text = result
                recordOperation("Concept Query completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Concept Query", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performImplementationQuery() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔧 Querying implementation details and code structure..."
        startOperation("Implementation Query")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val implementationName = inputText.text.toString().trim()
                    
                    // Query the implementation
                    val implInfo = systemIntrospectionBridge.queryImplementation(implementationName)
                    
                    // Get associated concepts
                    val concepts = systemIntrospectionBridge.getConceptsForImplementation(implementationName)
                    
                    // Get class hierarchy if it's a class
                    val hierarchy = if (implInfo.optString("type") == "class") {
                        systemIntrospectionBridge.getClassHierarchy(implementationName)
                    } else null
                    
                    buildImplementationQueryReport(implInfo, concepts, hierarchy, implementationName)
                }
                
                statusText.text = result
                recordOperation("Implementation Query completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Implementation Query", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performImplementationExplanation() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "📖 Generating detailed implementation explanation..."
        startOperation("Implementation Explanation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val implementationName = inputText.text.toString().trim()
                    
                    // Get explanation
                    val explanation = systemIntrospectionBridge.explainImplementation(implementationName)
                    
                    // Get implementation details for context
                    val implInfo = systemIntrospectionBridge.queryImplementation(implementationName)
                    
                    buildImplementationExplanationReport(explanation, implInfo, implementationName)
                }
                
                statusText.text = result
                recordOperation("Implementation Explanation completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Implementation Explanation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performConceptSearch() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔎 Searching concepts and mapping relationships..."
        startOperation("Concept Search")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val searchQuery = inputText.text.toString().trim()
                    
                    // Search concepts
                    val searchResults = systemIntrospectionBridge.searchConcepts(searchQuery)
                    
                    // Find implementations for the query
                    val implementations = systemIntrospectionBridge.findImplementationForConcept(searchQuery)
                    
                    buildConceptSearchReport(searchResults, implementations, searchQuery)
                }
                
                statusText.text = result
                recordOperation("Concept Search completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Concept Search", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performCodeAnalysis() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "📄 Analyzing code structure and extracting definitions..."
        startOperation("Code Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val input = inputText.text.toString().trim()
                    
                    // Determine if input is a file path, class name, function name, or method
                    val analysisResults = mutableMapOf<String, JSONObject>()
                    
                    // Try different analysis approaches
                    if (input.endsWith(".py")) {
                        // File analysis
                        val fileAnalysis = systemIntrospectionBridge.analyzeFile(input)
                        analysisResults["file_analysis"] = fileAnalysis
                    } else {
                        // Try as class name
                        val classDefinition = systemIntrospectionBridge.getClassDefinition(input)
                        if (!classDefinition.has("error")) {
                            analysisResults["class_definition"] = classDefinition
                        }
                        
                        // Try as function name
                        val functionDefinition = systemIntrospectionBridge.getFunctionDefinition(input)
                        if (!functionDefinition.has("error")) {
                            analysisResults["function_definition"] = functionDefinition
                        }
                        
                        // Try as method (if contains dot)
                        if (input.contains(".")) {
                            val parts = input.split(".")
                            if (parts.size == 2) {
                                val methodDefinition = systemIntrospectionBridge.getMethodDefinition(parts[0], parts[1])
                                if (!methodDefinition.has("error")) {
                                    analysisResults["method_definition"] = methodDefinition
                                }
                            }
                        }
                    }
                    
                    buildCodeAnalysisReport(analysisResults, input)
                }
                
                statusText.text = result
                recordOperation("Code Analysis completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Code Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performMemoryAccess() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "💾 Accessing runtime objects and memory state..."
        startOperation("Memory Access")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val input = inputText.text.toString().trim()
                    
                    // Try to get runtime object
                    val runtimeObject = systemIntrospectionBridge.getRuntimeObject(input)
                    
                    // Try to find objects by type
                    val objectsByType = systemIntrospectionBridge.findObjectsByType(input)
                    
                    // Get component diagnostics for memory access
                    val memoryDiagnostics = systemIntrospectionBridge.getComponentDiagnostics("memory_access")
                    
                    buildMemoryAccessReport(runtimeObject, objectsByType, memoryDiagnostics, input)
                }
                
                statusText.text = result
                recordOperation("Memory Access completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Memory Access", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performDiagnostics() {
        showLoading(true)
        statusText.text = "🏥 Running comprehensive system diagnostics..."
        startOperation("System Diagnostics")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    // Get input for specific test or run all
                    val input = inputText.text.toString().trim()
                    
                    val diagnosticsResult = if (input.isNotEmpty() && input != "all") {
                        // Run specific diagnostic test
                        systemIntrospectionBridge.runSpecificDiagnostic(input)
                    } else {
                        // Run all diagnostics
                        systemIntrospectionBridge.runDiagnostics()
                    }
                    
                    // Get diagnostic summary
                    val summary = systemIntrospectionBridge.getDiagnosticSummary()
                    
                    // Get available tests
                    val availableTests = systemIntrospectionBridge.getAvailableDiagnosticTests()
                    
                    buildDiagnosticsReport(diagnosticsResult, summary, availableTests, input)
                }
                
                statusText.text = result
                recordOperation("Diagnostics completed")
            } catch (e: Exception) {
                handleOperationError("System Diagnostics", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun displaySystemStatus() {
        showLoading(true)
        statusText.text = "📊 Retrieving comprehensive system status..."
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val systemStatus = systemIntrospectionBridge.getSystemStatus()
                    val connectivity = systemIntrospectionBridge.validateSystemConnectivity()
                    val performanceMetrics = systemIntrospectionBridge.getPerformanceMetrics()
                    
                    buildSystemStatusReport(systemStatus, connectivity, performanceMetrics)
                }
                
                statusText.text = result
                recordOperation("System Status retrieved")
            } catch (e: Exception) {
                handleOperationError("System Status", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performComprehensiveAnalysis() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🌟 Performing comprehensive introspective analysis..."
        startOperation("Comprehensive Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val input = inputText.text.toString().trim()
                    
                    // Perform comprehensive analysis
                    val analysis = systemIntrospectionBridge.performComprehensiveAnalysis(input)
                    
                    // Get system health report
                    val healthReport = systemIntrospectionBridge.generateHealthReport()
                    
                    buildComprehensiveAnalysisReport(analysis, healthReport, input)
                }
                
                statusText.text = result
                recordOperation("Comprehensive Analysis completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Comprehensive Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun generateSampleQueries() {
        showLoading(true)
        statusText.text = "📝 Generating sample queries for testing..."
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val sampleQueries = systemIntrospectionBridge.generateSampleQueries()
                    
                    buildSampleQueriesReport(sampleQueries)
                }
                
                // Update input field with first sample
                if (result.contains("Sample Query:")) {
                    val firstSample = result.substringAfter("Sample Query: ").substringBefore("\n").trim()
                    if (firstSample.isNotEmpty()) {
                        inputText.setText(firstSample)
                    }
                }
                
                statusText.text = result
                recordOperation("Sample queries generated")
            } catch (e: Exception) {
                handleOperationError("Sample Generation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    // ============================================================================
    // REPORT BUILDING METHODS
    // ============================================================================
    
    private fun buildConceptQueryReport(
        conceptInfo: JSONObject,
        knowledge: JSONObject,
        comparison: JSONObject,
        conceptName: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔍 CONCEPT QUERY RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Concept: $conceptName\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (conceptInfo.has("error")) {
            report.append("❌ CONCEPT NOT FOUND\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Error: ${conceptInfo.optString("error_message", "Unknown error")}\n\n")
            report.append("Suggestions:\n")
            report.append("• Check concept name spelling\n")
            report.append("• Try searching with partial terms\n")
            report.append("• Use the concept search function\n")
            return report.toString()
        }
        
        // Basic concept information
        if (conceptInfo.has("diagnostics")) {
            val diagnostics = conceptInfo.getJSONObject("diagnostics")
            report.append("CONCEPT STATUS:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Has General Knowledge: ${getStatusIcon(diagnostics.optBoolean("has_general_knowledge"))}\n")
            report.append("Has Implementation: ${getStatusIcon(diagnostics.optBoolean("has_specific_implementation"))}\n")
            report.append("Implementation Found: ${getStatusIcon(diagnostics.optBoolean("implementation_found"))}\n")
            report.append("Code Details: ${getStatusIcon(diagnostics.optBoolean("code_details_extracted"))}\n\n")
        }
        
        // Knowledge information
        if (knowledge.has("general")) {
            val general = knowledge.getJSONObject("general")
            report.append("GENERAL KNOWLEDGE:\n")
            report.append("-".repeat(30)).append("\n")
            general.keys().forEach { key ->
                val value = general.get(key)
                report.append("• ${key.capitalize()}: ${formatValue(value)}\n")
            }
            report.append("\n")
        }
        
        if (knowledge.has("specific")) {
            val specific = knowledge.getJSONObject("specific")
            report.append("SPECIFIC IMPLEMENTATION:\n")
            report.append("-".repeat(30)).append("\n")
            
            if (specific.has("classes")) {
                val classes = specific.getJSONArray("classes")
                if (classes.length() > 0) {
                    report.append("Classes: ")
                    val classList = mutableListOf<String>()
                    for (i in 0 until classes.length()) {
                        classList.add(classes.getString(i))
                    }
                    report.append("${classList.joinToString(", ")}\n")
                }
            }
            
            if (specific.has("methods")) {
                val methods = specific.getJSONArray("methods")
                if (methods.length() > 0) {
                    report.append("Methods: ")
                    val methodList = mutableListOf<String>()
                    for (i in 0 until methods.length()) {
                        methodList.add(methods.getString(i))
                    }
                    report.append("${methodList.joinToString(", ")}\n")
                }
            }
            
            if (specific.has("functions")) {
                val functions = specific.getJSONArray("functions")
                if (functions.length() > 0) {
                    report.append("Functions: ")
                    val functionList = mutableListOf<String>()
                    for (i in 0 until functions.length()) {
                        functionList.add(functions.getString(i))
                    }
                    report.append("${functionList.joinToString(", ")}\n")
                }
            }
            report.append("\n")
        }
        
        // Knowledge comparison
        if (!comparison.has("error") && comparison.optString("comparison") == "complete") {
            report.append("KNOWLEDGE ALIGNMENT:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Alignment Level: ${comparison.optString("alignment", "unknown").uppercase()}\n")
            
            if (comparison.has("alignments")) {
                val alignments = comparison.getJSONArray("alignments")
                if (alignments.length() > 0) {
                    report.append("Aligned Aspects: ")
                    val alignmentList = mutableListOf<String>()
                    for (i in 0 until alignments.length()) {
                        alignmentList.add(alignments.getString(i))
                    }
                    report.append("${alignmentList.joinToString(", ")}\n")
                }
            }
            
            if (comparison.has("differences")) {
                val differences = comparison.getJSONArray("differences")
                if (differences.length() > 0) {
                    report.append("Differences: ")
                    val differenceList = mutableListOf<String>()
                    for (i in 0 until differences.length()) {
                        differenceList.add(differences.getString(i))
                    }
                    report.append("${differenceList.joinToString(", ")}\n")
                }
            }
            
            if (comparison.has("detailed_analysis")) {
                val analysis = comparison.getJSONObject("detailed_analysis")
                val coverage = analysis.optDouble("conceptual_coverage", 0.0)
                val completeness = analysis.optDouble("implementation_completeness", 0.0)
                
                report.append("Conceptual Coverage: ${String.format("%.1f%%", coverage * 100)}\n")
                report.append("Implementation Completeness: ${String.format("%.1f%%", completeness * 100)}\n")
                
                if (analysis.has("recommendations")) {
                    val recommendations = analysis.getJSONArray("recommendations")
                    if (recommendations.length() > 0) {
                        report.append("\nRecommendations:\n")
                        for (i in 0 until recommendations.length()) {
                            report.append("• ${recommendations.getString(i)}\n")
                        }
                    }
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildImplementationQueryReport(
        implInfo: JSONObject,
        concepts: JSONArray,
        hierarchy: JSONObject?,
        implementationName: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔧 IMPLEMENTATION QUERY RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Implementation: $implementationName\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (implInfo.has("error")) {
            report.append("❌ IMPLEMENTATION NOT FOUND\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Error: ${implInfo.optString("error_message", "Unknown error")}\n\n")
            report.append("Suggestions:\n")
            report.append("• Check implementation name spelling\n")
            report.append("• Try with full module path\n")
            report.append("• Use format 'ClassName.methodName' for methods\n")
            return report.toString()
        }
        
        // Basic implementation info
        report.append("IMPLEMENTATION DETAILS:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Type: ${implInfo.optString("type", "unknown").uppercase()}\n")
        
        if (implInfo.has("details")) {
            val details = implInfo.getJSONObject("details")
            
            report.append("Module: ${details.optString("module", "unknown")}\n")
            
            if (details.has("line_number")) {
                val lineNumber = details.optInt("line_number", 0)
                val endLineNumber = details.optInt("end_line_number", 0)
                if (lineNumber > 0) {
                    report.append("Location: Line $lineNumber")
                    if (endLineNumber > 0 && endLineNumber != lineNumber) {
                        report.append("-$endLineNumber")
                    }
                    report.append("\n")
                }
            }
            
            if (details.has("complexity")) {
                report.append("Complexity: ${details.optInt("complexity", 0)}\n")
            }
            
            // Docstring
            val docstring = details.optString("docstring", "")
            if (docstring.isNotEmpty()) {
                report.append("\nDescription:\n")
                if (docstring.length > 200) {
                    report.append("${docstring.take(200)}...\n")
                } else {
                    report.append("$docstring\n")
                }
            }
            
            // Parameters for functions/methods
            if (details.has("parameters")) {
                val parameters = details.getJSONArray("parameters")
                if (parameters.length() > 0) {
                    report.append("\nParameters:\n")
                    for (i in 0 until parameters.length()) {
                        val param = parameters.getJSONObject(i)
                        val name = param.optString("name", "")
                        val type = param.optString("type", "")
                        val default = param.opt("default")
                        
                        report.append("• $name")
                        if (type.isNotEmpty()) report.append(": $type")
                        if (default != null && default.toString() != "null") {
                            report.append(" = $default")
                        }
                        report.append("\n")
                    }
                }
            }
            
            // Methods for classes
            if (details.has("methods")) {
                val methods = details.getJSONArray("methods")
                if (methods.length() > 0) {
                    report.append("\nMethods (${methods.length()}):\n")
                    for (i in 0 until minOf(10, methods.length())) {
                        val method = methods.getJSONObject(i)
                        val name = method.optString("name", "")
                        val complexity = method.optInt("complexity", 0)
                        val docstring = method.optString("docstring", "")
                        
                        report.append("• $name (complexity: $complexity)\n")
                        if (docstring.isNotEmpty()) {
                            val summary = docstring.split(".").firstOrNull()?.trim()
                            if (!summary.isNullOrEmpty()) {
                                report.append("  $summary\n")
                            }
                        }
                    }
                    if (methods.length() > 10) {
                        report.append("... and ${methods.length() - 10} more methods\n")
                    }
                }
            }
            
            // Attributes for classes
            if (details.has("attributes")) {
                val attributes = details.getJSONArray("attributes")
                if (attributes.length() > 0) {
                    report.append("\nAttributes:\n")
                    for (i in 0 until attributes.length()) {
                        val attr = attributes.getJSONObject(i)
                        val name = attr.optString("name", "")
                        val value = attr.opt("value")
                        
                        report.append("• $name")
                        if (value != null && value.toString() != "null") {
                          package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main Activity for the System Introspection Module
 * Provides comprehensive interface to code analysis, concept mapping, and self-reference operations
 */
class MainActivitySystemIntrospection : AppCompatActivity() {
    
    private lateinit var systemIntrospectionBridge: SystemIntrospectionBridge
    private lateinit var systemIntrospectionService: SystemIntrospectionService
    
    // UI Components
    private lateinit var statusText: TextView
    private lateinit var loader: ProgressBar
    private lateinit var inputText: TextInputEditText
    private lateinit var operationChips: ChipGroup
    
    // Operation Buttons
    private lateinit var queryConceptButton: Button
    private lateinit var queryImplementationButton: Button
    private lateinit var explainImplementationButton: Button
    private lateinit var searchConceptsButton: Button
    private lateinit var analyzeCodeButton: Button
    private lateinit var memoryAccessButton: Button
    private lateinit var runDiagnosticsButton: Button
    private lateinit var systemStatusButton: Button
    private lateinit var comprehensiveAnalysisButton: Button
    private lateinit var generateSamplesButton: Button
    
    // System state
    private var isInitialized = false
    private var operationStartTime: Long = 0
    private var operationHistory = mutableListOf<String>()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_system_introspection)
        
        initializeViews()
        setupButtonListeners()
        setupChipGroupListener()
        initializeIntrospectionSystem()
    }
    
    private fun initializeViews() {
        statusText = findViewById(R.id.status_text)
        loader = findViewById(R.id.loader)
        inputText = findViewById(R.id.input_text)
        operationChips = findViewById(R.id.operation_chips)
        
        queryConceptButton = findViewById(R.id.query_concept_button)
        queryImplementationButton = findViewById(R.id.query_implementation_button)
        explainImplementationButton = findViewById(R.id.explain_implementation_button)
        searchConceptsButton = findViewById(R.id.search_concepts_button)
        analyzeCodeButton = findViewById(R.id.analyze_code_button)
        memoryAccessButton = findViewById(R.id.memory_access_button)
        runDiagnosticsButton = findViewById(R.id.run_diagnostics_button)
        systemStatusButton = findViewById(R.id.system_status_button)
        comprehensiveAnalysisButton = findViewById(R.id.comprehensive_analysis_button)
        generateSamplesButton = findViewById(R.id.generate_samples_button)
    }
    
    private fun setupButtonListeners() {
        queryConceptButton.setOnClickListener { performConceptQuery() }
        queryImplementationButton.setOnClickListener { performImplementationQuery() }
        explainImplementationButton.setOnClickListener { performImplementationExplanation() }
        searchConceptsButton.setOnClickListener { performConceptSearch() }
        analyzeCodeButton.setOnClickListener { performCodeAnalysis() }
        memoryAccessButton.setOnClickListener { performMemoryAccess() }
        runDiagnosticsButton.setOnClickListener { performDiagnostics() }
        systemStatusButton.setOnClickListener { displaySystemStatus() }
        comprehensiveAnalysisButton.setOnClickListener { performComprehensiveAnalysis() }
        generateSamplesButton.setOnClickListener { generateSampleQueries() }
    }
    
    private fun setupChipGroupListener() {
        // Add operation type chips
        val operations = listOf(
            "Concepts" to "concepts",
            "Implementation" to "implementation", 
            "Code Analysis" to "code",
            "Memory" to "memory",
            "Diagnostics" to "diagnostics",
            "System" to "system"
        )
        
        operations.forEach { (display, value) ->
            val chip = Chip(this).apply {
                text = display
                tag = value
                isCheckable = true
            }
            operationChips.addView(chip)
        }
        
        operationChips.setOnCheckedStateChangeListener { group, checkedIds ->
            if (checkedIds.isNotEmpty()) {
                val selectedChip = findViewById<Chip>(checkedIds.first())
                val operation = selectedChip.tag.toString()
                highlightRelevantOperations(operation)
            }
        }
    }
    
    private fun initializeIntrospectionSystem() {
        showLoading(true)
        statusText.text = "Initializing System Introspection Module..."
        
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    systemIntrospectionBridge = SystemIntrospectionBridge.getInstance(applicationContext)
                    systemIntrospectionService = SystemIntrospectionService(applicationContext)
                    
                    // Test system initialization
                    val systemStatus = systemIntrospectionBridge.getSystemStatus()
                    if (!systemStatus.has("error")) {
                        isInitialized = true
                    }
                }
                
                if (isInitialized) {
                    displayInitializationSuccess()
                    enableButtons(true)
                } else {
                    statusText.text = "❌ Failed to initialize System Introspection Module"
                }
            } catch (e: Exception) {
                statusText.text = "❌ Initialization Error: ${e.message}\n\n" +
                        "Stack Trace:\n${e.stackTraceToString().take(500)}..."
                e.printStackTrace()
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun displayInitializationSuccess() {
        val welcomeText = """
            🔍 SYSTEM INTROSPECTION MODULE INITIALIZED 🔍
            
            ═══════════════════════════════════════════════
            
            📋 CONCEPT MAPPING: Ready
               • Abstract concept definitions
               • Implementation mappings
               • Knowledge comparison framework
            
            🔧 CODE ANALYSIS: Ready
               • Python file parsing
               • Class and method extraction
               • Function definition analysis
               • Complexity calculation
            
            🧠 SELF-REFERENCE FRAMEWORK: Ready
               • General vs specific knowledge
               • Implementation alignment analysis
               • Knowledge gap detection
            
            💾 MEMORY ACCESS LAYER: Ready
               • Runtime object registration
               • Attribute querying
               • Method invocation
               • State management
            
            🏥 DIAGNOSTIC FRAMEWORK: Ready
               • Component health monitoring
               • Performance metrics
               • Error detection and reporting
               • System validation
            
            ═══════════════════════════════════════════════
            
            Enter concept names, implementation identifiers, or
            code elements to begin introspective analysis.
            
            The system knows itself...
        """.trimIndent()
        
        statusText.text = welcomeText
    }
    
    private fun performConceptQuery() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔍 Querying concept information and knowledge mappings..."
        startOperation("Concept Query")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val conceptName = inputText.text.toString().trim()
                    
                    // Query the concept
                    val conceptInfo = systemIntrospectionBridge.queryConcept(conceptName)
                    
                    // Get additional knowledge if available
                    val knowledge = systemIntrospectionBridge.getConceptKnowledge(conceptName)
                    
                    // Compare knowledge if possible
                    val comparison = systemIntrospectionBridge.compareKnowledge(conceptName)
                    
                    buildConceptQueryReport(conceptInfo, knowledge, comparison, conceptName)
                }
                
                statusText.text = result
                recordOperation("Concept Query completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Concept Query", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performImplementationQuery() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔧 Querying implementation details and code structure..."
        startOperation("Implementation Query")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val implementationName = inputText.text.toString().trim()
                    
                    // Query the implementation
                    val implInfo = systemIntrospectionBridge.queryImplementation(implementationName)
                    
                    // Get associated concepts
                    val concepts = systemIntrospectionBridge.getConceptsForImplementation(implementationName)
                    
                    // Get class hierarchy if it's a class
                    val hierarchy = if (implInfo.optString("type") == "class") {
                        systemIntrospectionBridge.getClassHierarchy(implementationName)
                    } else null
                    
                    buildImplementationQueryReport(implInfo, concepts, hierarchy, implementationName)
                }
                
                statusText.text = result
                recordOperation("Implementation Query completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Implementation Query", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performImplementationExplanation() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "📖 Generating detailed implementation explanation..."
        startOperation("Implementation Explanation")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val implementationName = inputText.text.toString().trim()
                    
                    // Get explanation
                    val explanation = systemIntrospectionBridge.explainImplementation(implementationName)
                    
                    // Get implementation details for context
                    val implInfo = systemIntrospectionBridge.queryImplementation(implementationName)
                    
                    buildImplementationExplanationReport(explanation, implInfo, implementationName)
                }
                
                statusText.text = result
                recordOperation("Implementation Explanation completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Implementation Explanation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performConceptSearch() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🔎 Searching concepts and mapping relationships..."
        startOperation("Concept Search")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val searchQuery = inputText.text.toString().trim()
                    
                    // Search concepts
                    val searchResults = systemIntrospectionBridge.searchConcepts(searchQuery)
                    
                    // Find implementations for the query
                    val implementations = systemIntrospectionBridge.findImplementationForConcept(searchQuery)
                    
                    buildConceptSearchReport(searchResults, implementations, searchQuery)
                }
                
                statusText.text = result
                recordOperation("Concept Search completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Concept Search", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performCodeAnalysis() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "📄 Analyzing code structure and extracting definitions..."
        startOperation("Code Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val input = inputText.text.toString().trim()
                    
                    // Determine if input is a file path, class name, function name, or method
                    val analysisResults = mutableMapOf<String, JSONObject>()
                    
                    // Try different analysis approaches
                    if (input.endsWith(".py")) {
                        // File analysis
                        val fileAnalysis = systemIntrospectionBridge.analyzeFile(input)
                        analysisResults["file_analysis"] = fileAnalysis
                    } else {
                        // Try as class name
                        val classDefinition = systemIntrospectionBridge.getClassDefinition(input)
                        if (!classDefinition.has("error")) {
                            analysisResults["class_definition"] = classDefinition
                        }
                        
                        // Try as function name
                        val functionDefinition = systemIntrospectionBridge.getFunctionDefinition(input)
                        if (!functionDefinition.has("error")) {
                            analysisResults["function_definition"] = functionDefinition
                        }
                        
                        // Try as method (if contains dot)
                        if (input.contains(".")) {
                            val parts = input.split(".")
                            if (parts.size == 2) {
                                val methodDefinition = systemIntrospectionBridge.getMethodDefinition(parts[0], parts[1])
                                if (!methodDefinition.has("error")) {
                                    analysisResults["method_definition"] = methodDefinition
                                }
                            }
                        }
                    }
                    
                    buildCodeAnalysisReport(analysisResults, input)
                }
                
                statusText.text = result
                recordOperation("Code Analysis completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Code Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performMemoryAccess() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "💾 Accessing runtime objects and memory state..."
        startOperation("Memory Access")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val input = inputText.text.toString().trim()
                    
                    // Try to get runtime object
                    val runtimeObject = systemIntrospectionBridge.getRuntimeObject(input)
                    
                    // Try to find objects by type
                    val objectsByType = systemIntrospectionBridge.findObjectsByType(input)
                    
                    // Get component diagnostics for memory access
                    val memoryDiagnostics = systemIntrospectionBridge.getComponentDiagnostics("memory_access")
                    
                    buildMemoryAccessReport(runtimeObject, objectsByType, memoryDiagnostics, input)
                }
                
                statusText.text = result
                recordOperation("Memory Access completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Memory Access", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performDiagnostics() {
        showLoading(true)
        statusText.text = "🏥 Running comprehensive system diagnostics..."
        startOperation("System Diagnostics")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    // Get input for specific test or run all
                    val input = inputText.text.toString().trim()
                    
                    val diagnosticsResult = if (input.isNotEmpty() && input != "all") {
                        // Run specific diagnostic test
                        systemIntrospectionBridge.runSpecificDiagnostic(input)
                    } else {
                        // Run all diagnostics
                        systemIntrospectionBridge.runDiagnostics()
                    }
                    
                    // Get diagnostic summary
                    val summary = systemIntrospectionBridge.getDiagnosticSummary()
                    
                    // Get available tests
                    val availableTests = systemIntrospectionBridge.getAvailableDiagnosticTests()
                    
                    buildDiagnosticsReport(diagnosticsResult, summary, availableTests, input)
                }
                
                statusText.text = result
                recordOperation("Diagnostics completed")
            } catch (e: Exception) {
                handleOperationError("System Diagnostics", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun displaySystemStatus() {
        showLoading(true)
        statusText.text = "📊 Retrieving comprehensive system status..."
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val systemStatus = systemIntrospectionBridge.getSystemStatus()
                    val connectivity = systemIntrospectionBridge.validateSystemConnectivity()
                    val performanceMetrics = systemIntrospectionBridge.getPerformanceMetrics()
                    
                    buildSystemStatusReport(systemStatus, connectivity, performanceMetrics)
                }
                
                statusText.text = result
                recordOperation("System Status retrieved")
            } catch (e: Exception) {
                handleOperationError("System Status", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun performComprehensiveAnalysis() {
        if (!validateSystemAndInput()) return
        
        showLoading(true)
        statusText.text = "🌟 Performing comprehensive introspective analysis..."
        startOperation("Comprehensive Analysis")
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val input = inputText.text.toString().trim()
                    
                    // Perform comprehensive analysis
                    val analysis = systemIntrospectionBridge.performComprehensiveAnalysis(input)
                    
                    // Get system health report
                    val healthReport = systemIntrospectionBridge.generateHealthReport()
                    
                    buildComprehensiveAnalysisReport(analysis, healthReport, input)
                }
                
                statusText.text = result
                recordOperation("Comprehensive Analysis completed for: ${inputText.text}")
            } catch (e: Exception) {
                handleOperationError("Comprehensive Analysis", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun generateSampleQueries() {
        showLoading(true)
        statusText.text = "📝 Generating sample queries for testing..."
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val sampleQueries = systemIntrospectionBridge.generateSampleQueries()
                    
                    buildSampleQueriesReport(sampleQueries)
                }
                
                // Update input field with first sample
                if (result.contains("Sample Query:")) {
                    val firstSample = result.substringAfter("Sample Query: ").substringBefore("\n").trim()
                    if (firstSample.isNotEmpty()) {
                        inputText.setText(firstSample)
                    }
                }
                
                statusText.text = result
                recordOperation("Sample queries generated")
            } catch (e: Exception) {
                handleOperationError("Sample Generation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    // ============================================================================
    // REPORT BUILDING METHODS
    // ============================================================================
    
    private fun buildConceptQueryReport(
        conceptInfo: JSONObject,
        knowledge: JSONObject,
        comparison: JSONObject,
        conceptName: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔍 CONCEPT QUERY RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Concept: $conceptName\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (conceptInfo.has("error")) {
            report.append("❌ CONCEPT NOT FOUND\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Error: ${conceptInfo.optString("error_message", "Unknown error")}\n\n")
            report.append("Suggestions:\n")
            report.append("• Check concept name spelling\n")
            report.append("• Try searching with partial terms\n")
            report.append("• Use the concept search function\n")
            return report.toString()
        }
        
        // Basic concept information
        if (conceptInfo.has("diagnostics")) {
            val diagnostics = conceptInfo.getJSONObject("diagnostics")
            report.append("CONCEPT STATUS:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Has General Knowledge: ${getStatusIcon(diagnostics.optBoolean("has_general_knowledge"))}\n")
            report.append("Has Implementation: ${getStatusIcon(diagnostics.optBoolean("has_specific_implementation"))}\n")
            report.append("Implementation Found: ${getStatusIcon(diagnostics.optBoolean("implementation_found"))}\n")
            report.append("Code Details: ${getStatusIcon(diagnostics.optBoolean("code_details_extracted"))}\n\n")
        }
        
        // Knowledge information
        if (knowledge.has("general")) {
            val general = knowledge.getJSONObject("general")
            report.append("GENERAL KNOWLEDGE:\n")
            report.append("-".repeat(30)).append("\n")
            general.keys().forEach { key ->
                val value = general.get(key)
                report.append("• ${key.capitalize()}: ${formatValue(value)}\n")
            }
            report.append("\n")
        }
        
        if (knowledge.has("specific")) {
            val specific = knowledge.getJSONObject("specific")
            report.append("SPECIFIC IMPLEMENTATION:\n")
            report.append("-".repeat(30)).append("\n")
            
            if (specific.has("classes")) {
                val classes = specific.getJSONArray("classes")
                if (classes.length() > 0) {
                    report.append("Classes: ")
                    val classList = mutableListOf<String>()
                    for (i in 0 until classes.length()) {
                        classList.add(classes.getString(i))
                    }
                    report.append("${classList.joinToString(", ")}\n")
                }
            }
            
            if (specific.has("methods")) {
                val methods = specific.getJSONArray("methods")
                if (methods.length() > 0) {
                    report.append("Methods: ")
                    val methodList = mutableListOf<String>()
                    for (i in 0 until methods.length()) {
                        methodList.add(methods.getString(i))
                    }
                    report.append("${methodList.joinToString(", ")}\n")
                }
            }
            
            if (specific.has("functions")) {
                val functions = specific.getJSONArray("functions")
                if (functions.length() > 0) {
                    report.append("Functions: ")
                    val functionList = mutableListOf<String>()
                    for (i in 0 until functions.length()) {
                        functionList.add(functions.getString(i))
                    }
                    report.append("${functionList.joinToString(", ")}\n")
                }
            }
            report.append("\n")
        }
        
        // Knowledge comparison
        if (!comparison.has("error") && comparison.optString("comparison") == "complete") {
            report.append("KNOWLEDGE ALIGNMENT:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Alignment Level: ${comparison.optString("alignment", "unknown").uppercase()}\n")
            
            if (comparison.has("alignments")) {
                val alignments = comparison.getJSONArray("alignments")
                if (alignments.length() > 0) {
                    report.append("Aligned Aspects: ")
                    val alignmentList = mutableListOf<String>()
                    for (i in 0 until alignments.length()) {
                        alignmentList.add(alignments.getString(i))
                    }
                    report.append("${alignmentList.joinToString(", ")}\n")
                }
            }
            
            if (comparison.has("differences")) {
                val differences = comparison.getJSONArray("differences")
                if (differences.length() > 0) {
                    report.append("Differences: ")
                    val differenceList = mutableListOf<String>()
                    for (i in 0 until differences.length()) {
                        differenceList.add(differences.getString(i))
                    }
                    report.append("${differenceList.joinToString(", ")}\n")
                }
            }
            
            if (comparison.has("detailed_analysis")) {
                val analysis = comparison.getJSONObject("detailed_analysis")
                val coverage = analysis.optDouble("conceptual_coverage", 0.0)
                val completeness = analysis.optDouble("implementation_completeness", 0.0)
                
                report.append("Conceptual Coverage: ${String.format("%.1f%%", coverage * 100)}\n")
                report.append("Implementation Completeness: ${String.format("%.1f%%", completeness * 100)}\n")
                
                if (analysis.has("recommendations")) {
                    val recommendations = analysis.getJSONArray("recommendations")
                    if (recommendations.length() > 0) {
                        report.append("\nRecommendations:\n")
                        for (i in 0 until recommendations.length()) {
                            report.append("• ${recommendations.getString(i)}\n")
                        }
                    }
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildImplementationQueryReport(
        implInfo: JSONObject,
        concepts: JSONArray,
        hierarchy: JSONObject?,
        implementationName: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔧 IMPLEMENTATION QUERY RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Implementation: $implementationName\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (implInfo.has("error")) {
            report.append("❌ IMPLEMENTATION NOT FOUND\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Error: ${implInfo.optString("error_message", "Unknown error")}\n\n")
            report.append("Suggestions:\n")
            report.append("• Check implementation name spelling\n")
            report.append("• Try with full module path\n")
            report.append("• Use format 'ClassName.methodName' for methods\n")
            return report.toString()
        }
        
        // Basic implementation info
        report.append("IMPLEMENTATION DETAILS:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("Type: ${implInfo.optString("type", "unknown").uppercase()}\n")
        
        if (implInfo.has("details")) {
            val details = implInfo.getJSONObject("details")
            
            report.append("Module: ${details.optString("module", "unknown")}\n")
            
            if (details.has("line_number")) {
                val lineNumber = details.optInt("line_number", 0)
                val endLineNumber = details.optInt("end_line_number", 0)
                if (lineNumber > 0) {
                    report.append("Location: Line $lineNumber")
                    if (endLineNumber > 0 && endLineNumber != lineNumber) {
                        report.append("-$endLineNumber")
                    }
                    report.append("\n")
                }
            }
            
            if (details.has("complexity")) {
                report.append("Complexity: ${details.optInt("complexity", 0)}\n")
            }
            
            // Docstring
            val docstring = details.optString("docstring", "")
            if (docstring.isNotEmpty()) {
                report.append("\nDescription:\n")
                if (docstring.length > 200) {
                    report.append("${docstring.take(200)}...\n")
                } else {
                    report.append("$docstring\n")
                }
            }
            
            // Parameters for functions/methods
            if (details.has("parameters")) {
                val parameters = details.getJSONArray("parameters")
                if (parameters.length() > 0) {
                    report.append("\nParameters:\n")
                    for (i in 0 until parameters.length()) {
                        val param = parameters.getJSONObject(i)
                        val name = param.optString("name", "")
                        val type = param.optString("type", "")
                        val default = param.opt("default")
                        
                        report.append("• $name")
                        if (type.isNotEmpty()) report.append(": $type")
                        if (default != null && default.toString() != "null") {
                            report.append(" = $default")
                        }
                        report.append("\n")
                    }
                }
            }
            
            // Methods for classes
            if (details.has("methods")) {
                val methods = details.getJSONArray("methods")
                if (methods.length() > 0) {
                    report.append("\nMethods (${methods.length()}):\n")
                    for (i in 0 until minOf(10, methods.length())) {
                        val method = methods.getJSONObject(i)
                        val name = method.optString("name", "")
                        val complexity = method.optInt("complexity", 0)
                        val docstring = method.optString("docstring", "")
                        
                        report.append("• $name (complexity: $complexity)\n")
                        if (docstring.isNotEmpty()) {
                            val summary = docstring.split(".").firstOrNull()?.trim()
                            if (!summary.isNullOrEmpty()) {
                                report.append("  $summary\n")
                            }
                        }
                    }
                    if (methods.length() > 10) {
                        report.append("... and ${methods.length() - 10} more methods\n")
                    }
                }
            }
            
            // Attributes for classes
            if (details.has("attributes")) {
                val attributes = details.getJSONArray("attributes")
                if (attributes.length() > 0) {
                    report.append("\nAttributes:\n")
                    for (i in 0 until attributes.length()) {
                        val attr = attributes.getJSONObject(i)
                        val name = attr.optString("name", "")
                        val value = attr.opt("value")
                        
                        report.append("• $name")
                        if (value != null && value.toString() != "null") {
                          ```kotlin
                            report.append(" = $value")
                        }
                        report.append("\n")
                    }
                }
            }
            
            // Base classes for classes
            if (details.has("base_classes")) {
                val baseClasses = details.getJSONArray("base_classes")
                if (baseClasses.length() > 0) {
                    report.append("\nInherits From: ")
                    val baseClassList = mutableListOf<String>()
                    for (i in 0 until baseClasses.length()) {
                        val baseClass = baseClasses.getString(i)
                        if (baseClass != "null") {
                            baseClassList.add(baseClass)
                        }
                    }
                    report.append("${baseClassList.joinToString(", ")}\n")
                }
            }
        }
        
        // Associated concepts
        if (concepts.length() > 0) {
            report.append("\nASSOCIATED CONCEPTS:\n")
            report.append("-".repeat(30)).append("\n")
            for (i in 0 until concepts.length()) {
                report.append("• ${concepts.getString(i)}\n")
            }
        }
        
        // Class hierarchy
        if (hierarchy != null && !hierarchy.has("error")) {
            report.append("\nCLASS HIERARCHY:\n")
            report.append("-".repeat(30)).append("\n")
            report.append(formatHierarchy(hierarchy, 0))
        }
        
        return report.toString()
    }
    
    private fun buildImplementationExplanationReport(
        explanation: JSONObject,
        implInfo: JSONObject,
        implementationName: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("📖 IMPLEMENTATION EXPLANATION\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Implementation: $implementationName\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (explanation.has("error")) {
            report.append("❌ EXPLANATION NOT AVAILABLE\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Error: ${explanation.optString("error_message", "Unknown error")}\n")
            return report.toString()
        }
        
        val type = explanation.optString("type", "unknown")
        report.append("IMPLEMENTATION TYPE: ${type.uppercase()}\n")
        report.append("-".repeat(30)).append("\n")
        
        // Main explanation
        val explanationText = explanation.optString("explanation", "")
        if (explanationText.isNotEmpty()) {
            report.append("EXPLANATION:\n")
            report.append(explanationText).append("\n\n")
        }
        
        // Associated concepts
        if (explanation.has("concepts")) {
            val concepts = explanation.getJSONArray("concepts")
            if (concepts.length() > 0) {
                report.append("RELATED CONCEPTS:\n")
                report.append("-".repeat(20)).append("\n")
                for (i in 0 until concepts.length()) {
                    report.append("• ${concepts.getString(i)}\n")
                }
                report.append("\n")
            }
        }
        
        // Code details
        if (explanation.has("code")) {
            val code = explanation.getJSONObject("code")
            report.append("CODE STRUCTURE:\n")
            report.append("-".repeat(20)).append("\n")
            
            if (code.has("parameters")) {
                val parameters = code.getJSONArray("parameters")
                if (parameters.length() > 0) {
                    report.append("Parameters: ")
                    val paramList = mutableListOf<String>()
                    for (i in 0 until parameters.length()) {
                        paramList.add(parameters.getString(i))
                    }
                    report.append("${paramList.joinToString(", ")}\n")
                }
            }
            
            if (code.has("return_type")) {
                val returnType = code.optString("return_type", "")
                if (returnType.isNotEmpty() && returnType != "null") {
                    report.append("Return Type: $returnType\n")
                }
            }
            
            if (code.has("complexity")) {
                report.append("Complexity Score: ${code.optInt("complexity", 0)}\n")
            }
            
            if (code.has("line_number")) {
                report.append("Source Location: Line ${code.optInt("line_number", 0)}")
                if (code.has("module")) {
                    report.append(" in ${code.optString("module", "")}")
                }
                report.append("\n")
            }
            
            if (code.has("methods")) {
                val methods = code.getJSONArray("methods")
                if (methods.length() > 0) {
                    report.append("Methods: ")
                    val methodList = mutableListOf<String>()
                    for (i in 0 until minOf(5, methods.length())) {
                        methodList.add(methods.getString(i))
                    }
                    report.append("${methodList.joinToString(", ")}")
                    if (methods.length() > 5) {
                        report.append(" (and ${methods.length() - 5} more)")
                    }
                    report.append("\n")
                }
            }
            
            if (code.has("attributes")) {
                val attributes = code.getJSONArray("attributes")
                if (attributes.length() > 0) {
                    report.append("Attributes: ")
                    val attrList = mutableListOf<String>()
                    for (i in 0 until minOf(5, attributes.length())) {
                        attrList.add(attributes.getString(i))
                    }
                    report.append("${attrList.joinToString(", ")}")
                    if (attributes.length() > 5) {
                        report.append(" (and ${attributes.length() - 5} more)")
                    }
                    report.append("\n")
                }
            }
            
            if (code.has("body")) {
                val body = code.getJSONArray("body")
                if (body.length() > 0) {
                    report.append("\nCODE BODY (${body.length()} statements):\n")
                    report.append("-".repeat(20)).append("\n")
                    for (i in 0 until minOf(5, body.length())) {
                        val statement = body.getString(i)
                        if (statement.length > 80) {
                            report.append("${statement.take(77)}...\n")
                        } else {
                            report.append("$statement\n")
                        }
                    }
                    if (body.length() > 5) {
                        report.append("... and ${body.length() - 5} more statements\n")
                    }
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildConceptSearchReport(
        searchResults: JSONArray,
        implementations: JSONArray,
        searchQuery: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🔎 CONCEPT SEARCH RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Search Query: \"$searchQuery\"\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Search results
        if (searchResults.length() > 0) {
            report.append("MATCHING CONCEPTS (${searchResults.length()}):\n")
            report.append("-".repeat(30)).append("\n")
            for (i in 0 until searchResults.length()) {
                report.append("• ${searchResults.getString(i)}\n")
            }
            report.append("\n")
        } else {
            report.append("❌ NO MATCHING CONCEPTS FOUND\n")
            report.append("-".repeat(30)).append("\n")
            report.append("No concepts found matching \"$searchQuery\"\n\n")
        }
        
        // Implementation results
        if (implementations.length() > 0) {
            report.append("RELATED IMPLEMENTATIONS (${implementations.length()}):\n")
            report.append("-".repeat(30)).append("\n")
            
            for (i in 0 until implementations.length()) {
                val impl = implementations.getJSONObject(i)
                if (!impl.has("error")) {
                    val concept = impl.optString("concept", "Unknown")
                    val implementation = impl.optJSONObject("implementation")
                    
                    report.append("Concept: $concept\n")
                    
                    if (implementation != null) {
                        if (implementation.has("classes")) {
                            val classes = implementation.getJSONArray("classes")
                            if (classes.length() > 0) {
                                report.append("  Classes: ")
                                val classList = mutableListOf<String>()
                                for (j in 0 until classes.length()) {
                                    classList.add(classes.getString(j))
                                }
                                report.append("${classList.joinToString(", ")}\n")
                            }
                        }
                        
                        if (implementation.has("methods")) {
                            val methods = implementation.getJSONArray("methods")
                            if (methods.length() > 0) {
                                report.append("  Methods: ")
                                val methodList = mutableListOf<String>()
                                for (j in 0 until minOf(3, methods.length())) {
                                    methodList.add(methods.getString(j))
                                }
                                report.append("${methodList.joinToString(", ")}")
                                if (methods.length() > 3) {
                                    report.append(" (and ${methods.length() - 3} more)")
                                }
                                report.append("\n")
                            }
                        }
                        
                        if (implementation.has("functions")) {
                            val functions = implementation.getJSONArray("functions")
                            if (functions.length() > 0) {
                                report.append("  Functions: ")
                                val functionList = mutableListOf<String>()
                                for (j in 0 until functions.length()) {
                                    functionList.add(functions.getString(j))
                                }
                                report.append("${functionList.joinToString(", ")}\n")
                            }
                        }
                    }
                    report.append("\n")
                }
            }
        }
        
        // Search suggestions
        if (searchResults.length() == 0 && implementations.length() == 0) {
            report.append("SEARCH SUGGESTIONS:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("• Try partial keywords (e.g., 'boundary' instead of 'BoundaryDetection')\n")
            report.append("• Use common terms like 'detection', 'analysis', 'system'\n")
            report.append("• Check spelling and capitalization\n")
            report.append("• Try broader search terms\n")
        }
        
        return report.toString()
    }
    
    private fun buildCodeAnalysisReport(
        analysisResults: Map<String, JSONObject>,
        input: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("📄 CODE ANALYSIS RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Analysis Target: $input\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (analysisResults.isEmpty()) {
            report.append("❌ NO ANALYSIS RESULTS\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Could not analyze \"$input\"\n\n")
            report.append("Possible reasons:\n")
            report.append("• File path does not exist\n")
            report.append("• Class/function/method not found\n")
            report.append("• Invalid identifier format\n")
            return report.toString()
        }
        
        analysisResults.forEach { (analysisType, result) ->
            if (!result.has("error")) {
                report.append("${analysisType.replace("_", " ").uppercase()}:\n")
                report.append("-".repeat(30)).append("\n")
                
                when (analysisType) {
                    "file_analysis" -> {
                        val name = result.optString("name", "Unknown")
                        val path = result.optString("path", "Unknown")
                        val docstring = result.optString("docstring", "")
                        val linesOfCode = result.optInt("lines_of_code", 0)
                        val fileSize = result.optLong("file_size", 0)
                        
                        report.append("Module: $name\n")
                        report.append("Path: $path\n")
                        report.append("File Size: ${fileSize} bytes\n")
                        report.append("Lines of Code: $linesOfCode\n")
                        
                        if (docstring.isNotEmpty()) {
                            report.append("Description: $docstring\n")
                        }
                        
                        if (result.has("classes")) {
                            val classes = result.getJSONArray("classes")
                            report.append("Classes: ${classes.length()}\n")
                        }
                        
                        if (result.has("functions")) {
                            val functions = result.getJSONArray("functions")
                            report.append("Functions: ${functions.length()}\n")
                        }
                        
                        if (result.has("imports")) {
                            val imports = result.getJSONArray("imports")
                            report.append("Imports: ${imports.length()}\n")
                        }
                    }
                    
                    "class_definition" -> {
                        val name = result.optString("name", "Unknown")
                        val module = result.optString("module", "Unknown")
                        val docstring = result.optString("docstring", "")
                        val lineNumber = result.optInt("line_number", 0)
                        
                        report.append("Class: $name\n")
                        report.append("Module: $module\n")
                        if (lineNumber > 0) report.append("Line: $lineNumber\n")
                        
                        if (docstring.isNotEmpty()) {
                            report.append("Description: ")
                            if (docstring.length > 150) {
                                report.append("${docstring.take(150)}...\n")
                            } else {
                                report.append("$docstring\n")
                            }
                        }
                        
                        if (result.has("base_classes")) {
                            val baseClasses = result.getJSONArray("base_classes")
                            if (baseClasses.length() > 0) {
                                val baseList = mutableListOf<String>()
                                for (i in 0 until baseClasses.length()) {
                                    val base = baseClasses.getString(i)
                                    if (base != "null") baseList.add(base)
                                }
                                if (baseList.isNotEmpty()) {
                                    report.append("Inherits From: ${baseList.joinToString(", ")}\n")
                                }
                            }
                        }
                        
                        if (result.has("methods")) {
                            val methods = result.getJSONArray("methods")
                            report.append("Methods: ${methods.length()}\n")
                        }
                        
                        if (result.has("attributes")) {
                            val attributes = result.getJSONArray("attributes")
                            report.append("Attributes: ${attributes.length()}\n")
                        }
                    }
                    
                    "function_definition", "method_definition" -> {
                        val name = result.optString("name", "Unknown")
                        val module = result.optString("module", "Unknown")
                        val docstring = result.optString("docstring", "")
                        val lineNumber = result.optInt("line_number", 0)
                        val complexity = result.optInt("complexity", 0)
                        
                        if (analysisType == "method_definition") {
                            val className = result.optString("class", "Unknown")
                            report.append("Method: $className.$name\n")
                        } else {
                            report.append("Function: $name\n")
                        }
                        
                        report.append("Module: $module\n")
                        if (lineNumber > 0) report.append("Line: $lineNumber\n")
                        report.append("Complexity: $complexity\n")
                        
                        if (docstring.isNotEmpty()) {
                            report.append("Description: ")
                            if (docstring.length > 150) {
                                report.append("${docstring.take(150)}...\n")
                            } else {
                                report.append("$docstring\n")
                            }
                        }
                        
                        if (result.has("parameters")) {
                            val parameters = result.getJSONArray("parameters")
                            if (parameters.length() > 0) {
                                report.append("Parameters: ${parameters.length()}\n")
                                for (i in 0 until minOf(5, parameters.length())) {
                                    val param = parameters.getJSONObject(i)
                                    val paramName = param.optString("name", "")
                                    val paramType = param.optString("type", "")
                                    report.append("  • $paramName")
                                    if (paramType.isNotEmpty()) report.append(": $paramType")
                                    report.append("\n")
                                }
                                if (parameters.length() > 5) {
                                    report.append("  ... and ${parameters.length() - 5} more\n")
                                }
                            }
                        }
                        
                        if (result.has("return_type")) {
                            val returnType = result.optString("return_type", "")
                            if (returnType.isNotEmpty() && returnType != "null") {
                                report.append("Returns: $returnType\n")
                            }
                        }
                    }
                }
                report.append("\n")
            } else {
                report.append("❌ ${analysisType.replace("_", " ").uppercase()} ERROR:\n")
                report.append("-".repeat(30)).append("\n")
                report.append("${result.optString("error_message", "Unknown error")}\n\n")
            }
        }
        
        return report.toString()
    }
    
    private fun buildMemoryAccessReport(
        runtimeObject: JSONObject,
        objectsByType: JSONArray,
        memoryDiagnostics: JSONObject,
        input: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("💾 MEMORY ACCESS RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Target: $input\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Runtime object details
        if (!runtimeObject.has("error") && runtimeObject.has("id")) {
            report.append("RUNTIME OBJECT FOUND:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("ID: ${runtimeObject.optString("id", "Unknown")}\n")
            report.append("Type: ${runtimeObject.optString("type", "Unknown")}\n")
            
            if (runtimeObject.has("attributes")) {
                val attributes = runtimeObject.getJSONObject("attributes")
                if (attributes.length() > 0) {
                    report.append("Attributes: ${attributes.length()}\n")
                    
                    // Show first few attributes
                    val keys = attributes.keys().asSequence().take(5).toList()
                    keys.forEach { key ->
                        val value = attributes.get(key)
                        report.append("  • $key: ${formatValue(value)}\n")
                    }
                    if (attributes.length() > 5) {
                        report.append("  ... and ${attributes.length() - 5} more attributes\n")
                    }
                }
            }
            
            if (runtimeObject.has("methods")) {
                val methods = runtimeObject.getJSONArray("methods")
                if (methods.length() > 0) {
                    report.append("Available Methods: ${methods.length()}\n")
                    for (i in 0 until minOf(8, methods.length())) {
                        report.append("  • ${methods.getString(i)}\n")
                    }
                    if (methods.length() > 8) {
                        report.append("  ... and ${methods.length() - 8} more methods\n")
                    }
                }
            }
            report.append("\n")
        } else {
            // Try objects by type
            if (objectsByType.length() > 0) {
                report.append("OBJECTS OF TYPE \"$input\" (${objectsByType.length()}):\n")
                report.append("-".repeat(30)).append("\n")
                for (i in 0 until objectsByType.length()) {
                    report.append("• ${objectsByType.getString(i)}\n")
                }
                report.append("\n")
            } else {
                report.append("❌ NO OBJECTS FOUND\n")
                report.append("-".repeat(30)).append("\n")
                report.append("No runtime objects found for \"$input\"\n\n")
            }
        }
        
        // Memory system diagnostics
        if (!memoryDiagnostics.has("error")) {
            report.append("MEMORY SYSTEM STATUS:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Registered Objects: ${memoryDiagnostics.optInt("registered_objects", 0)}\n")
            report.append("Registered Types: ${memoryDiagnostics.optInt("registered_types", 0)}\n")
            report.append("Total Accesses: ${memoryDiagnostics.optInt("total_accesses", 0)}\n")
            report.append("Success Rate: ${String.format("%.1f%%", memoryDiagnostics.optDouble("success_rate", 0.0))}\n")
            
            if (memoryDiagnostics.has("type_distribution")) {
                val typeDistribution = memoryDiagnostics.getJSONObject("type_distribution")
                if (typeDistribution.length() > 0) {
                    report.append("\nObject Types:\n")
                    typeDistribution.keys().forEach { type ->
                        val count = typeDistribution.getInt(type)
                        report.append("  • $type: $count objects\n")
                    }
                }
            }
            
            if (memoryDiagnostics.has("recent_errors")) {
                val recentErrors = memoryDiagnostics.getJSONArray("recent_errors")
                if (recentErrors.length() > 0) {
                    report.append("\nRecent Errors: ${recentErrors.length()}\n")
                    for (i in 0 until minOf(3, recentErrors.length())) {
                        val error = recentErrors.getJSONObject(i)
                        val operation = error.optString("operation", "unknown")
                        val errorMessage = error.optString("error", "unknown")
                        report.append("  • $operation: $errorMessage\n")
                    }
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildDiagnosticsReport(
        diagnosticsResult: JSONObject,
        summary: JSONObject,
        availableTests: JSONArray,
        input: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🏥 SYSTEM DIAGNOSTICS RESULTS\n")
        report.append("═".repeat(50)).append("\n\n")
        if (input.isNotEmpty() && input != "all") {
            report.append("Test: $input\n")
        } else {
            report.append("Test: All Diagnostics\n")
        }
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Summary information
        if (!summary.has("error")) {
            report.append("DIAGNOSTIC SUMMARY:\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Total Tests: ${summary.optInt("total_tests", 0)}\n")
            report.append("Success: ${summary.optInt("success_count", 0)}\n")
            report.append("Warnings: ${summary.optInt("warning_count", 0)}\n")
            report.append("Failures: ${summary.optInt("failure_count", 0)}\n")
            report.append("Success Rate: ${summary.optString("success_rate", "0%")}\n")
            report.append("Overall Health: ${summary.optString("overall_health", "unknown").uppercase()}\n")
            
            if (summary.has("failing_tests")) {
                val failingTests = summary.getJSONArray("failing_tests")
                if (failingTests.length() > 0) {
                    report.append("\nFailing Tests:\n")
                    for (i in 0 until failingTests.length()) {
                        report.append("• ${failingTests.getString(i)}\n")
                    }
                }
            }
            report.append("\n")
        }
        
        // Detailed results (if specific test or component diagnostics)
        if (!diagnosticsResult.has("error")) {
            if (diagnosticsResult.has("test_name")) {
                // Single test result
                report.append("TEST DETAILS:\n")
                report.append("-".repeat(30)).append("\n")
                report.append("Test: ${diagnosticsResult.optString("test_name", "Unknown")}\n")
                report.append("Status: ${getStatusWithIcon(diagnosticsResult.optString("status", "unknown"))}\n")
                report.append("Message: ${diagnosticsResult.optString("message", "No message")}\n")
                
                if (diagnosticsResult.has("details")) {
                    val details = diagnosticsResult.getJSONObject("details")
                    if (details.length() > 0) {
                        report.append("\nDetails:\n")
                        details.keys().forEach { key ->
                            val value = details.get(key)
                            report.append("  • $key: ${formatValue(value)}\n")
                        }
                    }
                }
            } else if (diagnosticsResult.has("component_diagnostics")) {
                // Component diagnostics
                val componentDiagnostics = diagnosticsResult.getJSONObject("component_diagnostics")
                report.append("COMPONENT HEALTH:\n")
                report.append("-".repeat(30)).append("\n")
                
                componentDiagnostics.keys().forEach { component ->
                    val componentData = componentDiagnostics.getJSONObject(component)
                    val hasError = componentData.has("error")
                    
                    report.append("${component.replace("_", " ").uppercase()}: ")
                    report.append("${if (hasError) "❌ ERROR" else "✅ OK"}\n")
                    
                    if (!hasError) {
                        // Show key metrics
                        componentData.keys().forEach { key ->
                            if (key != "recent_errors" && key != "recent_accesses") {
                                val value = componentData.get(key)
                                report.append("  • ${key.replace("_", " ").capitalize()}: ${formatValue(value)}\n")
                            }
                        }
                    } else {
                        report.append("  Error: ${componentData.optString("error_message", "Unknown error")}\n")
                    }
                    report.append("\n")
                }
            }
        }
        
        // Available tests
        if (availableTests.length() > 0) {
            report.append("AVAILABLE DIAGNOSTIC TESTS:\n")
            report.append("-".repeat(30)).append("\n")
            for (i in 0 until availableTests.length()) {
                report.append("• ${availableTests.getString(i)}\n")
            }
            report.append("\nTo run a specific test, enter the test name and click 'Run Diagnostics'\n")
        }
        
        return report.toString()
    }
    
    private fun buildSystemStatusReport(
        systemStatus: JSONObject,
        connectivity: JSONObject,
        performanceMetrics: JSONObject
    ): String {
        val report = StringBuilder()
        
        report.append("📊 SYSTEM STATUS REPORT\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        // Core system status
        if (!systemStatus.has("error")) {
            report.append("CORE SYSTEM:\n")
            report.append("-".repeat(20)).append("\n")
            report.append("System: ${systemStatus.optString("system", "Unknown")}\n")
            report.append("Version: ${systemStatus.optString("version", "Unknown")}\n")
            report.append("Health: ${systemStatus.optString("health", "unknown").uppercase()}\n")
            report.append("Base Path: ${systemStatus.optString("base_path", "Unknown")}\n")
            
            if (systemStatus.has("uptime")) {
                report.append("Uptime: ${systemStatus.optString("uptime", "Unknown")}\n")
            }
            
            if (systemStatus.has("components")) {
                val components = systemStatus.getJSONObject("components")
                report.append("\nCOMPONENTS:\n")
                report.append("-".repeat(20)).append("\n")
                
                components.keys().forEach { component ->
                    val componentData = components.getJSONObject(component)
                    val status = componentData.optString("status", "unknown")
                    report.append("${component.replace("_", " ").uppercase()}: ")
                    report.append("${if (status == "active") "✅" else "❌"} $status\n")
                    
                    // Show key metrics for each component
                    componentData.keys().forEach { key ->
                        if (key != "status") {
                            val value = componentData.get(key)
                            if (value != null && value.toString() != "null") {
                                report.append("  • ${key.replace("_", " ").capitalize()}: $value\n")
                            }
                        }
                    }
                }
            }
            report.append("\n")
        }
        
        // Connectivity status
        if (!connectivity.has("error")) {
            report.append("CONNECTIVITY:\n")
            report.append("-".repeat(20)).append("\n")
            report.append("Overall Status: ${connectivity.optString("overall_status", "unknown").uppercase()}\n")
            report.append("Python: ${getStatusIcon(connectivity.optBoolean("python_connectivity"))}\n")
            report.append("Module: ${getStatusIcon(connectivity.optBoolean("module_accessibility"))}\n")
            report.append("System Instance: ${getStatusIcon(connectivity.optBoolean("system_instance"))}\n")
            
            if (connectivity.has("components")) {
                val components = connectivity.getJSONObject("components")
                report.append("Components:\n")
                components.keys().forEach { component ->
                    val isActive = components.optBoolean(component)
                    report.append("  • ${component.replace("_", " ").capitalize()}: ${getStatusIcon(isActive)}\n")
                }
            }
            
            if (connectivity.has("errors")) {
                val errors = connectivity.getJSONArray("errors")
                if (errors.length() > 0) {
                    report.append("\nConnectivity Errors:\n")
                    for (i in 0 until errors.length()) {
                        report.append("• ${errors.getString(i)}\n")
                    }
                }
            }
            report.append("\n")
        }
        
        // Performance metrics
        if (!performanceMetrics.has("error")) {
            report.append("PERFORMANCE:\n")
            report.append("-".repeat(20)).append("\n")
            
            if (performanceMetrics.has("memory")) {
                val memory = performanceMetrics.getJSONObject("memory")
                val totalMB = memory.optLong("total_memory", 0) / (1024 * 1024)
                val usedMB = memory.optLong("used_memory", 0) / (1024 * 1024)
                val freeMB = memory.optLong("free_memory", 0) / (1024 * 1024)
                val maxMB = memory.optLong("max_memory", 0) / (1024 * 1024)
                
                report.append("Memory Usage:\n")
                report.append("  • Total: ${totalMB}MB\n")
                report.append("  • Used: ${usedMB}MB\n")
                report.append("  • Free: ${freeMB}MB\n")
                report.append("  • Max: ${maxMB}MB\n")
                
                val usagePercent = if (totalMB > 0) (usedMB * 100 / totalMB) else 0
                report.append("  • Usage: ${usagePercent}%\n")
            }
            
            if (performanceMetrics.has("caches")) {
                val caches = performanceMetrics.getJSONObject("caches")
                report.append("\nCache Status:\n")
                caches.keys().forEach { cache ->
                    val value = caches.get(cache)
                    report.append("  • ${cache.replace("_", " ").capitalize()}: $value\n")
                }
            }
        }
        
        // Operation history
        if (operationHistory.isNotEmpty()) {
            report.append("\nRECENT OPERATIONS:\n")
            report.append("-".repeat(20)).append("\n")
            operationHistory.takeLast(8).forEach { operation ->
                report.append("• $operation\n")
            }
            if (operationHistory.size > 8) {
                report.append("... and ${operationHistory.size - 8} earlier operations\n")
            }
        }
        
        return report.toString()
    }
    
    private fun buildComprehensiveAnalysisReport(
        analysis: JSONObject,
        healthReport: JSONObject,
        input: String
    ): String {
        val report = StringBuilder()
        val executionTime = getExecutionTime()
        
        report.append("🌟 COMPREHENSIVE ANALYSIS RESULTS\n")
        report.append("═".repeat(60)).append("\n\n")
        report.append("Target: $input\n")
        report.append("Execution Time: ${executionTime}ms\n")
        report.append("Timestamp: ${getCurrentTimestamp()}\n\n")
        
        if (analysis.has("error")) {
            report.append("❌ ANALYSIS ERROR\n")
            report.append("-".repeat(30)).append("\n")
            report.append("Error: ${analysis.optString("error_message", "Unknown error")}\n")
            return report.toString()
        }
        
        // Analysis type and target
        val analysisType = analysis.optString("analysis_type", "auto")
        report.append("ANALYSIS TYPE: ${analysisType.uppercase()}\n")
        report.append("-".repeat(30)).append("\n")
        
        // Concept analysis results
        if (analysis.has("concept_analysis")) {
            val conceptAnalysis = analysis.getJSONObject("concept_analysis")
            report.append("🔍 CONCEPT ANALYSIS:\n")
            
            if (!conceptAnalysis.has("error")) {
                if (conceptAnalysis.has("diagnostics")) {
                    val diagnostics = conceptAnalysis.getJSONObject("diagnostics")
                    report.append("  • General Knowledge: ${getStatusIcon(diagnostics.optBoolean("has_general_knowledge"))}\n")
                    report.append("  • Implementation: ${getStatusIcon(diagnostics.optBoolean("has_specific_implementation"))}\n")
                    report.append("  • Code Details: ${getStatusIcon(diagnostics.optBoolean("code_details_extracted"))}\n")
                }
            } else {
                report.append("  ❌ ${conceptAnalysis.optString("error_message", "Error")}\n")
            }
            report.append("\n")
        }
        
        // Implementation analysis results
        if (analysis.has("implementation_analysis")) {
            val implAnalysis = analysis.getJSONObject("implementation_analysis")
            report.append("🔧 IMPLEMENTATION ANALYSIS:\n")
            
            if (!implAnalysis.has("error")) {
                val type = implAnalysis.optString("type", "unknown")
                report.append("  • Type: ${type.uppercase()}\n")
                
                if (implAnalysis.has("details")) {
                    val details = implAnalysis.getJSONObject("details")
                    val module = details.optString("module", "unknown")
                    val complexity = details.optInt("complexity", 0)
                    report.append("  • Module: $module\n")
                    if (complexity > 0) {
                        report.append("  • Complexity: $complexity\n")
                    }
                }
                
                if (implAnalysis.has("concepts")) {
                    val concepts = implAnalysis.getJSONArray("concepts")
                    if (concepts.length() > 0) {
                        report.append("  • Associated Concepts: ${concepts.length()}\n")
                    }
                }
            } else {
                report.append("  ❌ ${implAnalysis.optString("error_message", "Error")}\n")
            }
            report.append("\n")
        }
        
        // Knowledge comparison
        if (analysis.has("knowledge")) {
            val knowledge = analysis.getJSONObject("knowledge")
            report.append("🧠 KNOWLEDGE COMPARISON:\n")
            
            val hasGeneral = knowledge.has("general")
            val hasSpecific = knowledge.has("specific")
            
            report.append("  • General Knowledge: ${getStatusIcon(hasGeneral)}\n")
            report.append("  • Specific Implementation: ${getStatusIcon(hasSpecific)}\n")
            
            if (hasGeneral && hasSpecific) {
                report.append("  • Knowledge Alignment: Available\n")
            } else {
                report.append("  • Knowledge Alignment: Incomplete\n")
            }
            report.append("\n")
        }
        
        // Explanation summary
        if (analysis.has("explanation")) {
            val explanation = analysis.getJSONObject("explanation")
            report.append("📖 EXPLANATION SUMMARY:\n")
            
            if (!explanation.has("error")) {
                val explanationText = explanation.optString("explanation", "")
                if (explanationText.isNotEmpty()) {
                    val summary = explanationText.split("\n").firstOrNull()?.trim()
                    if (!summary.isNullOrEmpty()) {
                        if (summary.length > 100) {
                            report.append("  ${summary.take(100)}...\n")
                        } else {
                            report.append("  $summary\n")
                        }
                    }
                }
                
                if (explanation.has("concepts")) {
                    val concepts = explanation.getJSONArray("concepts")
                    if (concepts.length() > 0) {
                        report.append("  • Related Concepts: ${concepts.length()}\n")
                    }
                }
            } else {
                report.append("  ❌ ${explanation.optString("error_message", "Error")}\n")
            }
            report.append("\n")
        }
        
        // System health summary
        if (!healthReport.has("error")) {
            report.append("🏥 SYSTEM HEALTH SUMMARY:\n")
            report.append("-".repeat(30)).append("\n")
            
            if (healthReport.has("system_status")) {
                val systemStatus = healthReport.getJSONObject("system_status")
                val health = systemStatus.optString("health", "unknown")
                report.append("Overall Health: ${health.uppercase()}\n")
            }
            
            if (healthReport.has("diagnostics")) {
                val diagnostics = healthReport.getJSONObject("diagnostics")
                val overallHealth = diagnostics.optString("overall_health", "unknown")
                val successRate = diagnostics.optString("success_rate", "0%")
                val totalTests = diagnostics.optInt("total_tests", 0)
                
                report.append("Diagnostic Health: ${overallHealth.uppercase()}\n")
                report.append("Success Rate: $successRate ($totalTests tests)\n")
            }
            
            if (healthReport.has("component_diagnostics")) {
                val componentDiagnostics = healthReport.getJSONObject("component_diagnostics")
                val healthyComponents = mutableListOf<String>()
                val unhealthyComponents = mutableListOf<String>()
                
                componentDiagnostics.keys().forEach { component ->
                    val componentData = componentDiagnostics.getJSONObject(component)
                    if (componentData.has("error")) {
                        unhealthyComponents.add(component)
                    } else {
                        healthyComponents.add(component)
                    }
                }
                
                report.append("Healthy Components: ${healthyComponents.size}\n")
                if (unhealthyComponents.isNotEmpty()) {
                    report.append("Unhealthy Components: ${unhealthyComponents.size}\n")
                    report.append("Issues: ${unhealthyComponents.joinToString(", ")}\n")
                }
            }
        }
        
        return report.toString()
    }
    
    private fun buildSampleQueriesReport(sampleQueries: JSONArray): String {
        val report = StringBuilder()
        
        report.append("📝 SAMPLE QUERIES FOR TESTING\n")
        report.append("═".repeat(50)).append("\n\n")
        report.append("Generated: ${getCurrentTimestamp()}\n\n")
        
        if (sampleQueries.length() == 0) {
            report.append("❌ NO SAMPLE QUERIES AVAILABLE\n")
            return report.toString()
        }
        
        // Group samples by type
        val samplesByType = mutableMapOf<String, MutableList<JSONObject>>()
        
        for (i in 0 until sampleQueries.length()) {
            val sample = sampleQueries.getJSONObject(i)
            if (!sample.has("error")) {
                val type = sample.optString("type", "unknown")
                if (!samplesByType.containsKey(type)) {
                    samplesByType[type] = mutableListOf()
                }
                samplesByType[type]?.add(sample)
            }
        }
        
        // Display samples by category
        samplesByType.forEach { (type, samples) ->
            report.append("${type.replace("_", " ").uppercase()} SAMPLES:\n")
            report.append("-".repeat(30)).append("\n")
            
            samples.forEach { sample ->
                val target = sample.optString("target", "")
                val method = sample.optString("method", "")
                val description = sample.optString("description", "")
                
                report.append("Sample Query: $target\n")
                report.append("Method: $method\n")
                if (description.isNotEmpty()) {
                    report.append("Description: $description\n")
                }
                report.append("\n")
            }
        }
        
        report.append("USAGE INSTRUCTIONS:\n")
        report.append("-".repeat(30)).append("\n")
        report.append("1. Copy any sample query above\n")
        report.append("2. Paste it into the input field\n")
        report.append("3. Select the appropriate operation chip\n")
        report.append("4. Click the corresponding operation button\n\n")
        
        report.append("💡 The first sample has been loaded into the input field!")
        
        return report.toString()
    }
    
    // ============================================================================
    // UTILITY AND HELPER METHODS
    // ============================================================================
    
    private fun validateSystemAndInput(): Boolean {
        if (!isInitialized) {
            statusText.text = "❌ System not initialized. Please wait for initialization to complete."
            return false
        }
        
        val inputContent = inputText.text.toString().trim()
        if (inputContent.isEmpty()) {
            statusText.text = "⚠️ Please enter a concept name, implementation identifier, or file path to analyze."
            Toast.makeText(this, "Input required", Toast.LENGTH_SHORT).show()
            return false
        }
        
        return true
    }
    
    private fun highlightRelevantOperations(operationType: String) {
        // Reset all buttons to normal state
        resetButtonHighlights()
        
        // Highlight relevant buttons based on operation type
        when (operationType) {
            "concepts" -> {
                queryConceptButton.setBackgroundColor(getColor(R.color.highlight_color))
                searchConceptsButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "implementation" -> {
                queryImplementationButton.setBackgroundColor(getColor(R.color.highlight_color))
                explainImplementationButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "code" -> {
                analyzeCodeButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "memory" -> {
                memoryAccessButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "diagnostics" -> {
                runDiagnosticsButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
            "system" -> {
                systemStatusButton.setBackgroundColor(getColor(R.color.highlight_color))
                comprehensiveAnalysisButton.setBackgroundColor(getColor(R.color.highlight_color))
            }
        }
    }
    
    private fun resetButtonHighlights() {
        val buttons = listOf(
            queryConceptButton, queryImplementationButton, explainImplementationButton,
            searchConceptsButton, analyzeCodeButton, memoryAccessButton,
            runDiagnosticsButton, systemStatusButton, comprehensiveAnalysisButton
        )
        
        buttons.forEach { button ->
            button.setBackgroundColor(getColor(R.color.button_normal))
        }
    }
    
    private fun startOperation(operationName: String) {
        operationStartTime = System.currentTimeMillis()
        recordOperation("Started: $operationName")
    }
    
    private fun getExecutionTime(): Long {
        return System.currentTimeMillis() - operationStartTime
    }
    
    private fun getCurrentTimestamp(): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
    }
    
    private fun recordOperation(operation: String) {
        val timestampedOperation = "${getCurrentTimestamp()}: $operation"
        operationHistory.add(timestampedOperation)
        
        // Keep only last 25 operations
        if (operationHistory.size > 25) {
            operationHistory.removeAt(0)
        }
    }
    
    private fun handleOperationError(operationName: String, exception: Exception) {
        val errorMessage = """
            ❌ ERROR IN $operationName
            
            Error: ${exception.message}
            
            Execution Time: ${getExecutionTime()}ms
            Timestamp: ${getCurrentTimestamp()}
            
            Stack Trace:
            ${exception.stackTraceToString().take(800)}...
            
            Please check input and try again.
        """.trimIndent()
        
        statusText.text = errorMessage
        recordOperation("ERROR: $operationName - ${exception.message}")
        exception.printStackTrace()
    }
    
    private fun enableButtons(enabled: Boolean) {
        queryConceptButton.isEnabled = enabled
        queryImplementationButton.isEnabled = enabled
        explainImplementationButton.isEnabled = enabled
        searchConceptsButton.isEnabled = enabled
        analyzeCodeButton.isEnabled = enabled
        memoryAccessButton.isEnabled = enabled
        runDiagnosticsButton.isEnabled = enabled
        systemStatusButton.isEnabled = enabled
        comprehensiveAnalysisButton.isEnabled = enabled
        generateSamplesButton.isEnabled = enabled
    }
    
    private fun showLoading(show: Boolean) {
        loader.visibility = if (show) View.VISIBLE else View.GONE
    }
    
    // Formatting helper methods
    private fun getStatusIcon(status: Boolean): String {
        return if (status) "✅" else "❌"
    }
    
    private fun getStatusWithIcon(status: String): String {
        return when (status.lowercase()) {
            "success" -> "✅ SUCCESS"
            "warning" -> "⚠️ WARNING"
            "failure", "error" -> "❌ FAILURE"
            else -> "⚪ $status"
        }
    }
    
    private fun formatValue(value: Any?): String {
        return when (value) {
            null -> "null"
            is String -> if (value.length > 50) "${value.take(50)}..." else value
            is JSONObject -> if (value.length() > 0) "{${value.length()} keys}" else "{}"
            is JSONArray -> if (value.length() > 0) "[${value.length()} items]" else "[]"
            is Number -> {
                if (value is Double || value is Float) {
                    String.format("%.3f", value.toDouble())
                } else {
                    value.toString()
                }
            }
            else -> value.toString()
        }
    }
    
    private fun formatHierarchy(hierarchy: JSONObject, depth: Int): String {
        val indent = "  ".repeat(depth)
        val name = hierarchy.optString("name", "Unknown")
        val result = StringBuilder("$indent• $name\n")
        
        if (hierarchy.has("base_classes")) {
            val baseClasses = hierarchy.getJSONArray("base_classes")
            for (i in 0 until baseClasses.length()) {
                val baseClass = baseClasses.getJSONObject(i)
                result.append(formatHierarchy(baseClass, depth + 1))
            }
        }
        
        return result.toString()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clear any cached data
        try {
            systemIntrospectionBridge.clearSystemCaches()
        } catch (e: Exception) {
            android.util.Log.e("MainActivitySystemIntrospection", "Error clearing cache on destroy: ${e.message}")
        }
    }
    
    override fun onPause() {
        super.onPause()
        // Save current state if needed
        recordOperation("Activity paused")
    }
    
    override fun onResume() {
        super.onResume()
        // Restore system state if needed
        if (isInitialized) {
            lifecycleScope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        // Quick system health check
                        val connectivity = systemIntrospectionBridge.validateSystemConnectivity()
                        val isHealthy = connectivity.optString("overall_status", "") == "healthy"
                        
                        if (!isHealthy) {
                            // System may have been disrupted, reinitialize
                            recordOperation("System health check failed, reinitializing...")
                            initializeIntrospectionSystem()
                        } else {
                            recordOperation("Activity resumed - system healthy")
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.w("MainActivitySystemIntrospection", "System health check failed: ${e.message}")
                    recordOperation("System health check error: ${e.message}")
                }
            }
        }
    }
    
    override fun onBackPressed() {
        // Clear status text when going back
        if (statusText.text.toString().contains("ERROR") || statusText.text.toString().contains("❌")) {
            statusText.text = "System ready for introspection operations..."
        } else {
            super.onBackPressed()
        }
    }
    
    // Menu handling for additional features
    override fun onCreateOptionsMenu(menu: android.view.Menu?): Boolean {
        menuInflater.inflate(R.menu.introspection_menu, menu)
        return true
    }
    
    override fun onOptionsItemSelected(item: android.view.MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_clear_cache -> {
                clearSystemCache()
                true
            }
            R.id.action_export_config -> {
                exportSystemConfiguration()
                true
            }
            R.id.action_validate_connectivity -> {
                validateConnectivity()
                true
            }
            R.id.action_performance_metrics -> {
                showPerformanceMetrics()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
    
    private fun clearSystemCache() {
        lifecycleScope.launch {
            try {
                showLoading(true)
                val result = withContext(Dispatchers.IO) {
                    systemIntrospectionBridge.clearSystemCaches()
                }
                
                if (!result.has("error")) {
                    statusText.text = "✅ System caches cleared successfully\n\n" +
                            "Cleared: ${result.optJSONArray("cleared_caches")?.let { array ->
                                (0 until array.length()).map { array.getString(it) }.joinToString(", ")
                            } ?: "Unknown"}\n\n" +
                            "Timestamp: ${getCurrentTimestamp()}"
                    Toast.makeText(this@MainActivitySystemIntrospection, "Caches cleared", Toast.LENGTH_SHORT).show()
                } else {
                    statusText.text = "❌ Cache clear error: ${result.optString("error_message", "Unknown error")}"
                }
                recordOperation("System caches cleared")
            } catch (e: Exception) {
                handleOperationError("Clear Cache", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun exportSystemConfiguration() {
        lifecycleScope.launch {
            try {
                showLoading(true)
                val result = withContext(Dispatchers.IO) {
                    systemIntrospectionBridge.exportSystemConfiguration()
                }
                
                if (!result.has("error")) {
                    statusText.text = "📤 SYSTEM CONFIGURATION EXPORTED\n\n" +
                            "Export Version: ${result.optString("export_version", "Unknown")}\n" +
                            "Export Time: ${result.optString("export_timestamp", "Unknown")}\n\n" +
                            "Configuration includes:\n" +
                            "• System information\n" +
                            "• Component states\n" +
                            "• Performance metrics\n" +
                            "• Diagnostic data\n\n" +
                            "Note: Configuration exported to internal storage"
                } else {
                    statusText.text = "❌ Export error: ${result.optString("error_message", "Unknown error")}"
                }
                recordOperation("System configuration exported")
            } catch (e: Exception) {
                handleOperationError("Export Configuration", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun validateConnectivity() {
        lifecycleScope.launch {
            try {
                showLoading(true)
                statusText.text = "🔍 Validating system connectivity..."
                
                val result = withContext(Dispatchers.IO) {
                    systemIntrospectionBridge.validateSystemConnectivity()
                }
                
                val connectivityReport = StringBuilder()
                connectivityReport.append("🔗 CONNECTIVITY VALIDATION RESULTS\n")
                connectivityReport.append("═".repeat(50)).append("\n\n")
                connectivityReport.append("Timestamp: ${getCurrentTimestamp()}\n\n")
                
                val overallStatus = result.optString("overall_status", "unknown")
                connectivityReport.append("Overall Status: ${overallStatus.uppercase()}\n\n")
                
                connectivityReport.append("Component Connectivity:\n")
                connectivityReport.append("-".repeat(30)).append("\n")
                connectivityReport.append("Python: ${getStatusIcon(result.optBoolean("python_connectivity"))}\n")
                connectivityReport.append("Module: ${getStatusIcon(result.optBoolean("module_accessibility"))}\n")
                connectivityReport.append("System: ${getStatusIcon(result.optBoolean("system_instance"))}\n")
                
                if (result.has("components")) {
                    val components = result.getJSONObject("components")
                    components.keys().forEach { component ->
                        val isActive = components.optBoolean(component)
                        connectivityReport.append("${component.replace("_", " ").capitalize()}: ${getStatusIcon(isActive)}\n")
                    }
                }
                
                if (result.has("errors")) {
                    val errors = result.getJSONArray("errors")
                    if (errors.length() > 0) {
                        connectivityReport.append("\nErrors Found:\n")
                        connectivityReport.append("-".repeat(20)).append("\n")
                        for (i in 0 until errors.length()) {
                            connectivityReport.append("• ${errors.getString(i)}\n")
                        }
                    }
                }
                
                if (result.has("warnings")) {
                    val warnings = result.getJSONArray("warnings")
                    if (warnings.length() > 0) {
                        connectivityReport.append("\nWarnings:\n")
                        connectivityReport.append("-".repeat(20)).append("\n")
                        for (i in 0 until warnings.length()) {
                            connectivityReport.append("• ${warnings.getString(i)}\n")
                        }
                    }
                }
                
                statusText.text = connectivityReport.toString()
                recordOperation("Connectivity validation completed")
            } catch (e: Exception) {
                handleOperationError("Connectivity Validation", e)
            } finally {
                showLoading(false)
            }
        }
    }
    
    private fun showPerformanceMetrics() {
        lifecycleScope.launch {
            try {
                showLoading(true)
                statusText.text = "📊 Gathering performance metrics..."
                
                val result = withContext(Dispatchers.IO) {
                    systemIntrospectionBridge.getPerformanceMetrics()
                }
                
                val metricsReport = StringBuilder()
                metricsReport.append("📊 PERFORMANCE METRICS\n")
                metricsReport.append("═".repeat(50)).append("\n\n")
                metricsReport.append("Timestamp: ${getCurrentTimestamp()}\n\n")
                
                if (result.has("memory")) {
                    val memory = result.getJSONObject("memory")
                    val totalMB = memory.optLong("total_memory", 0) / (1024 * 1024)
                    val usedMB = memory.optLong("used_memory", 0) / (1024 * 1024)
                    val freeMB = memory.optLong("free_memory", 0) / (1024 * 1024)
                    val maxMB = memory.optLong("max_memory", 0) / (1024 * 1024)
                    
                    metricsReport.append("MEMORY USAGE:\n")
                    metricsReport.append("-".repeat(20)).append("\n")
                    metricsReport.append("Total: ${totalMB}MB\n")
                    metricsReport.append("Used: ${usedMB}MB\n")
                    metricsReport.append("Free: ${freeMB}MB\n")
                    metricsReport.append("Max: ${maxMB}MB\n")
                    
                    val usagePercent = if (totalMB > 0) (usedMB * 100 / totalMB) else 0
                    metricsReport.append("Usage: ${usagePercent}%\n\n")
                }
                
                if (result.has("caches")) {
                    val caches = result.getJSONObject("caches")
                    metricsReport.append("CACHE METRICS:\n")
                    metricsReport.append("-".repeat(20)).append("\n")
                    caches.keys().forEach { cache ->
                        val value = caches.get(cache)
                        metricsReport.append("${cache.replace("_", " ").capitalize()}: $value\n")
                    }
                }
                
                statusText.text = metricsReport.toString()
                recordOperation("Performance metrics displayed")
            } catch (e: Exception) {
                handleOperationError("Performance Metrics", e)
            } finally {
                showLoading(false)
            }
        }
    }
}
```

This complete MainActivitySystemIntrospection.kt provides:

1. **Comprehensive UI Operations**: All major introspection functions accessible via buttons
2. **Detailed Reporting**: Rich, formatted reports for each operation type
3. **Error Handling**: Robust error handling with detailed error messages
4. **Performance Monitoring**: Built-in performance metrics and timing
5. **System Health**: Connectivity validation and health checks
6. **Cache Management**: Cache clearing and performance optimization
7. **Sample Generation**: Built-in sample queries for testing
8. **Operation History**: Tracking of all operations for debugging
9. **Menu Actions**: Additional utility functions via options menu
10. **Lifecycle Management**: Proper state management across activity lifecycle

The activity follows the same patterns as your arcane knowledge example while providing comprehensive access to all system introspection capabilities! 😀
