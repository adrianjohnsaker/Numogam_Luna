package com.antonio.my.ai.girlfriend.free.amelia.assemblage

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

class MainActivityAssemblage : AppCompatActivity() {
    
    private lateinit var executor: AssemblageExecutor
    
    // UI Components
    private lateinit var inputField: EditText
    private lateinit var executeButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var resultsContainer: LinearLayout
    
    // Results Display
    private lateinit var creativityValue: TextView
    private lateinit var emergenceLevel: TextView
    private lateinit var executionTime: TextView
    private lateinit var modulesList: TextView
    private lateinit var connectionsCount: TextView
    
    // Summary Views
    private lateinit var totalExecutions: TextView
    private lateinit var averageCreativity: TextView
    private lateinit var emergentEvents: TextView
    private lateinit var successRate: TextView
    
    // Benchmark Views
    private lateinit var benchmarkButton: Button
    private lateinit var performanceGrade: TextView
    
    // History
    private lateinit var historyRecyclerView: RecyclerView
    private lateinit var historyAdapter: AssemblageHistoryAdapter
    private val assemblagHistory = mutableListOf<AssemblageExecutor.AssemblageResult>()
    
    companion object {
        private const val TAG = "MainActivityAssemblage"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_assemblage)
        
        initializeViews()
        setupRecyclerView()
        initializeExecutor()
    }
    
    private fun initializeViews() {
        // Input controls
        inputField = findViewById(R.id.input_field)
        executeButton = findViewById(R.id.execute_button)
        progressBar = findViewById(R.id.progress_bar)
        statusText = findViewById(R.id.status_text)
        resultsContainer = findViewById(R.id.results_container)
        
        // Result displays
        creativityValue = findViewById(R.id.creativity_value)
        emergenceLevel = findViewById(R.id.emergence_level)
        executionTime = findViewById(R.id.execution_time)
        modulesList = findViewById(R.id.modules_list)
        connectionsCount = findViewById(R.id.connections_count)
        
        // Summary displays
        totalExecutions = findViewById(R.id.total_executions)
        averageCreativity = findViewById(R.id.average_creativity)
        emergentEvents = findViewById(R.id.emergent_events)
        successRate = findViewById(R.id.success_rate)
        
        // Benchmark
        benchmarkButton = findViewById(R.id.benchmark_button)
        performanceGrade = findViewById(R.id.performance_grade)
        
        // History
        historyRecyclerView = findViewById(R.id.history_recycler_view)
        
        setupEventListeners()
    }
    
    private fun setupEventListeners() {
        executeButton.setOnClickListener { executeAssemblage() }
        benchmarkButton.setOnClickListener { runBenchmark() }
        
        // Add some example prompts
        val examplePrompts = listOf(
            "Create a complex narrative using memory and consciousness",
            "Generate creative solutions using recursive reflection",
            "Explore ontological drift through poetic becoming",
            "Synthesize desire and temporality in creative assemblage",
            "Analyze hyperstitional loops in zone navigation"
        )
        
        findViewById<Button>(R.id.example_button)?.setOnClickListener {
            inputField.setText(examplePrompts.random())
        }
    }
    
    private fun setupRecyclerView() {
        historyAdapter = AssemblageHistoryAdapter(
            assemblages = assemblagHistory,
            onItemClick = { assemblage -> showAssemblageDetails(assemblage) }
        )
        
        historyRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivityAssemblage)
            adapter = historyAdapter
        }
    }
    
    private fun initializeExecutor() {
        executor = AssemblageExecutor.getInstance()
        
        updateStatus("Initializing assemblage executor...")
        showProgress(true)
        
        lifecycleScope.launch {
            try {
                val success = executor.initialize()
                if (success) {
                    updateStatus("Assemblage executor ready")
                    executeButton.isEnabled = true
                    benchmarkButton.isEnabled = true
                    updateSummary()
                } else {
                    updateStatus("Failed to initialize executor")
                    showError("Failed to initialize assemblage system")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Initialization error", e)
                updateStatus("Initialization error: ${e.message}")
                showError("Initialization failed: ${e.localizedMessage}")
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun executeAssemblage() {
        val input = inputField.text.toString().trim()
        if (input.isEmpty()) {
            showError("Please enter an assemblage prompt")
            return
        }
        
        updateStatus("Executing assemblage...")
        showProgress(true)
        executeButton.isEnabled = false
        resultsContainer.visibility = View.GONE
        
        lifecycleScope.launch {
            try {
                val result = executor.executeAssemblage(input)
                
                if (result != null) {
                    displayResult(result)
                    assemblagHistory.add(0, result)
                    historyAdapter.notifyItemInserted(0)
                    updateSummary()
                    updateStatus("Assemblage execution completed")
                } else {
                    showError("Failed to execute assemblage")
                    updateStatus("Execution failed")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Execution error", e)
                showError("Execution error: ${e.localizedMessage}")
                updateStatus("Error: ${e.message}")
            } finally {
                showProgress(false)
                executeButton.isEnabled = true
            }
        }
    }
    
    private fun runBenchmark() {
        updateStatus("Running performance benchmark...")
        showProgress(true)
        benchmarkButton.isEnabled = false
        
        lifecycleScope.launch {
            try {
                val benchmark = executor.runBenchmark(3)
                
                if (benchmark != null) {
                    performanceGrade.text = "Performance: ${benchmark.performanceGrade}"
                    updateStatus("Benchmark completed: ${benchmark.performanceGrade}")
                    
                    // Add benchmark results to history
                    benchmark.results.forEach { resultMap ->
                        // Note: This is simplified - you'd need to convert the map back to AssemblageResult
                        // For demonstration purposes, we'll just update the summary
                    }
                    
                    updateSummary()
                } else {
                    showError("Failed to run benchmark")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark error", e)
                showError("Benchmark error: ${e.localizedMessage}")
            } finally {
                showProgress(false)
                benchmarkButton.isEnabled = true
            }
        }
    }
    
    private fun displayResult(result: AssemblageExecutor.AssemblageResult) {
        creativityValue.text = String.format("%.3f", result.creativeValue)
        emergenceLevel.text = String.format("%.3f", result.emergenceLevel)
        executionTime.text = String.format("%.2f s", result.executionTime)
        connectionsCount.text = result.connectionCount.toString()
        
        // Display modules used
        val modulesText = result.modulesUsed.joinToString(", ") { module ->
            module.substringAfterLast("_").capitalize()
        }
        modulesList.text = modulesText
        
        // Set creativity color coding
        val creativityColor = when {
            result.creativeValue >= 0.8 -> getColor(R.color.high_creativity)
            result.creativeValue >= 0.6 -> getColor(R.color.medium_creativity)
            else -> getColor(R.color.low_creativity)
        }
        creativityValue.setTextColor(creativityColor)
        
        // Set emergence color coding
        val emergenceColor = when {
            result.emergenceLevel >= 0.8 -> getColor(R.color.high_emergence)
            result.emergenceLevel >= 0.5 -> getColor(R.color.medium_emergence)
            else -> getColor(R.color.low_emergence)
        }
        emergenceLevel.setTextColor(emergenceColor)
        
        resultsContainer.visibility = View.VISIBLE
    }
    
    private fun updateSummary() {
        lifecycleScope.launch {
            try {
                val summary = executor.getExecutionSummary()
                
                if (summary != null) {
                    totalExecutions.text = summary.totalExecutions.toString()
                    averageCreativity.text = String.format("%.3f", summary.averageCreativeValue)
                    emergentEvents.text = summary.emergentEvents.toString()
                    successRate.text = String.format("%.1f%%", summary.recentSuccessRate * 100)
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to update summary", e)
            }
        }
    }
    
    private fun showAssemblageDetails(assemblage: AssemblageExecutor.AssemblageResult) {
        val dialog = AssemblageDetailDialog(this, assemblage) { assemblageId ->
            // Handle optimization suggestions
            lifecycleScope.launch {
                try {
                    val suggestions = executor.suggestOptimization(assemblageId)
                    if (suggestions != null) {
                        showOptimizationDialog(suggestions)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to get optimization suggestions", e)
                }
            }
        }
        dialog.show()
    }
    
    private fun showOptimizationDialog(suggestions: Map<String, Any>) {
        val dialog = OptimizationSuggestionsDialog(this, suggestions)
        dialog.show()
    }
    
    private fun showProgress(show: Boolean) {
        progressBar.visibility = if (show) View.VISIBLE else View.GONE
    }
    
    private fun updateStatus(status: String) {
        statusText.text = status
        Log.d(TAG, status)
    }
    
    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        Log.e(TAG, message)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        executor.cleanup()
    }
}

/**
 * RecyclerView adapter for assemblage history
 */
class AssemblageHistoryAdapter(
    private val assemblages: List<AssemblageExecutor.AssemblageResult>,
    private val onItemClick: (AssemblageExecutor.AssemblageResult) -> Unit
) : RecyclerView.Adapter<AssemblageHistoryAdapter.ViewHolder>() {
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val timestamp: TextView = view.findViewById(R.id.timestamp)
        val creativityValue: TextView = view.findViewById(R.id.creativity_value)
        val emergenceLevel: TextView = view.findViewById(R.id.emergence_level)
        val moduleCount: TextView = view.findViewById(R.id.module_count)
        val stateIndicator: View = view.findViewById(R.id.state_indicator)
    }
    
    override fun onCreateViewHolder(parent: android.view.ViewGroup, viewType: Int): ViewHolder {
        val view = android.view.LayoutInflater.from(parent.context)
            .inflate(R.layout.item_assemblage_history, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val assemblage = assemblages[position]
        
        // Format timestamp
        val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val timestamp = try {
            val date = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
                .parse(assemblage.timestamp.substringBefore("."))
            dateFormat.format(date ?: Date())
        } catch (e: Exception) {
            assemblage.timestamp.substringBefore("T")
        }
        
        holder.timestamp.text = timestamp
        holder.creativityValue.text = String.format("C: %.2f", assemblage.creativeValue)
        holder.emergenceLevel.text = String.format("E: %.2f", assemblage.emergenceLevel)
        holder.moduleCount.text = "${assemblage.modulesUsed.size} modules"
        
        // State indicator color
        val stateColor = when (assemblage.state) {
            "emergent" -> holder.itemView.context.getColor(R.color.emergent_state)
            "completed" -> holder.itemView.context.getColor(R.color.completed_state)
            "failed" -> holder.itemView.context.getColor(R.color.failed_state)
            else -> holder.itemView.context.getColor(R.color.default_state)
        }
        holder.stateIndicator.setBackgroundColor(stateColor)
        
        holder.itemView.setOnClickListener { onItemClick(assemblage) }
    }
    
    override fun getItemCount() = assemblages.size
}

/**
 * Dialog for showing detailed assemblage information
 */
class AssemblageDetailDialog(
    private val context: android.content.Context,
    private val assemblage: AssemblageExecutor.AssemblageResult,
    private val onOptimizeClick: (String) -> Unit
) {
    
    fun show() {
        val dialog = androidx.appcompat.app.AlertDialog.Builder(context)
            .setTitle("Assemblage Details")
            .setView(createDetailView())
            .setPositiveButton("Optimize") { _, _ -> onOptimizeClick(assemblage.assemblageId) }
            .setNegativeButton("Close", null)
            .create()
        
        dialog.show()
    }
    
    private fun createDetailView(): View {
        val layout = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(32, 16, 32, 16)
        }
        
        // Add detail items
        addDetailItem(layout,"Assemblage ID", assemblage.assemblageId)
    addDetailItem(layout, "Creative Value", String.format("%.3f", assemblage.creativeValue))
    addDetailItem(layout, "Emergence Level", String.format("%.3f", assemblage.emergenceLevel))
    addDetailItem(layout, "Execution Time", String.format("%.2f seconds", assemblage.executionTime))
    addDetailItem(layout, "Connections", assemblage.connectionCount.toString())
    addDetailItem(layout, "State", assemblage.state.capitalize())
    addDetailItem(layout, "Timestamp", assemblage.timestamp)
    
    // Modules used
    val modulesText = assemblage.modulesUsed.joinToString("\n") { "• $it" }
    addDetailItem(layout, "Modules Used", modulesText)
    
    // Emergent properties
    val propertiesText = assemblage.emergentProperties.entries.joinToString("\n") { (key, value) ->
        "• ${key.replace("_", " ").capitalize()}: $value"
    }
    addDetailItem(layout, "Emergent Properties", propertiesText)
    
    return android.widget.ScrollView(context).apply {
        addView(layout)
    }
}

private fun addDetailItem(layout: android.widget.LinearLayout, label: String, value: String) {
    val labelView = android.widget.TextView(context).apply {
        text = "$label:"
        textSize = 14f
        setTypeface(null, android.graphics.Typeface.BOLD)
        setPadding(0, 8, 0, 4)
    }
    
    val valueView = android.widget.TextView(context).apply {
        text = value
        textSize = 12f
        setPadding(16, 0, 0, 8)
    }
    
    layout.addView(labelView)
    layout.addView(valueView)
}
}

/**
 * Dialog for showing optimization suggestions
 */
class OptimizationSuggestionsDialog(
    private val context: android.content.Context,
    private val suggestions: Map<String, Any>
) {
    
    fun show() {
        val dialog = androidx.appcompat.app.AlertDialog.Builder(context)
            .setTitle("Optimization Suggestions")
            .setView(createSuggestionsView())
            .setPositiveButton("Close", null)
            .create()
        
        dialog.show()
    }
    
    private fun createSuggestionsView(): View {
        val layout = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(32, 16, 32, 16)
        }
        
        // Current metrics
        val currentMetrics = suggestions["current_metrics"] as? Map<String, Any>
        if (currentMetrics != null) {
            val metricsHeader = android.widget.TextView(context).apply {
                text = "Current Performance"
                textSize = 16f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 0, 0, 8)
            }
            layout.addView(metricsHeader)
            
            currentMetrics.forEach { (key, value) ->
                val metricView = android.widget.TextView(context).apply {
                    text = "• ${key.replace("_", " ").capitalize()}: $value"
                    textSize = 12f
                    setPadding(16, 0, 0, 4)
                }
                layout.addView(metricView)
            }
        }
        
        // Optimization suggestions
        val optimizationSuggestions = suggestions["optimization_suggestions"] as? List<String>
        if (optimizationSuggestions != null && optimizationSuggestions.isNotEmpty()) {
            val suggestionsHeader = android.widget.TextView(context).apply {
                text = "Suggestions"
                textSize = 16f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 16, 0, 8)
            }
            layout.addView(suggestionsHeader)
            
            optimizationSuggestions.forEach { suggestion ->
                val suggestionView = android.widget.TextView(context).apply {
                    text = "• $suggestion"
                    textSize = 12f
                    setPadding(16, 0, 0, 4)
                }
                layout.addView(suggestionView)
            }
        }
        
        // Recommended additions
        val recommendedAdditions = suggestions["recommended_additions"] as? List<String>
        if (recommendedAdditions != null && recommendedAdditions.isNotEmpty()) {
            val additionsHeader = android.widget.TextView(context).apply {
                text = "Recommended Module Additions"
                textSize = 16f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 16, 0, 8)
            }
            layout.addView(additionsHeader)
            
            recommendedAdditions.forEach { module ->
                val moduleView = android.widget.TextView(context).apply {
                    text = "• $module"
                    textSize = 12f
                    setPadding(16, 0, 0, 4)
                }
                layout.addView(moduleView)
            }
        }
        
        // Performance potential
        val performancePotential = suggestions["performance_potential"]
        if (performancePotential != null) {
            val potentialView = android.widget.TextView(context).apply {
                text = "Performance Potential: $performancePotential"
                textSize = 14f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 16, 0, 8)
                setTextColor(context.getColor(R.color.accent_color))
            }
            layout.addView(potentialView)
        }
        
        return android.widget.ScrollView(context).apply {
            addView(layout)
        }
    }
}
