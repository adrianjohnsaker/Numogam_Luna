
```kotlin
package com.antonio.my.ai.girlfriend.free.adaptive.systemarchitect

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import com.adaptive.systemarchitect.bridge.AdaptiveSystemBridge
import com.adaptive.systemarchitect.databinding.ActivityMainAdaptiveBinding
import com.adaptive.systemarchitect.model.ScenarioSummary
import kotlinx.coroutines.launch

class MainActivityAdaptive : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivityAdaptive"
        private const val DEFAULT_SYSTEM_NAME = "Adaptive Analysis System"
    }

    private lateinit var binding: ActivityMainAdaptiveBinding
    private val adaptiveBridge = AdaptiveSystemBridge()
    private var selectedScenarioId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainAdaptiveBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupListeners()
        initializeSystem()
    }

    private fun setupListeners() {
        // Initialize button
        binding.btnInitialize.setOnClickListener {
            val systemName = binding.etSystemName.text.toString().takeIf { it.isNotBlank() }
                ?: DEFAULT_SYSTEM_NAME
            initializeSystem(systemName)
        }

        // Create scenario button
        binding.btnCreateScenario.setOnClickListener {
            showCreateScenarioDialog()
        }

        // Analyze scenario button
        binding.btnAnalyzeScenario.setOnClickListener {
            selectedScenarioId?.let { scenarioId ->
                showAnalyzeScenarioDialog(scenarioId)
            } ?: showToast(getString(R.string.select_scenario_first))
        }

        // Train neural network button
        binding.btnTrainNeuralNetwork.setOnClickListener {
            showTrainNeuralNetworkDialog()
        }

        // Generate insights button
        binding.btnGenerateInsights.setOnClickListener {
            selectedScenarioId?.let { scenarioId ->
                generateInsights(scenarioId)
            } ?: showToast(getString(R.string.select_scenario_first))
        }

        // Refine patterns button
        binding.btnRefinePatterns.setOnClickListener {
            selectedScenarioId?.let { scenarioId ->
                refinePatterns(scenarioId)
            } ?: showToast(getString(R.string.select_scenario_first))
        }

        // View scenario details button
        binding.btnViewScenarioDetails.setOnClickListener {
            selectedScenarioId?.let { scenarioId ->
                viewScenarioDetails(scenarioId)
            } ?: showToast(getString(R.string.select_scenario_first))
        }

        // Generate system report button
        binding.btnGenerateSystemReport.setOnClickListener {
            generateSystemReport()
        }

        // Export system data button
        binding.btnExportData.setOnClickListener {
            exportSystemData()
        }

        // Import system data button
        binding.btnImportData.setOnClickListener {
            showImportDataDialog()
        }

        // Scenario spinner
        binding.spinnerScenarios.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position >= 0) {
                    val summary = parent?.getItemAtPosition(position) as? ScenarioSummary
                    selectedScenarioId = summary?.id
                    updateStatusMessage("Selected scenario: ${summary?.name}")
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectedScenarioId = null
            }
        }
    }

    private fun initializeSystem(systemName: String = DEFAULT_SYSTEM_NAME) {
        showLoading(true, getString(R.string.initializing_system))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.initialize(applicationContext, systemName)
                if (result.isSuccess) {
                    updateStatusMessage(getString(R.string.system_initialized, systemName))
                    
                    // Enable UI elements
                    setUIEnabled(true)
                    
                    // Load scenarios
                    loadScenarios()
                } else {
                    updateStatusMessage(getString(R.string.initialization_failed))
                    showToast(getString(R.string.initialization_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing system", e)
                updateStatusMessage(getString(R.string.error_initializing_system))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun loadScenarios() {
        showLoading(true, getString(R.string.loading_scenarios))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.getAllScenarios()
                if (result.isSuccess) {
                    val scenarios = result.getOrDefault(emptyList())
                    updateScenarioSpinner(scenarios)
                    updateStatusMessage(getString(R.string.scenarios_loaded, scenarios.size))
                } else {
                    updateStatusMessage(getString(R.string.failed_to_load_scenarios))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading scenarios", e)
                updateStatusMessage(getString(R.string.error_loading_scenarios))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun showCreateScenarioDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_create_scenario, null)
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.create_scenario)
            .setView(dialogView)
            .setPositiveButton(R.string.create) { _, _ ->
                // Get values from dialog
                val name = dialogView.findViewById<android.widget.EditText>(R.id.et_scenario_name).text.toString()
                val description = dialogView.findViewById<android.widget.EditText>(R.id.et_scenario_description).text.toString()
                val domainsText = dialogView.findViewById<android.widget.EditText>(R.id.et_domains).text.toString()
                val themesText = dialogView.findViewById<android.widget.EditText>(R.id.et_themes).text.toString()
                val stakeholdersText = dialogView.findViewById<android.widget.EditText>(R.id.et_stakeholders).text.toString()
                
                if (name.isBlank()) {
                    showToast(getString(R.string.name_required))
                    return@setPositiveButton
                }
                
                // Parse comma-separated values
                val domains = domainsText.split(",").map { it.trim() }.filter { it.isNotBlank() }
                val themes = themesText.split(",").map { it.trim() }.filter { it.isNotBlank() }
                val stakeholders = stakeholdersText.split(",").map { it.trim() }.filter { it.isNotBlank() }
                
                // Create simple parameters map
                val parameters = mutableMapOf<String, Any>()
                parameters["time_horizon"] = 10 // Default value
                
                createScenario(name, description, parameters, domains, themes, stakeholders)
            }
            .setNegativeButton(R.string.cancel, null)
            .create()
        
        dialog.show()
    }

    private fun createScenario(
        name: String,
        description: String,
        parameters: Map<String, Any>,
        domains: List<String>,
        themes: List<String>,
        stakeholders: List<String>
    ) {
        showLoading(true, getString(R.string.creating_scenario))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.createScenario(
                    name = name,
                    description = description,
                    parameters = parameters,
                    domains = domains,
                    themes = themes,
                    stakeholders = stakeholders
                )
                
                if (result.isSuccess) {
                    val scenarioId = result.getOrThrow()
                    updateStatusMessage(getString(R.string.scenario_created, name))
                    showToast(getString(R.string.scenario_created_success))
                    
                    // Refresh scenarios list
                    loadScenarios()
                } else {
                    updateStatusMessage(getString(R.string.scenario_creation_failed))
                    showToast(getString(R.string.creation_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error creating scenario", e)
                updateStatusMessage(getString(R.string.error_creating_scenario))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun showAnalyzeScenarioDialog(scenarioId: String) {
        val dialogView = layoutInflater.inflate(R.layout.dialog_analyze_scenario, null)
        
        // Setup depth seekbar
        val depthSeekBar = dialogView.findViewById<android.widget.SeekBar>(R.id.seekbar_depth)
        val depthTextView = dialogView.findViewById<android.widget.TextView>(R.id.tv_depth_value)
        
        // Update text when seekbar changes
        depthSeekBar.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                depthTextView.text = progress.toString()
            }
            
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })
        
        // Setup checkboxes
        val tensionsCheckBox = dialogView.findViewById<android.widget.CheckBox>(R.id.cb_identify_tensions)
        val edgeCasesCheckBox = dialogView.findViewById<android.widget.CheckBox>(R.id.cb_explore_edge_cases)
        
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.analyze_scenario)
            .setView(dialogView)
            .setPositiveButton(R.string.analyze) { _, _ ->
                // Get values from dialog
                val depth = depthSeekBar.progress
                val identifyTensions = tensionsCheckBox.isChecked
                val exploreEdgeCases = edgeCasesCheckBox.isChecked
                
                // Generate random neural input (in a real app, this would be meaningful data)
                val neuralInput = DoubleArray(10) { Math.random() }
                
                analyzeScenario(scenarioId, neuralInput, depth, identifyTensions, exploreEdgeCases)
            }
            .setNegativeButton(R.string.cancel, null)
            .create()
        
        dialog.show()
    }

    private fun analyzeScenario(
        scenarioId: String,
        neuralInput: DoubleArray,
        consequenceDepth: Int,
        identifyTensions: Boolean,
        exploreEdgeCases: Boolean
    ) {
        showLoading(true, getString(R.string.analyzing_scenario))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.analyzeScenario(
                    scenarioId = scenarioId,
                    neuralInput = neuralInput,
                    consequenceDepth = consequenceDepth,
                    identifyTensions = identifyTensions,
                    exploreEdgeCases = exploreEdgeCases
                )
                
                if (result.isSuccess) {
                    val analysis = result.getOrThrow()
                    updateStatusMessage(getString(R.string.scenario_analyzed))
                    
                    // Show analysis summary
                    val summaryDialog = AlertDialog.Builder(this@MainActivityAdaptive)
                        .setTitle(R.string.analysis_summary)
                        .setMessage(formatAnalysisSummary(analysis))
                        .setPositiveButton(R.string.ok, null)
                        .create()
                    
                    summaryDialog.show()
                } else {
                    updateStatusMessage(getString(R.string.scenario_analysis_failed))
                    showToast(getString(R.string.analysis_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error analyzing scenario", e)
                updateStatusMessage(getString(R.string.error_analyzing_scenario))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun showTrainNeuralNetworkDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_train_neural_network, null)
        
        // Setup epochs seekbar
        val epochsSeekBar = dialogView.findViewById<android.widget.SeekBar>(R.id.seekbar_epochs)
        val epochsTextView = dialogView.findViewById<android.widget.TextView>(R.id.tv_epochs_value)
        
        // Update text when seekbar changes
        epochsSeekBar.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                epochsTextView.text = progress.toString()
            }
            
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })
        
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.train_neural_network)
            .setView(dialogView)
            .setPositiveButton(R.string.train) { _, _ ->
                // Get values from dialog
                val epochs = epochsSeekBar.progress
                
                // Generate random training data (in a real app, this would be meaningful data)
                val inputs = List(5) { DoubleArray(10) { Math.random() } }
                val targets = List(5) { DoubleArray(5) { Math.random() } }
                
                trainNeuralNetwork(inputs, targets, epochs)
            }
            .setNegativeButton(R.string.cancel, null)
            .create()
        
        dialog.show()
    }

    private fun trainNeuralNetwork(
        inputs: List<DoubleArray>,
        targets: List<DoubleArray>,
        epochs: Int
    ) {
        showLoading(true, getString(R.string.training_neural_network))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.trainNeuralNetwork(
                    inputs = inputs,
                    targets = targets,
                    epochs = epochs
                )
                
                if (result.isSuccess) {
                    val trainingResults = result.getOrThrow()
                    updateStatusMessage(getString(R.string.neural_network_trained))
                    
                    // Show training results
                    val resultsDialog = AlertDialog.Builder(this@MainActivityAdaptive)
                        .setTitle(R.string.training_results)
                        .setMessage(formatTrainingResults(trainingResults))
                        .setPositiveButton(R.string.ok, null)
                        .create()
                    
                    resultsDialog.show()
                } else {
                    updateStatusMessage(getString(R.string.neural_network_training_failed))
                    showToast(getString(R.string.training_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error training neural network", e)
                updateStatusMessage(getString(R.string.error_training_neural_network))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun generateInsights(scenarioId: String) {
        showLoading(true, getString(R.string.generating_insights))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.generateInsightsReport(scenarioId)
                
                if (result.isSuccess) {
                    val insightsReport = result.getOrThrow()
                    updateStatusMessage(getString(R.string.insights_generated))
                    
                    // Show insights report
                    val insightsDialog = AlertDialog.Builder(this@MainActivityAdaptive)
                        .setTitle(getString(R.string.insights_report_title, insightsReport.scenarioName))
                        .setMessage(formatInsightsReport(insightsReport))
                        .setPositiveButton(R.string.ok, null)
                        .create()
                    
                    insightsDialog.show()
                } else {
                    updateStatusMessage(getString(R.string.insights_generation_failed))
                    showToast(getString(R.string.generation_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error generating insights", e)
                updateStatusMessage(getString(R.string.error_generating_insights))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun refinePatterns(scenarioId: String) {
        showLoading(true, getString(R.string.refining_patterns))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.refinePatterns(scenarioId)
                
                if (result.isSuccess) {
                    val refinement = result.getOrThrow()
                    updateStatusMessage(getString(R.string.patterns_refined))
                    
                    // Show refinement results
                    val refinementDialog = AlertDialog.Builder(this@MainActivityAdaptive)
                        .setTitle(R.string.pattern_refinement_results)
                        .setMessage(formatPatternRefinement(refinement))
                        .setPositiveButton(R.string.ok, null)
                        .create()
                    
                    refinementDialog.show()
                } else {
                    updateStatusMessage(getString(R.string.pattern_refinement_failed))
                    showToast(getString(R.string.refinement_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error refining patterns", e)
                updateStatusMessage(getString(R.string.error_refining_patterns))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun viewScenarioDetails(scenarioId: String) {
        showLoading(true, getString(R.string.loading_scenario_details))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.getScenarioDetails(scenarioId)
                
                if (result.isSuccess) {
                    val details = result.getOrThrow()
                    updateStatusMessage(getString(R.string.scenario_details_loaded))
                    
                    // Show scenario details
                    val detailsDialog = AlertDialog.Builder(this@MainActivityAdaptive)
                        .setTitle(details.name)
                        .setMessage(formatScenarioDetails(details))
                        .setPositiveButton(R.string.ok, null)
                        .create()
                    
                    detailsDialog.show()
                } else {
                    updateStatusMessage(getString(R.string.scenario_details_loading_failed))
                    showToast(getString(R.string.loading_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading scenario details", e)
                updateStatusMessage(getString(R.string.error_loading_scenario_details))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun generateSystemReport() {
        showLoading(true, getString(R.string.generating_system_report))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.generateSystemReport()
                
                if (result.isSuccess) {
                    val report = result.getOrThrow()
                    updateStatusMessage(getString(R.string.system_report_generated))
                    
                    // Show system report
                    val reportDialog = AlertDialog.Builder(this@MainActivityAdaptive)
                        .setTitle(getString(R.string.system_report_title, report.name))
                        .setMessage(formatSystemReport(report))
                        .setPositiveButton(R.string.ok, null)
                        .create()
                    
                    reportDialog.show()
                } else {
                    updateStatusMessage(getString(R.string.system_report_generation_failed))
                    showToast(getString(R.string.generation_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error generating system report", e)
                updateStatusMessage(getString(R.string.error_generating_system_report))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun exportSystemData() {
        showLoading(true, getString(R.string.exporting_system_data))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.exportSystemData()
                
                if (result.isSuccess) {
                    val exportData = result.getOrThrow()
                    updateStatusMessage(getString(R.string.system_data_exported))
                    
                    // Show success dialog with preview
                    val previewText = if (exportData.length > 100) 
                        exportData.substring(0, 100) + "..." 
                    else 
                        exportData
                    
                    val exportDialog = AlertDialog.Builder(this@MainActivityAdaptive)
                        .setTitle(R.string.export_successful)
                        .setMessage(getString(R.string.export_preview, previewText))
                        .setPositiveButton(R.string.ok, null)
                        .create()
                    
                    exportDialog.show()
                    
                    // In a real app, you would save this to a file
                } else {
                    updateStatusMessage(getString(R.string.system_data_export_failed))
                    showToast(getString(R.string.export_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error exporting system data", e)
                updateStatusMessage(getString(R.string.error_exporting_system_data))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    private fun showImportDataDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_import_data, null)
        
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.import_system_data)
            .setView(dialogView)
            .setPositiveButton(R.string.import_btn) { _, _ ->
                val jsonData = dialogView.findViewById<android.widget.EditText>(R.id.et_import_data).text.toString()
                if (jsonData.isBlank()) {
                    showToast(getString(R.string.json_data_required))
                    return@setPositiveButton
                }
                
                importSystemData(jsonData)
            }
            .setNegativeButton(R.string.cancel, null)
            .create()
        
        dialog.show()
    }

    private fun importSystemData(jsonData: String) {
        showLoading(true, getString(R.string.importing_system_data))
        
        lifecycleScope.launch {
            try {
                val result = adaptiveBridge.importSystemData(jsonData)
                
                if (result.isSuccess && result.getOrThrow()) {
                    updateStatusMessage(getString(R.string.system_data_imported))
                    showToast(getString(R.string.import_successful))
                    
                    // Refresh scenarios
                    loadScenarios()
                } else {
                    updateStatusMessage(getString(R.string.system_data_import_failed))
                    showToast(getString(R.string.import_failed))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error importing system data", e)
                updateStatusMessage(getString(R.string.error_importing_system_data))
                showToast(e.message ?: getString(R.string.unknown_error))
            } finally {
                showLoading(false)
            }
        }
    }

    // Formatting helper methods

    private fun formatAnalysisSummary(analysis: com.adaptive.systemarchitect.model.ScenarioAnalysis): String {
        val sb = StringBuilder()
        sb.appendLine(getString(R.string.scenario_name_label, analysis.scenarioName))
        sb.appendLine(getString(R.string.analysis_time_label, analysis.analysisTime))
        sb.appendLine()
        
        analysis.consequenceChains?.let { chains ->
            sb.appendLine(getString(R.string.consequence_chains_header))
            sb.appendLine(getString(R.string.total_effects_label, chains.totalEffects))
            sb.appendLine(getString(R.string.analysis_depth_label, chains.analysisDepth))
            sb.appendLine(getString(R.string.feedback_loops_label, chains.feedbackLoops.size))
            sb.appendLine(getString(R.string.emergent_patterns_label, chains.emergentPatterns.size))
            sb.appendLine()
        }
        
        analysis.tensions?.let { tensions ->
            sb.appendLine(getString(R.string.tensions_header))
            sb.appendLine(getString(R.string.tension_count_label, tensions.tensionCount))
            sb.appendLine(getString(R.string.stakeholders_label, tensions.stakeholders.size))
            sb.appendLine(getString(R.string.principles_label, tensions.principles.size))
            sb.appendLine()
        }
        
        analysis.edgeCases?.let { edgeCases ->
            sb.appendLine(getString(R.string.edge_cases_header))
            sb.appendLine(getString(R.string.case_count_label, edgeCases.caseCount))
            sb.appendLine(getString(R.string.parameters_tested_label, edgeCases.parametersTested.joinToString(", ")))
            sb.appendLine(getString(R.string.high_severity_implications_label, edgeCases.highSeverityImplications))
            sb.appendLine()
        }
        
        analysis.neuralProcessing?.let { neural ->
            sb.appendLine(getString(R.string.neural_processing_header))
            sb.appendLine(getString(R.string.coherence_label, String.format("%.2f", neural.coherence)))
            sb.appendLine(getString(R.string.current_rhythm_label, neural.currentRhythm))
            sb.appendLine(getString(R.string.current_mode_label, neural.currentMode))
            sb.appendLine(getString(R.string.insights_count_label, neural.insights.size))
        }
        
        return sb.toString()
    }

    private fun formatTrainingResults(results: com.adaptive.systemarchitect.model.TrainingResults): String {
        val sb = StringBuilder()
        sb.appendLine(getString(R.string.epochs_label, results.epochs))
        sb.appendLine(getString(R.string.final_error_label, String.format("%.4f", results.finalError)))
        sb.appendLine(getString(R.string.coherence_label, String.format("%.2f", results.coherence)))
        sb.appendLine(getString(R.string.clusters_label, results.clusters))
        sb.appendLine(getString(R.string.current_mode_label, results.currentMode))
        sb.appendLine(getString(R.string.current_rhythm_label, results.currentRhythm))
        
        return sb.toString()
    }

    private fun formatInsightsReport(report: com.adaptive.systemarchitect.model.InsightsReport): String {
        val sb = StringBuilder()
        
        sb.appendLine(getString(R.string.key_insights_header))
        report.keyInsights.forEach { insight ->
            sb.appendLine("${insight.title} (${String.format("%.0f%%", insight.confidence * 100)} confidence)")
            sb.appendLine("  ${insight.description}")
            sb.appendLine()
        }
        
        if (report.systemInsights.isNotEmpty()) {
            sb.appendLine(getString(R.string.system_insights_header))
            report.systemInsights.forEach { insight ->
                sb.appendLine("${insight.title}")
                sb.appendLine("  ${insight.description}")
                sb.appendLine()
            }
        }
        
        sb.appendLine(getString(R.string.recommendations_header))
        report.recommendations.forEach { recommendation ->
            sb.appendLine("[${recommendation.priority.uppercase()}] ${recommendation.title}")
            sb.appendLine("  ${recommendation.description}")
            sb.appendLine()
        }
        
        sb.appendLine(getString(R.string.analysis_elements_label))
        val elements = report.analyzedElements
        sb.appendLine("  ${getString(R.string.patterns_label, elements.patterns)}")
        sb.appendLine("  ${getString(R.string.feedback_loops_label, elements.feedbackLoops)}")
        sb.appendLine("  ${getString(R.string.tensions_label, elements.tensions)}")
        sb.appendLine("  ${getString(R.string.edge_cases_label, elements.edgeCases)}")
        
        return sb.toString()
    }

    private fun formatPatternRefinement(refinement: com.adaptive.systemarchitect.model.PatternRefinement): String {
        val sb = StringBuilder()
        
        sb.appendLine(getString(R.string.validated_patterns_header))
        sb.appendLine(getString(R.string.validated_patterns_count, refinement.validatedPatterns.size))
        sb.appendLine(getString(R.string.high_confidence_patterns_label, refinement.highConfidencePatterns))
        sb.appendLine()
        
        // Show a few validated patterns
        refinement.validatedPatterns.take(3).forEach { pattern ->
            sb.appendLine("${getString(R.string.confidence_label, String.format("%.0f%%", pattern.refinedConfidence * 100))} (${pattern.validationLevel})")
            sb.appendLine("  ${getString(R.string.neural_support_label, if (pattern.neuralSupport) "Yes" else "No")}")
            sb.appendLine("  ${getString(R.string.feedback_support_label, if (pattern.feedbackSupport) "Yes" else "No")}")
            sb.appendLine()
        }
        
        if (refinement.contradictions.isNotEmpty()) {
            sb.appendLine(getString(R.string.contradictions_header))
            sb.appendLine(getString(R.string.contradictions_count, refinement.contradictions.size))
            
            // Show contradictions
            refinement.contradictions.forEach { contradiction ->
                sb.appendLine("${getString(R.string.contradiction_type_label, contradiction.type)}")
                sb.appendLine("  ${getString(R.string.severity_label, String.format("%.0f%%", contradiction.severity * 100))}")
                sb.appendLine()
            }
        }
        
        return sb.toString()
    }

    private fun formatScenarioDetails(details: com.adaptive.systemarchitect.model.ScenarioDetails): String {
        val sb = StringBuilder()
        
        sb.appendLine(details.description)
        sb.appendLine()
        
        sb.appendLine(getString(R.string.parameters_header))
        details.parameters.entries.forEach { (key, value) ->
            sb.appendLine("  $key: $value")
        }
        sb.appendLine()
        
        sb.appendLine(getString(R.string.domains_label, details.domains.joinToString(", ")))
        sb.appendLine(getString(R.string.themes_label, details.themes.joinToString(", ")))
        sb.appendLine(getString(R.string.stakeholders_label, details.stakeholders.joinToString(", ")))
        sb.appendLine()
        
        sb.appendLine(getString(R.string.creation_time_label, details.creationTime))
        sb.appendLine()
        
        sb.appendLine(getString(R.string.analysis_elements_label))
        sb.appendLine("  ${getString(R.string.effects_count_label, details.effectsCount)}")
        sb.appendLine("  ${getString(R.string.feedback_loops_count_label, details.feedbackLoopsCount)}")
        sb.appendLine("  ${getString(R.string.patterns_count_label, details.patternsCount)}")
        sb.appendLine("  ${getString(R.string.tensions_count_label, details.tensionsCount)}")
        sb.appendLine("  ${getString(R.string.edge_cases_count_label, details.edgeCasesCount)}")
        sb.appendLine()
        
        sb.appendLine(getString(R.string.has_neural_activations_label, if (details.hasNeuralActivations) "Yes" else "No"))
        sb.appendLine(getString(R.string.cross_domain_patterns_count_label, details.crossDomainPatternsCount))
        
        return sb.toString()
    }

    private fun formatSystemReport(report: com.adaptive.systemarchitect.model.SystemReport): String {
        val sb = StringBuilder()
        
        sb.appendLine(getString(R.string.system_id_label, report.systemId))
        sb.appendLine(getString(R.string.creation_time_label, report.creationTime))
        sb.appendLine(getString(R.string.last_modified_label, report.lastModified))
        sb.appendLine()
        
        sb.appendLine(getString(R.string.metrics_header))
        sb.appendLine("  ${getString(R.string.scenario_count_label, report.scenarioCount)}")
        sb.appendLine("  ${getString(R.string.effect_count_label, report.effectCount)}")
        sb.appendLine("  ${getString(R.string.pattern_count_label, report.patternCount)}")
        sb.appendLine("  ${getString(R.string.feedback_loop_count_label, report.feedbackLoopCount)}")
        sb.appendLine("  ${getString(R.string.tension_count_label, report.tensionCount)}")
        sb.appendLine("  ${getString(R.string.edge_case_count_label, report.edgeCaseCount)}")
        sb.appendLine("  ${getString(R.string.cross_domain_pattern_count_label, report.crossDomainPatternCount)}")
        sb.appendLine()
        
        sb.appendLine(getString(R.string.system_coherence_label, String.format("%.2f", report.systemCoherence)))
        sb.appendLine()
        
        if (report.mostActiveScenarios.isNotEmpty()) {
            sb.appendLine(getString(R.string.most_active_scenarios_header))
            report.mostActiveScenarios.forEach { scenario ->
                sb.appendLine("  ${scenario.name} (${scenario.activityLevel} activities)")
            }
            sb.appendLine()
        }
        
        if (report.topPatternScenarios.isNotEmpty()) {
            sb.appendLine(getString(R.string.top_pattern_scenarios_header))
            report.topPatternScenarios.forEach { scenario ->
                sb.appendLine("  ${scenario.name} (${scenario.patternCount} patterns)")
            }
            sb.appendLine()
        }
        
        sb.appendLine(getString(R.string.neural_network_status_header))
        val neural = report.neuralNetworkStatus
        sb.appendLine("  ${getString(R.string.total_training_epochs_label, neural.totalTrainingEpochs)}")
        sb.appendLine("  ${getString(R.string.current_mode_label, neural.currentMode)}")
        sb.appendLine("  ${getString(R.string.current_rhythm_label, neural.currentRhythm)}")
        sb.appendLine("  ${getString(R.string.coherence_label, String.format("%.2f", neural.coherence))}")
        
        return sb.toString()
    }

    // UI helper methods

    private fun updateScenarioSpinner(scenarios: List<ScenarioSummary>) {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            scenarios
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerScenarios.adapter = adapter
    }

    private fun updateStatusMessage(message: String) {
        binding.tvStatus.text = message
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun showLoading(show: Boolean, message: String = "") {
        binding.progressBar.isVisible = show
        if (show && message.isNotBlank()) {
            updateStatusMessage(message)
        }
    }

    private fun setUIEnabled(enabled: Boolean) {
        binding.btnCreateScenario.isEnabled = enabled
        binding.btnAnalyzeScenario.isEnabled = enabled
        binding.btnTrainNeuralNetwork.isEnabled = enabled
        binding.btnGenerateInsights.isEnabled = enabled
        binding.btnRefinePatterns.isEnabled = enabled
        binding.btnViewScenarioDetails.isEnabled = enabled
        binding.btnGenerateSystemReport.isEnabled = enabled
        binding.btnExportData.isEnabled = enabled
        binding.btnImportData.isEnabled = enabled
        binding.spinnerScenarios.isEnabled = enabled
    }
}
```
