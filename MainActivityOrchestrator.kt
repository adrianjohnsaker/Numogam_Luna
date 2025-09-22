  package com.antonio.my.ai.girlfriend.free.amelia.assemblage

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import androidx.recyclerview.widget.GridLayoutManager
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

class MainActivityOrchestrator : AppCompatActivity() {
    
    private lateinit var orchestrator: ModuleOrchestrator
    
    // UI Components
    private lateinit var inputField: EditText
    private lateinit var analyzeButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var resultsContainer: LinearLayout
    
    // Task Analysis Display
    private lateinit var categoriesChipGroup: ChipGroup
    private lateinit var complexityBudget: TextView
    private lateinit var creativeThreshold: TextView
    private lateinit var phasePreference: TextView
    private lateinit var deleuzeConceptsChipGroup: ChipGroup
    
    // Module Selection
    private lateinit var selectedModulesRecyclerView: RecyclerView
    private lateinit var selectedModulesAdapter: ModuleAdapter
    private val selectedModules = mutableListOf<String>()
    
    // Statistics Display
    private lateinit var totalModulesText: TextView
    private lateinit var avgIntensityText: TextView
    private lateinit var avgComplexityText: TextView
    private lateinit var connectionDensityText: TextView
    
    // Module Browser
    private lateinit var categorySpinner: Spinner
    private lateinit var moduleListRecyclerView: RecyclerView
    private lateinit var moduleListAdapter: ModuleListAdapter
    private val allModules = mutableListOf<ModuleOrchestrator.ModuleMetadata>()
    private val filteredModules = mutableListOf<ModuleOrchestrator.ModuleMetadata>()
    
    // Controls
    private lateinit var assemblageIntensityText: TextView
    private lateinit var suggestButton: Button
    private lateinit var clearButton: Button
    
    companion object {
        private const val TAG = "MainActivityOrchestrator"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_orchestrator)
        
        initializeViews()
        setupRecyclerViews()
        initializeOrchestrator()
    }
    
    private fun initializeViews() {
        // Input controls
        inputField = findViewById(R.id.input_field)
        analyzeButton = findViewById(R.id.analyze_button)
        progressBar = findViewById(R.id.progress_bar)
        statusText = findViewById(R.id.status_text)
        resultsContainer = findViewById(R.id.results_container)
        
        // Task analysis display
        categoriesChipGroup = findViewById(R.id.categories_chip_group)
        complexityBudget = findViewById(R.id.complexity_budget)
        creativeThreshold = findViewById(R.id.creative_threshold)
        phasePreference = findViewById(R.id.phase_preference)
        deleuzeConceptsChipGroup = findViewById(R.id.deleuze_concepts_chip_group)
        
        // Module selection
        selectedModulesRecyclerView = findViewById(R.id.selected_modules_recycler_view)
        
        // Statistics
        totalModulesText = findViewById(R.id.total_modules)
        avgIntensityText = findViewById(R.id.avg_intensity)
        avgComplexityText = findViewById(R.id.avg_complexity)
        connectionDensityText = findViewById(R.id.connection_density)
        
        // Module browser
        categorySpinner = findViewById(R.id.category_spinner)
        moduleListRecyclerView = findViewById(R.id.module_list_recycler_view)
        
        // Controls
        assemblageIntensityText = findViewById(R.id.assemblage_intensity)
        suggestButton = findViewById(R.id.suggest_button)
        clearButton = findViewById(R.id.clear_button)
        
        setupEventListeners()
    }
    
    private fun setupEventListeners() {
        analyzeButton.setOnClickListener { analyzeTask() }
        suggestButton.setOnClickListener { suggestAssemblage() }
        clearButton.setOnClickListener { clearSelection() }
        
        categorySpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                filterModulesByCategory()
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // Add example prompts
        val examplePrompts = listOf(
            "Create a complex narrative using memory and consciousness",
            "Generate creative solutions with recursive reflection",
            "Explore ontological drift through poetic becoming",
            "Synthesize desire and temporality in assemblages",
            "Analyze hyperstitional loops with zone navigation"
        )
        
        findViewById<Button>(R.id.example_button)?.setOnClickListener {
            inputField.setText(examplePrompts.random())
        }
    }
    
    private fun setupRecyclerViews() {
        // Selected modules adapter
        selectedModulesAdapter = ModuleAdapter(
            modules = selectedModules,
            onModuleClick = { moduleName -> showModuleDetails(moduleName) },
            onModuleRemove = { moduleName -> removeModule(moduleName) }
        )
        
        selectedModulesRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivityOrchestrator, LinearLayoutManager.HORIZONTAL, false)
            adapter = selectedModulesAdapter
        }
        
        // Module list adapter
        moduleListAdapter = ModuleListAdapter(
            modules = filteredModules,
            onModuleClick = { metadata -> addModule(metadata.name) },
            onModuleDetails = { metadata -> showModuleDetails(metadata.name) }
        )
        
        moduleListRecyclerView.apply {
            layoutManager = GridLayoutManager(this@MainActivityOrchestrator, 2)
            adapter = moduleListAdapter
        }
    }
    
    private fun initializeOrchestrator() {
        orchestrator = ModuleOrchestrator.getInstance()
        
        updateStatus("Initializing module orchestrator...")
        showProgress(true)
        
        lifecycleScope.launch {
            try {
                val success = orchestrator.initialize()
                if (success) {
                    updateStatus("Module orchestrator ready")
                    analyzeButton.isEnabled = true
                    suggestButton.isEnabled = true
                    
                    // Load initial data
                    loadStatistics()
                    loadCategories()
                    loadAllModules()
                } else {
                    updateStatus("Failed to initialize orchestrator")
                    showError("Failed to initialize module system")
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
    
    private fun analyzeTask() {
        val input = inputField.text.toString().trim()
        if (input.isEmpty()) {
            showError("Please enter a task description")
            return
        }
        
        updateStatus("Analyzing task requirements...")
        showProgress(true)
        analyzeButton.isEnabled = false
        resultsContainer.visibility = View.GONE
        
        lifecycleScope.launch {
            try {
                val taskAnalysis = orchestrator.analyzeTaskRequirements(input)
                
                if (taskAnalysis != null) {
                    displayTaskAnalysis(taskAnalysis)
                    
                    // Auto-select modules based on analysis
                    val suggestedModules = orchestrator.selectModulesForTask(taskAnalysis)
                    selectedModules.clear()
                    selectedModules.addAll(suggestedModules)
                    selectedModulesAdapter.notifyDataSetChanged()
                    
                    updateAssemblageIntensity()
                    updateStatus("Task analysis completed")
                    resultsContainer.visibility = View.VISIBLE
                } else {
                    showError("Failed to analyze task")
                    updateStatus("Analysis failed")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Analysis error", e)
                showError("Analysis error: ${e.localizedMessage}")
                updateStatus("Error: ${e.message}")
            } finally {
                showProgress(false)
                analyzeButton.isEnabled = true
            }
        }
    }
    
    private fun displayTaskAnalysis(analysis: ModuleOrchestrator.TaskAnalysis) {
        // Display categories
        categoriesChipGroup.removeAllViews()
        analysis.categories.forEach { category ->
            val chip = Chip(this).apply {
                text = category.capitalize()
                isClickable = false
                setChipBackgroundColorResource(R.color.category_chip_background)
            }
            categoriesChipGroup.addView(chip)
        }
        
        // Display metrics
        complexityBudget.text = String.format("%.1f", analysis.complexityBudget)
        creativeThreshold.text = String.format("%.2f", analysis.creativeThreshold)
        phasePreference.text = if (analysis.phasePreference > 0) 
            "Phase ${analysis.phasePreference}" else "No preference"
        
        // Display Deleuze concepts
        deleuzeConceptsChipGroup.removeAllViews()
        analysis.deleuzeConcepts.forEach { concept ->
            val chip = Chip(this).apply {
                text = concept.replace("_", " ").capitalize()
                isClickable = false
                setChipBackgroundColorResource(R.color.concept_chip_background)
            }
            deleuzeConceptsChipGroup.addView(  chip)
        }
    }
    
    private fun suggestAssemblage() {
        if (selectedModules.isEmpty()) {
            showError("Please select at least one module first")
            return
        }
        
        updateStatus("Suggesting assemblage connections...")
        showProgress(true)
        
        lifecycleScope.launch {
            try {
                val suggestions = orchestrator.suggestModuleAssemblage(selectedModules, 12)
                
                selectedModules.clear()
                selectedModules.addAll(suggestions)
                selectedModulesAdapter.notifyDataSetChanged()
                
                updateAssemblageIntensity()
                updateStatus("Assemblage suggestions applied")
                
            } catch (e: Exception) {
                Log.e(TAG, "Suggestion error", e)
                showError("Failed to suggest assemblage: ${e.localizedMessage}")
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun addModule(moduleName: String) {
        if (!selectedModules.contains(moduleName)) {
            selectedModules.add(moduleName)
            selectedModulesAdapter.notifyItemInserted(selectedModules.size - 1)
            updateAssemblageIntensity()
        }
    }
    
    private fun removeModule(moduleName: String) {
        val index = selectedModules.indexOf(moduleName)
        if (index >= 0) {
            selectedModules.removeAt(index)
            selectedModulesAdapter.notifyItemRemoved(index)
            updateAssemblageIntensity()
        }
    }
    
    private fun clearSelection() {
        selectedModules.clear()
        selectedModulesAdapter.notifyDataSetChanged()
        updateAssemblageIntensity()
        updateStatus("Selection cleared")
    }
    
    private fun updateAssemblageIntensity() {
        lifecycleScope.launch {
            try {
                val intensity = orchestrator.calculateAssemblageIntensity(selectedModules)
                assemblageIntensityText.text = String.format("%.3f", intensity)
                
                // Color code the intensity
                val color = when {
                    intensity >= 0.8 -> getColor(R.color.high_intensity)
                    intensity >= 0.6 -> getColor(R.color.medium_intensity)
                    else -> getColor(R.color.low_intensity)
                }
                assemblageIntensityText.setTextColor(color)
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to update assemblage intensity", e)
            }
        }
    }
    
    private fun loadStatistics() {
        lifecycleScope.launch {
            try {
                val stats = orchestrator.getAssemblageStatistics()
                if (stats != null) {
                    totalModulesText.text = stats.totalModules.toString()
                    avgIntensityText.text = String.format("%.3f", stats.averageCreativeIntensity)
                    avgComplexityText.text = String.format("%.3f", stats.averageComplexity)
                    connectionDensityText.text = String.format("%.3f", stats.connectionDensity)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load statistics", e)
            }
        }
    }
    
    private fun loadCategories() {
        lifecycleScope.launch {
            try {
                val categories = orchestrator.getAvailableCategories()
                val categoryList = mutableListOf("All Categories")
                categoryList.addAll(categories.map { it.capitalize() })
                
                val adapter = ArrayAdapter(
                    this@MainActivityOrchestrator,
                    android.R.layout.simple_spinner_item,
                    categoryList
                )
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                categorySpinner.adapter = adapter
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load categories", e)
            }
        }
    }
    
    private fun loadAllModules() {
        lifecycleScope.launch {
            try {
                val stats = orchestrator.getAssemblageStatistics()
                if (stats != null) {
                    // Load all modules by iterating through categories
                    allModules.clear()
                    
                    for ((category, count) in stats.categoryDistribution) {
                        if (count > 0) {
                            val categoryModules = orchestrator.getModulesByCategory(category)
                            for (moduleName in categoryModules) {
                                val metadata = orchestrator.getModuleMetadata(moduleName)
                                if (metadata != null) {
                                    allModules.add(metadata)
                                }
                            }
                        }
                    }
                    
                    filteredModules.clear()
                    filteredModules.addAll(allModules)
                    moduleListAdapter.notifyDataSetChanged()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load all modules", e)
            }
        }
    }
    
    private fun filterModulesByCategory() {
        val selectedCategory = categorySpinner.selectedItem.toString()
        
        filteredModules.clear()
        if (selectedCategory == "All Categories") {
            filteredModules.addAll(allModules)
        } else {
            filteredModules.addAll(allModules.filter { 
                it.category.equals(selectedCategory, ignoreCase = true) 
            })
        }
        
        moduleListAdapter.notifyDataSetChanged()
    }
    
    private fun showModuleDetails(moduleName: String) {
        lifecycleScope.launch {
            try {
                val metadata = orchestrator.getModuleMetadata(moduleName)
                val connections = orchestrator.getModuleConnections(moduleName)
                
                if (metadata != null) {
                    val dialog = ModuleDetailDialog(this@MainActivityOrchestrator, metadata, connections) { 
                        connectionsToAdd ->
                        // Add connected modules to selection
                        connectionsToAdd.forEach { connection ->
                            addModule(connection)
                        }
                    }
                    dialog.show()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to show module details", e)
            }
        }
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
        orchestrator.cleanup()
    }
}

/**
 * RecyclerView adapter for selected modules (horizontal chips)
 */
class ModuleAdapter(
    private val modules: MutableList<String>,
    private val onModuleClick: (String) -> Unit,
    private val onModuleRemove: (String) -> Unit
) : RecyclerView.Adapter<ModuleAdapter.ViewHolder>() {
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val moduleChip: Chip = view.findViewById(R.id.module_chip)
    }
    
    override fun onCreateViewHolder(parent: android.view.ViewGroup, viewType: Int): ViewHolder {
        val view = android.view.LayoutInflater.from(parent.context)
            .inflate(R.layout.item_selected_module, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val moduleName = modules[position]
        
        holder.moduleChip.apply {
            text = moduleName.replace("_", " ").split(" ").joinToString(" ") { 
                it.capitalize() 
            }
            
            isCloseIconVisible = true
            setOnClickListener { onModuleClick(moduleName) }
            setOnCloseIconClickListener { onModuleRemove(moduleName) }
        }
    }
    
    override fun getItemCount() = modules.size
}

/**
 * RecyclerView adapter for module list (grid view)
 */
class ModuleListAdapter(
    private val modules: MutableList<ModuleOrchestrator.ModuleMetadata>,
    private val onModuleClick: (ModuleOrchestrator.ModuleMetadata) -> Unit,
    private val onModuleDetails: (ModuleOrchestrator.ModuleMetadata) -> Unit
) : RecyclerView.Adapter<ModuleListAdapter.ViewHolder>() {
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val moduleName: TextView = view.findViewById(R.id.module_name)
        val moduleCategory: TextView = view.findViewById(R.id.module_category)
        val creativeIntensity: TextView = view.findViewById(R.id.creative_intensity)
        val complexityLevel: TextView = view.findViewById(R.id.complexity_level)
        val purposeText: TextView = view.findViewById(R.id.purpose_text)
        val intensityBar: View = view.findViewById(R.id.intensity_bar)
        val detailsButton: Button = view.findViewById(R.id.details_button)
    }
    
    override fun onCreateViewHolder(parent: android.view.ViewGroup, viewType: Int): ViewHolder {
        val view = android.view.LayoutInflater.from(parent.context)
            .inflate(R.layout.item_module_list, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHelper(holder: ViewHolder, position: Int) {
        val module = modules[position]
        
        holder.moduleName.text = module.name.replace("_", " ").split(" ").joinToString(" ") { 
            it.capitalize() 
        }
        holder.moduleCategory.text = module.category.capitalize()
        holder.creativeIntensity.text = String.format("%.2f", module.creativeIntensity)
        holder.complexityLevel.text = String.format("%.2f", module.complexityLevel)
        holder.purposeText.text = module.purpose
        
        // Set intensity bar width and color
        val layoutParams = holder.intensityBar.layoutParams
        layoutParams.width = (holder.intensityBar.context.resources.displayMetrics.density * 
                            module.creativeIntensity * 100).toInt()
        holder.intensityBar.layoutParams = layoutParams
        
        val intensityColor = when {
            module.creativeIntensity >= 0.8 -> holder.itemView.context.getColor(R.color.high_intensity)
            module.creativeIntensity >= 0.6 -> holder.itemView.context.getColor(R.color.medium_intensity)
            else -> holder.itemView.context.getColor(R.color.low_intensity)
        }
        holder.intensityBar.setBackgroundColor(intensityColor)
        
        holder.itemView.setOnClickListener { onModuleClick(module) }
        holder.detailsButton.setOnClickListener { onModuleDetails(module) }
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        onBindViewHelper(holder, position)
    }
    
    override fun getItemCount() = modules.size
}

/**
 * Dialog for showing detailed module information
 */
class ModuleDetailDialog(
    private val context: android.content.Context,
    private val metadata: ModuleOrchestrator.ModuleMetadata,
    private val connections: List<String>,
    private val onAddConnections: (List<String>) -> Unit
) {
    
    fun show() {
        val dialog = androidx.appcompat.app.AlertDialog.Builder(context)
            .setTitle(metadata.name.replace("_", " ").split(" ").joinToString(" ") { it.capitalize() })
            .setView(createDetailView())
            .setPositiveButton("Add Connections") { _, _ -> 
                onAddConnections(connections.take(3)) // Add top 3 connections
            }
            .setNegativeButton("Close", null)
            .create()
        
        dialog.show()
    }
    
    private fun createDetailView(): View {
        val scrollView = android.widget.ScrollView(context)
        val layout = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(32, 16, 32, 16)
        }
        
        // Add detail items
        addDetailItem(layout, "Category", metadata.category.capitalize())
        addDetailItem(layout, "Purpose", metadata.purpose)
        addDetailItem(layout, "Creative Intensity", String.format("%.3f", metadata.creativeIntensity))
        addDetailItem(layout, "Complexity Level", String.format("%.3f", metadata.complexityLevel))
        addDetailItem(layout, "Processing Weight", String.format("%.3f", metadata.processingWeight))
        
        if (metadata.phaseAlignment > 0) {
            addDetailItem(layout, "Phase Alignment", "Phase ${metadata.phaseAlignment}")
        }
        
        if (metadata.deleuzeConcepts.isNotEmpty()) {
            val conceptsText = metadata.deleuzeConcepts.joinToString(", ") { 
                it.replace("_", " ").capitalize() 
            }
            addDetailItem(layout, "Deleuze Concepts", conceptsText)
        }
        
        if (connections.isNotEmpty()) {
            val connectionsText = connections.take(5).joinToString("\n") { "• $it" }
            addDetailItem(layout, "Connection Affinities", connectionsText)
        }
        
        scrollView.addView(layout)
        return scrollView
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
