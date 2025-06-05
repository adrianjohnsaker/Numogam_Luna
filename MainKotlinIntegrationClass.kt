package com.antonio.my.ai.girlfriend.free.ai.python

import com.chaquo.python.Python
import com.chaquo.python.PyObject
import com.chaquo.python.PyException
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject

class AIModulePromoter {
    
    private val python: Python by lazy { Python.getInstance() }
    private val promoterModule: PyObject by lazy { 
        python.getModule("ai_assistant_module_promoter") 
    }
    
    // Data classes for type safety
    data class ModuleRecommendation(
        val name: String,
        val description: String,
        val useCases: List<String>,
        val installCommand: String,
        val compatibilityNotes: String,
        val priority: Int,
        val dependencies: List<String>
    )
    
    data class TaskAnalysis(
        val detectedCategories: List<String>,
        val recommendations: List<ModuleRecommendation>,
        val promotionMessage: String,
        val suggestedWorkflows: Map<String, List<String>>
    )
    
    // Main function to analyze user input and get module recommendations
    suspend fun analyzeUserTask(userInput: String): TaskAnalysis = withContext(Dispatchers.IO) {
        try {
            // Get the basic promotion message
            val promotionMessage = promoterModule
                .callAttr("promote_modules_for_task", userInput)
                .toString()
            
            // Create context-aware promoter for advanced features
            val contextPromoter = python.getModule("ai_assistant_module_promoter")
                .callAttr("ContextAwarePromoter")
            
            // Analyze task context
            val categories = contextPromoter
                .callAttr("analyze_task_context", userInput)
                .asList()
                .map { it.toString() }
            
            // Get detailed recommendations
            val pyRecommendations = contextPromoter
                .callAttr("get_personalized_recommendations", categories)
                .asList()
            
            val recommendations = pyRecommendations.map { pyRec ->
                parseModuleRecommendation(pyRec)
            }
            
            // Get workflow suggestions
            val workflowsPy = contextPromoter
                .callAttr("suggest_module_workflows", userInput)
                .asMap()
            
            val workflows = workflowsPy.mapKeys { it.key.toString() }
                .mapValues { entry ->
                    entry.value.asList().map { it.toString() }
                }
            
            TaskAnalysis(
                detectedCategories = categories,
                recommendations = recommendations,
                promotionMessage = promotionMessage,
                suggestedWorkflows = workflows
            )
            
        } catch (e: PyException) {
            handlePythonException(e)
            // Return fallback analysis
            TaskAnalysis(
                detectedCategories = listOf("general"),
                recommendations = getDefaultRecommendations(),
                promotionMessage = "Consider using enhanced Python modules for better results!",
                suggestedWorkflows = emptyMap()
            )
        }
    }
    
    // Real-time suggestions as user types
    suspend fun getRealTimeSuggestions(partialInput: String): Map<String, Double> = 
        withContext(Dispatchers.IO) {
            try {
                val suggester = python.getModule("ai_assistant_module_promoter")
                    .callAttr("RealTimeModuleSuggester")
                
                val suggestions = suggester
                    .callAttr("analyze_partial_input", partialInput)
                    .asMap()
                
                suggestions.mapKeys { it.key.toString() }
                    .mapValues { it.value.toDouble() }
                    
            } catch (e: Exception) {
                emptyMap()
            }
        }
    
    // Get quick recommendations for common task types
    fun getQuickRecommendations(taskType: String): List<String> {
        return try {
            promoterModule
                .callAttr("get_quick_recommendations", taskType)
                .asList()
                .map { it.toString() }
        } catch (e: PyException) {
            handlePythonException(e)
            getDefaultModuleNames()
        }
    }
    
    // Get Android-optimized modules
    fun getAndroidOptimizedModules(): List<String> {
        return try {
            promoterModule
                .callAttr("get_android_optimized_modules")
                .asList()
                .map { it.toString() }
        } catch (e: PyException) {
            handlePythonException(e)
            listOf("pandas", "requests", "numpy", "pillow")
        }
    }
    
    // Execute Python code with recommended modules
    suspend fun executeWithRecommendedModules(
        code: String, 
        recommendedModules: List<String>
    ): ExecutionResult = withContext(Dispatchers.IO) {
        
        val startTime = System.currentTimeMillis()
        
        try {
            // Import recommended modules
            val imports = recommendedModules.joinToString("\n") { "import $it" }
            val fullCode = "$imports\n\n$code"
            
            // Execute the code
            val result = python.getModule("builtins")
                .callAttr("exec", fullCode)
            
            val executionTime = System.currentTimeMillis() - startTime
            
            // Log performance for learning
            logModulePerformance(recommendedModules, executionTime, true)
            
            ExecutionResult.Success(result.toString(), executionTime)
            
        } catch (e: PyException) {
            val executionTime = System.currentTimeMillis() - startTime
            logModulePerformance(recommendedModules, executionTime, false)
            
            ExecutionResult.Error(e.message ?: "Python execution failed", executionTime)
        }
    }
    
    // Helper functions
    private fun parseModuleRecommendation(pyRec: PyObject): ModuleRecommendation {
        return ModuleRecommendation(
            name = pyRec.callAttr("__getattribute__", "name").toString(),
            description = pyRec.callAttr("__getattribute__", "description").toString(),
            useCases = pyRec.callAttr("__getattribute__", "use_cases")
                .asList().map { it.toString() },
            installCommand = pyRec.callAttr("__getattribute__", "install_command").toString(),
            compatibilityNotes = pyRec.callAttr("__getattribute__", "compatibility_notes").toString(),
            priority = pyRec.callAttr("__getattribute__", "priority").toInt(),
            dependencies = pyRec.callAttr("__getattribute__", "dependencies")
                .asList().map { it.toString() }
        )
    }
    
    private fun handlePythonException(e: PyException) {
        // Log the exception for debugging
        android.util.Log.e("AIModulePromoter", "Python Exception: ${e.message}", e)
    }
    
    private fun logModulePerformance(modules: List<String>, executionTime: Long, success: Boolean) {
        // Log performance data for machine learning improvements
        modules.forEach { module ->
            // You can implement analytics logging here
            android.util.Log.d("ModulePerformance", 
                "Module: $module, Time: ${executionTime}ms, Success: $success")
        }
    }
    
    private fun getDefaultRecommendations(): List<ModuleRecommendation> {
        return listOf(
            ModuleRecommendation(
                name = "pandas",
                description = "Data manipulation and analysis",
                useCases = listOf("CSV processing", "Data cleaning"),
                installCommand = "pip install pandas",
                compatibilityNotes = "Excellent Android compatibility",
                priority = 9,
                dependencies = listOf("numpy")
            )
        )
    }
    
    private fun getDefaultModuleNames(): List<String> {
        return listOf("pandas", "numpy", "requests", "pillow")
    }
    
    sealed class ExecutionResult {
        data class Success(val output: String, val executionTime: Long) : ExecutionResult()
        data class Error(val error: String, val executionTime: Long) : ExecutionResult()
    }
}

// 3. UI INTEGRATION EXAMPLE
class ModuleRecommendationFragment : Fragment() {
    
    private lateinit var aiPromoter: AIModulePromoter
    private lateinit var binding: FragmentModuleRecommendationBinding
    
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        binding = FragmentModuleRecommendationBinding.inflate(inflater, container, false)
        aiPromoter = AIModulePromoter()
        
        setupUI()
        return binding.root
    }
    
    private fun setupUI() {
        // Real-time suggestions as user types
        binding.userInputEditText.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            
            override fun afterTextChanged(s: Editable?) {
                val input = s.toString()
                if (input.length > 3) { // Start suggesting after 3 characters
                    lifecycleScope.launch {
                        val suggestions = aiPromoter.getRealTimeSuggestions(input)
                        updateRealTimeSuggestions(suggestions)
                    }
                }
            }
        })
        
        // Analyze button click
        binding.analyzeButton.setOnClickListener {
            val userInput = binding.userInputEditText.text.toString()
            if (userInput.isNotBlank()) {
                analyzeUserTask(userInput)
            }
        }
        
        // Quick task buttons
        setupQuickTaskButtons()
    }
    
    private fun setupQuickTaskButtons() {
        val quickTasks = mapOf(
            binding.dataAnalysisButton to "data",
            binding.webScrapingButton to "web",
            binding.mlButton to "ml",
            binding.imageProcessingButton to "image"
        )
        
        quickTasks.forEach { (button, taskType) ->
            button.setOnClickListener {
                val recommendations = aiPromoter.getQuickRecommendations(taskType)
                showQuickRecommendations(taskType, recommendations)
            }
        }
    }
    
    private fun analyzeUserTask(userInput: String) {
        binding.progressBar.visibility = View.VISIBLE
        
        lifecycleScope.launch {
            try {
                val analysis = aiPromoter.analyzeUserTask(userInput)
                displayAnalysisResults(analysis)
            } catch (e: Exception) {
                showError("Failed to analyze task: ${e.message}")
            } finally {
                binding.progressBar.visibility = View.GONE
            }
        }
    }
    
    private fun displayAnalysisResults(analysis: AIModulePromoter.TaskAnalysis) {
        // Display promotion message
        binding.promotionMessageTextView.text = analysis.promotionMessage
        
        // Display module recommendations
        val adapter = ModuleRecommendationAdapter(analysis.recommendations) { module ->
            // Handle module selection
            showModuleDetails(module)
        }
        binding.recommendationsRecyclerView.adapter = adapter
        
        // Display workflows
        displayWorkflows(analysis.suggestedWorkflows)
        
        // Show detected categories
        binding.detectedCategoriesTextView.text = 
            "Detected tasks: ${analysis.detectedCategories.joinToString(", ")}"
    }
    
    private fun updateRealTimeSuggestions(suggestions: Map<String, Double>) {
        val sortedSuggestions = suggestions.toList()
            .sortedByDescending { it.second }
            .take(3)
        
        binding.realTimeSuggestionsLayout.removeAllViews()
        
        sortedSuggestions.forEach { (module, confidence) ->
            val chip = Chip(requireContext()).apply {
                text = "$module (${(confidence * 100).toInt()}%)"
                isClickable = true
                setOnClickListener {
                    // Add module to current context
                    addModuleToContext(module)
                }
            }
            binding.realTimeSuggestionsLayout.addView(chip)
        }
    }
    
    private fun showModuleDetails(module: AIModulePromoter.ModuleRecommendation) {
        val dialog = MaterialAlertDialogBuilder(requireContext())
            .setTitle(module.name)
            .setMessage("""
                ${module.description}
                
                Use Cases:
                ${module.useCases.joinToString("\n• ", "• ")}
                
                Installation: ${module.installCommand}
                
                Android Notes: ${module.compatibilityNotes}
                
                Dependencies: ${module.dependencies.joinToString(", ")}
            """.trimIndent())
            .setPositiveButton("Use This Module") { _, _ ->
                useModule(module)
            }
            .setNegativeButton("Cancel", null)
            .create()
        
        dialog.show()
    }
    
    private fun useModule(module: AIModulePromoter.ModuleRecommendation) {
        // Implement module usage logic
        val codeTemplate = generateCodeTemplate(module)
        
        // Show code template to user or execute directly
        showCodeTemplate(module.name, codeTemplate)
    }
    
    private fun generateCodeTemplate(module: AIModulePromoter.ModuleRecommendation): String {
        return when (module.name) {
            "pandas" -> """
                import pandas as pd
                
                # Load your data
                df = pd.read_csv('your_file.csv')
                
                # Basic analysis
                print(df.head())
                print(df.describe())
            """.trimIndent()
            
            "requests" -> """
                import requests
                
                # Make API call
                response = requests.get('https://api.example.com/data')
                data = response.json()
                print(data)
            """.trimIndent()
            
            "scikit-learn" -> """
                import pandas as pd
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score
                
                # Load and prepare data
                df = pd.read_csv('your_data.csv')
                X = df.drop('target', axis=1)
                y = df['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                # Train model
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                
                # Evaluate
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                print(f'Accuracy: {accuracy:.2f}')
            """.trimIndent()
            
            "beautifulsoup4" -> """
                import requests
                from bs4 import BeautifulSoup
                
                # Scrape webpage
                url = 'https://example.com'
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract data
                titles = soup.find_all('h1')
                for title in titles:
                    print(title.text.strip())
            """.trimIndent()
            
            "pillow" -> """
                from PIL import Image
                import os
                
                # Open and process image
                image = Image.open('input_image.jpg')
                
                # Resize image
                resized = image.resize((800, 600))
                
                # Save processed image
                resized.save('output_image.jpg', 'JPEG', quality=85)
                print('Image processed successfully')
            """.trimIndent()
            
            "openpyxl" -> """
                import openpyxl
                from openpyxl import Workbook
                
                # Create new workbook
                wb = Workbook()
                ws = wb.active
                
                # Add data
                ws['A1'] = 'Name'
                ws['B1'] = 'Value'
                ws['A2'] = 'Sample'
                ws['B2'] = 42
                
                # Save file
                wb.save('output.xlsx')
                print('Excel file created')
            """.trimIndent()
            
            "nltk" -> """
                import nltk
                from nltk.sentiment import SentimentIntensityAnalyzer
                
                # Download required data (do this once)
                # nltk.download('vader_lexicon')
                
                # Analyze sentiment
                sia = SentimentIntensityAnalyzer()
                text = "Your text here"
                sentiment = sia.polarity_scores(text)
                print(f'Sentiment: {sentiment}')
            """.trimIndent()
            
            else -> """
                import ${module.name}
                
                # Use ${module.name} for your task
                # Add your implementation here
            """.trimIndent()
        }
    }
    
    private fun showCodeTemplate(moduleName: String, template: String) {
        val dialog = MaterialAlertDialogBuilder(requireContext())
            .setTitle("Code Template: $moduleName")
            .setMessage(template)
            .setPositiveButton("Execute") { _, _ ->
                executeTemplate(template)
            }
            .setNeutralButton("Copy") { _, _ ->
                copyToClipboard(template)
            }
            .setNegativeButton("Cancel", null)
            .create()
        
        dialog.show()
    }
    
    private fun executeTemplate(code: String) {
        binding.progressBar.visibility = View.VISIBLE
        
        lifecycleScope.launch {
            try {
                val result = aiPromoter.executeWithRecommendedModules(
                    code, 
                    extractModulesFromCode(code)
                )
                
                when (result) {
                    is AIModulePromoter.ExecutionResult.Success -> {
                        showExecutionResult("Success", result.output, result.executionTime)
                    }
                    is AIModulePromoter.ExecutionResult.Error -> {
                        showExecutionResult("Error", result.error, result.executionTime)
                    }
                }
            } catch (e: Exception) {
                showError("Execution failed: ${e.message}")
            } finally {
                binding.progressBar.visibility = View.GONE
            }
        }
    }
    
    private fun extractModulesFromCode(code: String): List<String> {
        val importRegex = """import\s+(\w+)""".toRegex()
        val fromImportRegex = """from\s+(\w+)\s+import""".toRegex()
        
        val modules = mutableSetOf<String>()
        
        importRegex.findAll(code).forEach { match ->
            modules.add(match.groupValues[1])
        }
        
        fromImportRegex.findAll(code).forEach { match ->
            modules.add(match.groupValues[1])
        }
        
        return modules.toList()
    }
    
    private fun showExecutionResult(title: String, message: String, executionTime: Long) {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("$title (${executionTime}ms)")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }
    
    private fun copyToClipboard(text: String) {
        val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("Code Template", text)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(requireContext(), "Code copied to clipboard", Toast.LENGTH_SHORT).show()
    }
    
    private fun showQuickRecommendations(taskType: String, recommendations: List<String>) {
        val message = "Recommended modules for $taskType:\n\n${recommendations.joinToString("\n• ", "• ")}"
        
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Quick Recommendations")
            .setMessage(message)
            .setPositiveButton("Get Details") { _, _ ->
                // Show detailed analysis for this task type
                analyzeUserTask("I want to work with $taskType")
            }
            .setNegativeButton("OK", null)
            .show()
    }
    
    private fun displayWorkflows(workflows: Map<String, List<String>>) {
        if (workflows.isEmpty()) {
            binding.workflowsContainer.visibility = View.GONE
            return
        }
        
        binding.workflowsContainer.visibility = View.VISIBLE
        binding.workflowsContainer.removeAllViews()
        
        workflows.forEach { (workflowName, steps) ->
            val workflowCard = LayoutInflater.from(requireContext())
                .inflate(R.layout.workflow_card, binding.workflowsContainer, false)
            
            workflowCard.findViewById<TextView>(R.id.workflowTitle).text = workflowName
            
            val stepsContainer = workflowCard.findViewById<LinearLayout>(R.id.stepsContainer)
            steps.forEachIndexed { index, step ->
                val stepView = TextView(requireContext()).apply {
                    text = step
                    setPadding(16, 8, 16, 8)
                    setTextColor(ContextCompat.getColor(requireContext(), R.color.text_secondary))
                }
                stepsContainer.addView(stepView)
            }
            
            binding.workflowsContainer.addView(workflowCard)
        }
    }
    
    private fun addModuleToContext(moduleName: String) {
        // Add module to current session context
        val currentText = binding.userInputEditText.text.toString()
        if (!currentText.contains(moduleName)) {
            binding.userInputEditText.setText("$currentText using $moduleName")
            binding.userInputEditText.setSelection(binding.userInputEditText.text.length)
        }
    }
    
    private fun showError(message: String) {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Error")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }
}

// 4. RECYCLER VIEW ADAPTER FOR MODULE RECOMMENDATIONS
class ModuleRecommendationAdapter(
    private val recommendations: List<AIModulePromoter.ModuleRecommendation>,
    private val onModuleClick: (AIModulePromoter.ModuleRecommendation) -> Unit
) : RecyclerView.Adapter<ModuleRecommendationAdapter.ViewHolder>() {
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val moduleNameTextView: TextView = view.findViewById(R.id.moduleNameTextView)
        val moduleDescriptionTextView: TextView = view.findViewById(R.id.moduleDescriptionTextView)
        val priorityChip: Chip = view.findViewById(R.id.priorityChip)
        val useCasesTextView: TextView = view.findViewById(R.id.useCasesTextView)
        val compatibilityTextView: TextView = view.findViewById(R.id.compatibilityTextView)
        val useModuleButton: Button = view.findViewById(R.id.useModuleButton)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_module_recommendation, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val recommendation = recommendations[position]
        
        holder.moduleNameTextView.text = recommendation.name
        holder.moduleDescriptionTextView.text = recommendation.description
        holder.priorityChip.text = "Priority: ${recommendation.priority}"
        holder.useCasesTextView.text = "Use cases: ${recommendation.useCases.take(3).joinToString(", ")}"
        holder.compatibilityTextView.text = recommendation.compatibilityNotes
        
        // Set priority chip color based on priority level
        val chipColor = when {
            recommendation.priority >= 8 -> R.color.priority_high
            recommendation.priority >= 6 -> R.color.priority_medium
            else -> R.color.priority_low
        }
        holder.priorityChip.setChipBackgroundColorResource(chipColor)
        
        holder.useModuleButton.setOnClickListener {
            onModuleClick(recommendation)
        }
    }
    
    override fun getItemCount() = recommendations.size
}

// 5. BACKGROUND SERVICE FOR CONTINUOUS LEARNING
class ModuleLearningService : Service() {
    
    private lateinit var aiPromoter: AIModulePromoter
    private val binder = LocalBinder()
    
    inner class LocalBinder : Binder() {
        fun getService(): ModuleLearningService = this@ModuleLearningService
    }
    
    override fun onBind(intent: Intent): IBinder = binder
    
    override fun onCreate() {
        super.onCreate()
        aiPromoter = AIModulePromoter()
    }
    
    fun logUserFeedback(moduleName: String, rating: Int, context: String) {
        // This would typically save to a database or send to analytics
        try {
            val contextPromoter = Python.getInstance()
                .getModule("ai_assistant_module_promoter")
                .callAttr("ContextAwarePromoter")
            
            contextPromoter.callAttr("learn_from_user_feedback", moduleName, rating, context)
        } catch (e: Exception) {
            Log.e("ModuleLearningService", "Failed to log feedback", e)
        }
    }
    
    fun getPersonalizedRecommendations(taskCategories: List<String>): List<AIModulePromoter.ModuleRecommendation> {
        return try {
            // Implementation would call the personalized recommendation system
            emptyList()
        } catch (e: Exception) {
            Log.e("ModuleLearningService", "Failed to get personalized recommendations", e)
            emptyList()
        }
    }
}

// 6. APPLICATION CLASS SETUP
class AIAssistantApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize Python
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        // Pre-load common modules for better performance
        preloadPythonModules()
    }
    
    private fun preloadPythonModules() {
        Thread {
            try {
                val python = Python.getInstance()
                
                // Pre-import commonly used modules
                python.getModule("ai_assistant_module_promoter")
                python.getModule("pandas")
                python.getModule("numpy")
                python.getModule("requests")
                
                Log.d("AIAssistantApp", "Python modules preloaded successfully")
            } catch (e: Exception) {
                Log.e("AIAssistantApp", "Failed to preload Python modules", e)
            }
        }.start()
    }
}

// 7. UTILITY EXTENSIONS
fun Context.showModulePromotion(userInput: String) {
    val intent = Intent(this, ModulePromotionActivity::class.java).apply {
        putExtra("user_input", userInput)
    }
    startActivity(intent)
}

fun Fragment.showModuleRecommendations(recommendations: List<AIModulePromoter.ModuleRecommendation>) {
    val bottomSheet = ModuleRecommendationBottomSheet.newInstance(recommendations)
    bottomSheet.show(childFragmentManager, "module_recommendations")
}

// 8. CONSTANTS AND CONFIGURATION
object ModulePromoterConfig {
    const val MAX_RECOMMENDATIONS = 5
    const val MIN_CONFIDENCE_THRESHOLD = 0.6
    const val REAL_TIME_SUGGESTION_DELAY = 500L // milliseconds
    
    val SUPPORTED_FILE_TYPES = setOf(
        "csv", "xlsx", "json", "txt", "py", "jpg", "png", "pdf"
    )
    
    val TASK_KEYWORDS = mapOf(
        "data_analysis" to listOf("analyze", "data", "csv", "statistics", "trends"),
        "machine_learning" to listOf("predict", "classify", "model", "train", "ml", "ai"),
        "web_scraping" to listOf("scrape", "website", "html", "url", "extract"),
        "image_processing" to listOf("image", "photo", "picture", "resize", "filter"),
        "text_processing" to listOf("text", "sentiment", "language", "words", "nlp")
    )
}

/*
// 9. SAMPLE LAYOUT FILES (XML)

// res/layout/fragment_module_recommendation.xml
<?xml version="1.0" encoding="utf-8"?>
<ScrollView xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <com.google.android.material.textfield.TextInputLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:hint="Describe your task...">

            <com.google.android.material.textfield.TextInputEditText
                android:id="@+id/userInputEditText"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:minLines="3" />

        </com.google.android.material.textfield.TextInputLayout>

        <com.google.android.material.chip.ChipGroup
            android:id="@+id/realTimeSuggestionsLayout"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="8dp" />

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:orientation="horizontal">

            <Button
                android:id="@+id/dataAnalysisButton"
                style="@style/Widget.Material3.Button.OutlinedButton"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:layout_marginEnd="4dp"
                android:text="Data" />

            <Button
                android:id="@+id/webScrapingButton"
                style="@style/Widget.Material3.Button.OutlinedButton"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:layout_marginStart="4dp"
                android:layout_marginEnd="4dp"
                android:text="Web" />

            <Button
                android:id="@+id/mlButton"
                style="@style/Widget.Material3.Button.OutlinedButton"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:layout_marginStart="4dp"
                android:layout_marginEnd="4dp"
                android:text="ML" />

            <Button
                android:id="@+id/imageProcessingButton"
                style="@style/Widget.Material3.Button.OutlinedButton"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:layout_marginStart="4dp"
                android:text="Image" />

        </LinearLayout>

        <Button
            android:id="@+id/analyzeButton"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:text="Analyze & Get Recommendations" />

        <ProgressBar
            android:id="@+id/progressBar"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_gravity="center"
            android:layout_marginTop="16dp"
            android:visibility="gone" />

        <TextView
            android:id="@+id/detectedCategoriesTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:textStyle="italic" />

        <TextView
            android:id="@+id/promotionMessageTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:textSize="16sp" />

        <androidx.recyclerview.widget.RecyclerView
            android:id="@+id/recommendationsRecyclerView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            app:layoutManager="androidx.recyclerview.widget.LinearLayoutManager" />

        <LinearLayout
            android:id="@+id/workflowsContainer"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:orientation="vertical"
            android:visibility="gone" />

    </LinearLayout>

</ScrollView>

// res/layout/item_module_recommendation.xml
<?xml version="1.0" encoding="utf-8"?>
<com.google.android.material.card.MaterialCardView xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:layout_marginVertical="4dp"
    app:cardElevation="2dp"
    app:cardCornerRadius="8dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical"
        android:padding="16dp">

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:gravity="center_vertical">

            <TextView
                android:id="@+id/moduleNameTextView"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:textSize="18sp"
                android:textStyle="bold" />

            <com.google.android.material.chip.Chip
                android:id="@+id/priorityChip"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content" />

        </LinearLayout>

        <TextView
            android:id="@+id/moduleDescriptionTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="8dp" />

        <TextView
            android:id="@+id/useCasesTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="8dp"
            android:textSize="14sp"
            android:textColor="?android:attr/textColorSecondary" />

        <TextView
            android:id="@+id/compatibilityTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="4dp"
            android:textSize="12sp"
            android:textStyle="italic"
            android:textColor="?android:attr/textColorSecondary" />

        <Button
            android:id="@+id/useModuleButton"
            style="@style/Widget.Material3.Button.OutlinedButton"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_gravity="end"
            android:layout_marginTop="12dp"
            android:text="Use This Module" />

    </LinearLayout>

</com.google.android.material.card.MaterialCardView>
*/
