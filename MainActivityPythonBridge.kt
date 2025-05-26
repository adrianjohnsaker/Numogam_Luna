package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.snackbar.Snackbar
import com.universal.python.bridge.*
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*

class MainActivityPythonBridge : AppCompatActivity() {
    
    // Bridge components
    private lateinit var bridge: UniversalPythonBridge
    private lateinit var bridgeManager: PythonBridgeManager
    private lateinit var performanceMonitor: PythonBridgePerformanceMonitor
    
    // UI components
    private lateinit var moduleSpinner: Spinner
    private lateinit var functionInput: EditText
    private lateinit var argumentsInput: EditText
    private lateinit var resultTextView: TextView
    private lateinit var executeButton: Button
    private lateinit var statusTextView: TextView
    private lateinit var chipGroup: ChipGroup
    private lateinit var historyRecyclerView: RecyclerView
    private lateinit var progressBar: ProgressBar
    private lateinit var fabQuickActions: FloatingActionButton
    
    // Module management
    private val loadedModules = mutableListOf<String>()
    private val executionHistory = mutableListOf<ExecutionRecord>()
    private lateinit var historyAdapter: HistoryAdapter
    
    // Amelia-specific modules
    private val ameliaModules = listOf(
        "astral_symbolic_module",
        "consciousness_engine",
        "creativity_enhancer", 
        "emotion_processor",
        "memory_weaver",
        "dream_interpreter",
        "intuition_matrix",
        "narrative_generator"
    )
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_python_bridge)
        
        initializeViews()
        initializePythonBridge()
        setupUI()
    }
    
    private fun initializeViews() {
        moduleSpinner = findViewById(R.id.moduleSpinner)
        functionInput = findViewById(R.id.functionInput)
        argumentsInput = findViewById(R.id.argumentsInput)
        resultTextView = findViewById(R.id.resultTextView)
        executeButton = findViewById(R.id.executeButton)
        statusTextView = findViewById(R.id.statusTextView)
        chipGroup = findViewById(R.id.chipGroup)
        historyRecyclerView = findViewById(R.id.historyRecyclerView)
        progressBar = findViewById(R.id.progressBar)
        fabQuickActions = findViewById(R.id.fabQuickActions)
    }
    
    private fun initializePythonBridge() {
        // Initialize Python environment
        UniversalPythonBridge.initialize(this)
        bridge = UniversalPythonBridge.getInstance(this)
        bridgeManager = PythonBridgeManager.getInstance(this)
        performanceMonitor = PythonBridgePerformanceMonitor()
        
        // Register Amelia's modules
        registerAmeliaModules()
        
        // Initialize the bridge manager
        bridgeManager.initialize()
        
        updateStatus(getString(R.string.status_initializing))
    }
    
    private fun registerAmeliaModules() {
        ameliaModules.forEach { moduleName ->
            bridgeManager.registerModule(
                PythonModuleConfig(
                    moduleName = moduleName,
                    autoInitialize = false,
                    requiredPythonPackages = when(moduleName) {
                        "astral_symbolic_module" -> listOf("numpy", "json")
                        "consciousness_engine" -> listOf("numpy", "tensorflow")
                        "creativity_enhancer" -> listOf("numpy", "random")
                        else -> listOf("numpy")
                    }
                )
            )
        }
    }
    
    private fun setupUI() {
        // Setup module spinner
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, ameliaModules)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        moduleSpinner.adapter = adapter
        
        // Setup execute button
        executeButton.setOnClickListener {
            executePythonFunction()
        }
        
        // Setup history RecyclerView
        historyAdapter = HistoryAdapter(executionHistory)
        historyRecyclerView.layoutManager = LinearLayoutManager(this)
        historyRecyclerView.adapter = historyAdapter
        
        // Setup FAB for quick actions
        fabQuickActions.setOnClickListener {
            showQuickActionsMenu()
        }
        
        // Setup quick action chips
        setupQuickActionChips()
    }
    
    private fun setupQuickActionChips() {
        val quickActions = listOf(
            "Initialize Astral" to { initializeAstralModule() },
            "Check Status" to { checkModuleStatus() },
            "Run Demo" to { runDemoSession() },
            "Clear History" to { clearHistory() }
        )
        
        quickActions.forEach { (label, action) ->
            val chip = Chip(this).apply {
                text = label
                isClickable = true
                setOnClickListener { action() }
            }
            chipGroup.addView(chip)
        }
    }
    
    private fun executePythonFunction() {
        val module = moduleSpinner.selectedItem.toString()
        val function = functionInput.text.toString()
        val arguments = argumentsInput.text.toString()
        
        if (function.isBlank()) {
            showError(getString(R.string.error_empty_function))
            return
        }
        
        lifecycleScope.launch {
            try {
                showProgress(true)
                updateStatus(getString(R.string.status_executing))
                
                // Parse arguments (simple comma-separated for demo)
                val args = if (arguments.isBlank()) {
                    emptyArray()
                } else {
                    arguments.split(",").map { it.trim() }.toTypedArray()
                }
                
                // Execute with performance monitoring
                val result = performanceMonitor.measureTime("$module.$function") {
                    bridge.callModuleFunction(module, function, *args)
                }
                
                // Display result
                val resultStr = result?.toString() ?: "null"
                displayResult(resultStr)
                
                // Add to history
                addToHistory(module, function, arguments, resultStr)
                
                updateStatus(getString(R.string.status_success))
                
            } catch (e: Exception) {
                showError(getString(R.string.error_execution, e.message))
                updateStatus(getString(R.string.status_error))
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun initializeAstralModule() {
        lifecycleScope.launch {
            try {
                showProgress(true)
                val astral = AstralModuleWrapper(this@MainActivityPythonBridge)
                val success = astral.createModule()
                
                if (success) {
                    val session = astral.initializeSession("Amelia Session")
                    displayResult("Astral Module Initialized!\n${session?.toString(2)}")
                    updateStatus(getString(R.string.status_astral_ready))
                } else {
                    showError(getString(R.string.error_astral_init))
                }
            } catch (e: Exception) {
                showError(getString(R.string.error_astral_init_exception, e.message))
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun checkModuleStatus() {
        lifecycleScope.launch {
            try {
                showProgress(true)
                val stats = bridgeManager.getStatistics()
                val bridgeStats = bridge.getStatistics()
                
                val combined = JSONObject().apply {
                    put("bridge", bridgeStats)
                    put("manager", stats)
                    put("performance", JSONObject(performanceMonitor.getStatistics()))
                }
                
                displayResult(combined.toString(2))
                updateStatus(getString(R.string.status_checked))
            } catch (e: Exception) {
                showError(getString(R.string.error_status_check, e.message))
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun runDemoSession() {
        lifecycleScope.launch {
            try {
                showProgress(true)
                updateStatus(getString(R.string.status_running_demo))
                
                val result = bridge.callModuleFunction("astral_symbolic_module", "demo_session")
                displayResult(result?.toString() ?: "Demo completed")
                
                updateStatus(getString(R.string.status_demo_complete))
            } catch (e: Exception) {
                showError(getString(R.string.error_demo, e.message))
            } finally {
                showProgress(false)
            }
        }
    }
    
    private fun clearHistory() {
        executionHistory.clear()
        historyAdapter.notifyDataSetChanged()
        Snackbar.make(findViewById(R.id.rootLayout), 
            getString(R.string.history_cleared), 
            Snackbar.LENGTH_SHORT).show()
    }
    
    private fun showQuickActionsMenu() {
        // Could implement a bottom sheet or dialog here
        Snackbar.make(fabQuickActions, 
            getString(R.string.quick_actions_hint), 
            Snackbar.LENGTH_SHORT).show()
    }
    
    private fun displayResult(result: String) {
        resultTextView.text = result
        resultTextView.visibility = View.VISIBLE
    }
    
    private fun updateStatus(status: String) {
        statusTextView.text = status
    }
    
    private fun showProgress(show: Boolean) {
        progressBar.visibility = if (show) View.VISIBLE else View.GONE
        executeButton.isEnabled = !show
    }
    
    private fun showError(message: String) {
        Snackbar.make(findViewById(R.id.rootLayout), message, Snackbar.LENGTH_LONG)
            .setBackgroundTint(getColor(android.R.color.holo_red_dark))
            .show()
    }
    
    private fun addToHistory(module: String, function: String, args: String, result: String) {
        val record = ExecutionRecord(
            timestamp = System.currentTimeMillis(),
            module = module,
            function = function,
            arguments = args,
            result = result
        )
        executionHistory.add(0, record)
        if (executionHistory.size > 50) {
            executionHistory.removeAt(executionHistory.size - 1)
        }
        historyAdapter.notifyItemInserted(0)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        bridge.cleanup()
    }
}

// Data classes
data class ExecutionRecord(
    val timestamp: Long,
    val module: String,
    val function: String,
    val arguments: String,
    val result: String
)

// RecyclerView Adapter for history
class HistoryAdapter(private val history: List<ExecutionRecord>) : 
    RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {
    
    private val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val cardView: MaterialCardView = view.findViewById(R.id.cardView)
        val timeTextView: TextView = view.findViewById(R.id.timeTextView)
        val moduleTextView: TextView = view.findViewById(R.id.moduleTextView)
        val functionTextView: TextView = view.findViewById(R.id.functionTextView)
        val resultPreviewTextView: TextView = view.findViewById(R.id.resultPreviewTextView)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_execution_history, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val record = history[position]
        holder.timeTextView.text = dateFormat.format(Date(record.timestamp))
        holder.moduleTextView.text = record.module
        holder.functionTextView.text = "${record.function}(${record.arguments})"
        holder.resultPreviewTextView.text = record.result.take(100) + 
            if (record.result.length > 100) "..." else ""
            
        holder.cardView.setOnClickListener {
            // Could expand to show full result
        }
    }
    
    override fun getItemCount() = history.size
}
