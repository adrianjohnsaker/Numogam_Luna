package com.antonio.my.ai.girlfriend.free.amelia.android.activities

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.StaggeredGridLayoutManager
import com.amelia.android.R
import com.amelia.android.adapters.DreamSessionAdapter
import com.amelia.android.adapters.QuickAnalysisAdapter
import com.amelia.android.databinding.ActivityMainBinding
import com.amelia.android.models.*
import com.amelia.android.services.DreamAnalysisService
import com.amelia.android.utils.SettingsManager
import com.amelia.android.viewmodels.MainViewModel
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Enhanced Main Activity for Amelia Dream Analysis
 * Integrates all analysis components with modern UI
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()
    
    @Inject
    lateinit var dreamAnalysisService: DreamAnalysisService
    
    @Inject
    lateinit var settingsManager: SettingsManager
    
    private lateinit var dreamSessionAdapter: DreamSessionAdapter
    private lateinit var quickAnalysisAdapter: QuickAnalysisAdapter
    
    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_DREAM_INPUT = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        // Handle splash screen
        val splashScreen = installSplashScreen()
        
        super.onCreate(savedInstanceState)
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setupToolbar()
        setupAdapters()
        setupObservers()
        setupClickListeners()
        initializeServices()
        
        // Handle splash screen exit
        splashScreen.setKeepOnScreenCondition { 
            !viewModel.isInitialized.value 
        }
        
        // Check if first launch
        if (settingsManager.isFirstLaunch) {
            showWelcomeDialog()
            settingsManager.isFirstLaunch = false
        }
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.apply {
            title = getString(R.string.app_name)
            subtitle = "Dream Analysis & Narrative Generation"
        }
    }

    private fun setupAdapters() {
        // Dream sessions adapter
        dreamSessionAdapter = DreamSessionAdapter(
            dreamSessions = mutableListOf(),
            onItemClick = { session -> openSessionDetails(session) },
            onItemLongClick = { session -> showSessionOptions(session) }
        )
        
        binding.rvRecentSessions.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = dreamSessionAdapter
        }
        
        // Quick analysis adapter
        quickAnalysisAdapter = QuickAnalysisAdapter { analysisType ->
            startQuickAnalysis(analysisType)
        }
        
        binding.rvQuickAnalysis.apply {
            layoutManager = StaggeredGridLayoutManager(2, StaggeredGridLayoutManager.VERTICAL)
            adapter = quickAnalysisAdapter
        }
    }

    private fun setupObservers() {
        // Initialization status
        viewModel.isInitialized.observe(this) { initialized ->
            if (initialized) {
                binding.progressInitialization.hide()
                loadDashboardData()
            } else {
                binding.progressInitialization.show()
            }
        }
        
        // Recent sessions
        viewModel.recentSessions.observe(this) { sessions ->
            dreamSessionAdapter.updateSessions(sessions)
            updateDashboardStats(sessions)
        }
        
        // Analysis progress
        viewModel.analysisProgress.observe(this) { progress ->
            handleAnalysisProgress(progress)
        }
        
        // Error messages
        viewModel.error.observe(this) { error ->
            error?.let {
                Toast.makeText(this, it, Toast.LENGTH_LONG).show()
            }
        }
        
        // Service status
        viewModel.serviceStatus.observe(this) { status ->
            updateServiceStatusIndicator(status)
        }
    }

    private fun setupClickListeners() {
        // Floating Action Button - New Dream Analysis
        binding.fabNewAnalysis.setOnClickListener {
            startNewDreamAnalysis()
        }
        
        // Quick Analysis Cards
        binding.cardSymbolicMapper.setOnClickListener {
            startActivity(Intent(this, SymbolicDreamMapperActivity::class.java))
        }
        
        binding.cardMythogenicEngine.setOnClickListener {
            startActivity(Intent(this, MainActivityMythogenic::class.java))
        }
        
        binding.cardFieldDreaming.setOnClickListener {
            startActivity(Intent(this, FieldDreamingSystemActivity::class.java))
        }
        
        binding.cardNarrativeGenerator.setOnClickListener {
            startNarrativeGeneration()
        }
        
        // Dashboard stats cards
        binding.cardTotalAnalyses.setOnClickListener {
            viewAllAnalyses()
        }
        
        binding.cardPatternInsights.setOnClickListener {
            viewPatternInsights()
        }
        
        binding.cardTransformationScenarios.setOnClickListener {
            viewTransformationScenarios()
        }
        
        // Refresh button
        binding.swipeRefreshLayout.setOnRefreshListener {
            refreshDashboard()
        }
    }

    private fun initializeServices() {
        lifecycleScope.launch {
            try {
                binding.progressInitialization.show()
                
                // Initialize dream analysis service
                val initialized = dreamAnalysisService.initialize()
                
                if (initialized) {
                    viewModel.setInitialized(true)
                    Toast.makeText(
                        this@MainActivity, 
                        "Amelia Dream Analysis Ready", 
                        Toast.LENGTH_SHORT
                    ).show()
                } else {
                    viewModel.setError("Failed to initialize analysis services")
                }
            } catch (e: Exception) {
                viewModel.setError("Initialization failed: ${e.message}")
            } finally {
                binding.progressInitialization.hide()
            }
        }
    }

    private fun loadDashboardData() {
        lifecycleScope.launch {
            viewModel.loadRecentSessions()
            viewModel.loadDashboardStats()
        }
    }

    private fun startNewDreamAnalysis() {
        val intent = Intent(this, DreamInputActivity::class.java)
        startActivityForResult(intent, REQUEST_DREAM_INPUT)
    }

    private fun startQuickAnalysis(analysisType: AnalysisType) {
        when (analysisType) {
            AnalysisType.SYMBOLIC_MAPPING -> {
                startActivity(Intent(this, SymbolicDreamMapperActivity::class.java))
            }
            AnalysisType.MYTHOGENIC_DREAMING -> {
                startActivity(Intent(this, MainActivityMythogenic::class.java))
            }
            AnalysisType.FIELD_DREAMING -> {
                startActivity(Intent(this, FieldDreamingSystemActivity::class.java))
            }
            AnalysisType.COMPREHENSIVE -> {
                startComprehensiveAnalysis()
            }
        }
    }

    private fun startComprehensiveAnalysis() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Comprehensive Dream Analysis")
            .setMessage("This will perform a complete analysis using all available engines. Continue?")
            .setPositiveButton("Analyze") { _, _ ->
                val intent = Intent(this, ComprehensiveAnalysisActivity::class.java)
                startActivity(intent)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startNarrativeGeneration() {
        // Check if there are recent analyses available
        lifecycleScope.launch {
            val recentSessions = viewModel.recentSessions.value
            if (recentSessions.isNullOrEmpty()) {
                MaterialAlertDialogBuilder(this@MainActivity)
                    .setTitle("No Analyses Available")
                    .setMessage("You need to perform a dream analysis first before generating narratives.")
                    .setPositiveButton("Start Analysis") { _, _ ->
                        startNewDreamAnalysis()
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            } else {
                // Show analysis selection dialog
                showAnalysisSelectionForNarrative(recentSessions)
            }
        }
    }

    private fun showAnalysisSelectionForNarrative(sessions: List<DreamSession>) {
        val sessionTitles = sessions.map { it.title }.toTypedArray()
        
        MaterialAlertDialogBuilder(this)
            .setTitle("Select Analysis for Narrative")
            .setItems(sessionTitles) { _, which ->
                val selectedSession = sessions[which]
                val intent = Intent(this, NarrativeGenerationActivity::class.java).apply {
                    putExtra("analysis_id", selectedSession.id)
                }
                startActivity(intent)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun openSessionDetails(session: DreamSession) {
        val intent = Intent(this, EnhancedAnalysisResultActivity::class.java).apply {
            putExtra("analysis_id", session.id)
        }
        startActivity(intent)
    }

    private fun showSessionOptions(session: DreamSession) {
        val options = arrayOf(
            "View Details",
            "Generate Narrative", 
            "Field Dynamics",
            "Export Analysis",
            "Delete Session"
        )
        
        MaterialAlertDialogBuilder(this)
            .setTitle(session.title)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> openSessionDetails(session)
                    1 -> generateNarrativeForSession(session)
                    2 -> openFieldDynamics(session)
                    3 -> exportSession(session)
                    4 -> deleteSession(session)
                }
            }
            .show()
    }

    private fun generateNarrativeForSession(session: DreamSession) {
        val intent = Intent(this, NarrativeGenerationActivity::class.java).apply {
            putExtra("analysis_id", session.id)
        }
        startActivity(intent)
    }

    private fun openFieldDynamics(session: DreamSession) {
        val intent = Intent(this, FieldDynamicsActivity::class.java).apply {
            putExtra("analysis_id", session.id)
        }
        startActivity(intent)
    }

    private fun exportSession(session: DreamSession) {
        lifecycleScope.launch {
            try {
                viewModel.exportSession(session, ExportFormat.JSON)
                Toast.makeText(this@MainActivity, "Session exported successfully", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun deleteSession(session: DreamSession) {
        MaterialAlertDialogBuilder(this)
            .setTitle("Delete Session")
            .setMessage("Are you sure you want to delete '${session.title}'? This action cannot be undone.")
            .setPositiveButton("Delete") { _, _ ->
                lifecycleScope.launch {
                    viewModel.deleteSession(session.id)
                    Toast.makeText(this@MainActivity, "Session deleted", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun viewAllAnalyses() {
        val intent = Intent(this, AnalysisHistoryActivity::class.java)
        startActivity(intent)
    }

    private fun viewPatternInsights() {
        val intent = Intent(this, PatternInsightsActivity::class.java)
        startActivity(intent)
    }

    private fun viewTransformationScenarios() {
        val intent = Intent(this, TransformationInsightsActivity::class.java)
        startActivity(intent)
    }

    private fun refreshDashboard() {
        lifecycleScope.launch {
            try {
                viewModel.refreshDashboard()
            } finally {
                binding.swipeRefreshLayout.isRefreshing = false
            }
        }
    }

    private fun updateDashboardStats(sessions: List<DreamSession>) {
        // Total analyses
        binding.tvTotalAnalyses.text = sessions.size.toString()
        
        // Pattern insights count
        val patternCount = sessions.sumOf { it.symbolMappings.size }
        binding.tvPatternInsights.text = patternCount.toString()
        
        // Transformation scenarios count
        val transformationCount = sessions.count { 
            it.mythogenicElements.isNotEmpty() || it.fieldDreamingData != null 
        }
        binding.tvTransformationScenarios.text = transformationCount.toString()
        
        // Recent activity
        val recentActivity = if (sessions.isNotEmpty()) {
            val latestSession = sessions.maxByOrNull { it.timestamp }
            val timeAgo = calculateTimeAgo(latestSession?.timestamp ?: 0L)
            "Last analysis: $timeAgo"
        } else {
            "No recent activity"
        }
        binding.tvRecentActivity.text = recentActivity
    }

    private fun calculateTimeAgo(timestamp: Long): String {
        val now = System.currentTimeMillis()
        val diff = now - timestamp
        
        return when {
            diff < 60_000 -> "Just now"
            diff < 3600_000 -> "${diff / 60_000}m ago"
            diff < 86400_000 -> "${diff / 3600_000}h ago"
            else -> "${diff / 86400_000}d ago"
        }
    }

    private fun handleAnalysisProgress(progress: com.amelia.android.services.AnalysisProgress) {
        when (progress) {
            is com.amelia.android.services.AnalysisProgress.Processing -> {
                binding.progressAnalysis.show()
                binding.tvAnalysisStatus.text = "${progress.step}: ${progress.progress}%"
            }
            is com.amelia.android.services.AnalysisProgress.Completed -> {
                binding.progressAnalysis.hide()
                binding.tvAnalysisStatus.text = "Analysis completed"
                
                // Navigate to results
                val intent = Intent(this, EnhancedAnalysisResultActivity::class.java).apply {
                    putExtra("analysis_id", progress.analysisId)
                }
                startActivity(intent)
            }
            is com.amelia.android.services.AnalysisProgress.Error -> {
                binding.progressAnalysis.hide()
                binding.tvAnalysisStatus.text = "Analysis failed"
                Toast.makeText(this, progress.message, Toast.LENGTH_LONG).show()
            }
            else -> {
                binding.progressAnalysis.hide()
                binding.tvAnalysisStatus.text = ""
            }
        }
    }

    private fun updateServiceStatusIndicator(status: ServiceStatus) {
        when (status) {
            ServiceStatus.INITIALIZING -> {
                binding.ivServiceStatus.setImageResource(R.drawable.ic_service_initializing)
                binding.tvServiceStatus.text = "Initializing..."
            }
            ServiceStatus.READY -> {
                binding.ivServiceStatus.setImageResource(R.drawable.ic_service_ready)
                binding.tvServiceStatus.text = "Ready"
            }
            ServiceStatus.PROCESSING -> {
                binding.ivServiceStatus.setImageResource(R.drawable.ic_service_processing)
                binding.tvServiceStatus.text = "Processing..."
            }
            ServiceStatus.ERROR -> {
                binding.ivServiceStatus.setImageResource(R.drawable.ic_service_error)
                binding.tvServiceStatus.text = "Error"
            }
        }
    }

    private fun showWelcomeDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Welcome to Amelia")
            .setMessage("""
                Amelia is an advanced dream analysis system that combines:
                
                • Neuro-Symbolic AI for deep pattern recognition
                • Mythogenic analysis for archetypal insights  
                • Field dynamics for consciousness mapping
                • Narrative generation for story creation
                
                Ready to explore your dreams?
            """.trimIndent())
            .setPositiveButton("Get Started") { _, _ ->
                startNewDreamAnalysis()
            }
            .setNegativeButton("Explore Features", null)
            .setCancelable(false)
            .show()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> {
                startActivity(Intent(this, SettingsActivity::class.java))
                true
            }
            R.id.action_about -> {
                showAboutDialog()
                true
            }
            R.id.action_backup -> {
                performBackup()
                true
            }
            R.id.action_restore -> {
                performRestore()
                true
            }
            R.id.action_research_mode -> {
                toggleResearchMode()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun showAboutDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("About Amelia")
            .setMessage("""
                Amelia Dream Analysis v${BuildConfig.VERSION_NAME}
                
                Advanced neuro-symbolic dream analysis system featuring:
                - Enhanced symbolic mapping with VSA
                - Mythogenic pattern recognition  
                - Field dynamics visualization
                - Deterritorialization vectors
                - Narrative generation engine
                
                Built with Kotlin, Python AI, and Chaquopy integration.
            """.trimIndent())
            .setPositiveButton("OK", null)
            .show()
    }

    private fun performBackup() {
        lifecycleScope.launch {
            try {
                viewModel.performBackup()
                Toast.makeText(this@MainActivity, "Backup completed", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Backup failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun performRestore() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Restore from Backup")
            .setMessage("This will replace all current data. Continue?")
            .setPositiveButton("Restore") { _, _ ->
                lifecycleScope.launch {
                    try {
                        viewModel.performRestore()
                        Toast.makeText(this@MainActivity, "Restore completed", Toast.LENGTH_SHORT).show()
                        recreate() // Refresh the activity
                    } catch (e: Exception) {
                        Toast.makeText(this@MainActivity, "Restore failed: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun toggleResearchMode() {
        val currentMode = settingsManager.analysisDepth
        val newMode = if (currentMode == AnalysisDepth.COMPREHENSIVE) {
            AnalysisDepth.MODERATE
        } else {
            AnalysisDepth.COMPREHENSIVE
        }
        
        settingsManager.analysisDepth = newMode
        
        val modeText = if (newMode == AnalysisDepth.COMPREHENSIVE) "Research" else "Standard"
        Toast.makeText(this, "$modeText mode enabled", Toast.LENGTH_SHORT).show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == REQUEST_DREAM_INPUT && resultCode == RESULT_OK) {
            // Refresh the dashboard to show the new analysis
            refreshDashboard()
        }
    }

    override fun onResume() {
        super.onResume()
        // Refresh data when returning to main activity
        if (viewModel.isInitialized.value == true) {
            loadDashboardData()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Cleanup services
        dreamAnalysisService.cleanup()
    }
}

// Supporting enums and data classes
enum class ServiceStatus {
    INITIALIZING,
    READY,
    PROCESSING,
    ERROR
}

data class QuickAnalysisOption(
    val type: AnalysisType,
    val title: String,
    val description: String,
    val iconRes: Int,
    val isEnabled: Boolean = true
)
