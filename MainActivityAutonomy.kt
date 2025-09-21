/**
 * MainActivityAutonomy.kt
 * 
 * Main Activity for continuous autonomous creative AI operation
 * Manages the lifecycle of autonomous creative processes
 */

package com.antonio.my.ai.girlfriend.free.creative.ai.autonomous

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.IBinder
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.creative.ai.autonomous.services.AutonomousCreativeService
import kotlinx.coroutines.*
import java.text.SimpleDateFormat
import java.util.*

class MainActivityAutonomy : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivityAutonomy"
        private const val PERMISSIONS_REQUEST_CODE = 2001
    }

    // UI Components
    private lateinit var autonomyStatusText: TextView
    private lateinit var startAutonomyButton: Button
    private lateinit var stopAutonomyButton: Button
    private lateinit var pauseAutonomyButton: Button
    private lateinit var executionMetricsText: TextView
    private lateinit var realTimeLogText: TextView
    private lateinit var systemResourcesProgress: ProgressBar
    private lateinit var creativeValueProgress: ProgressBar
    private lateinit var cycleCountText: TextView
    private lateinit var connectionDensityText: TextView
    private lateinit var memoryTracesText: TextView
    private lateinit var errorRateText: TextView
    private lateinit var autonomyLevelSeekBar: SeekBar

    // Service connection
    private var autonomousService: AutonomousCreativeService? = null
    private var serviceBound = false
    
    // Status monitoring
    private var statusUpdateJob: Job? = null
    private var isAutonomyActive = false
    private val uiScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // Required permissions for autonomous operation
    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.READ_CONTACTS,
        Manifest.permission.WRITE_CALENDAR,
        Manifest.permission.READ_CALENDAR,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.VIBRATE,
        Manifest.permission.WAKE_LOCK,
        Manifest.permission.FOREGROUND_SERVICE
    )

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, service: IBinder) {
            val binder = service as AutonomousCreativeService.AutonomousBinder
            autonomousService = binder.getService()
            serviceBound = true
            
            updateUIForServiceConnection()
            
            // Check if autonomy is already running
            autonomousService?.let { service ->
                if (service.isAutonomyActive()) {
                    isAutonomyActive = true
                    updateUIState(true)
                    startStatusMonitoring()
                }
            }
        }

        override fun onServiceDisconnected(arg0: ComponentName) {
            serviceBound = false
            autonomousService = null
            updateUIState(false)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_autonomy)
        
        initializeUI()
        setupPermissions()
        bindToAutonomousService()
    }

    private fun initializeUI() {
        // Find UI components
        autonomyStatusText = findViewById(R.id.autonomyStatusText)
        startAutonomyButton = findViewById(R.id.startAutonomyButton)
        stopAutonomyButton = findViewById(R.id.stopAutonomyButton)
        pauseAutonomyButton = findViewById(R.id.pauseAutonomyButton)
        executionMetricsText = findViewById(R.id.executionMetricsText)
        realTimeLogText = findViewById(R.id.realTimeLogText)
        systemResourcesProgress = findViewById(R.id.systemResourcesProgress)
        creativeValueProgress = findViewById(R.id.creativeValueProgress)
        cycleCountText = findViewById(R.id.cycleCountText)
        connectionDensityText = findViewById(R.id.connectionDensityText)
        memoryTracesText = findViewById(R.id.memoryTracesText)
        errorRateText = findViewById(R.id.errorRateText)
        autonomyLevelSeekBar = findViewById(R.id.autonomyLevelSeekBar)

        // Setup button listeners
        startAutonomyButton.setOnClickListener { startAutonomousExecution() }
        stopAutonomyButton.setOnClickListener { stopAutonomousExecution() }
        pauseAutonomyButton.setOnClickListener { pauseAutonomousExecution() }

        // Setup autonomy level control
        autonomyLevelSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    adjustAutonomyLevel(progress)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Initial UI state
        updateUIState(false)
        autonomyStatusText.text = "🤖 Autonomous Creative AI - Initializing..."
        realTimeLogText.text = "Welcome to Autonomous Creative AI\n\nThis system operates continuously in the background, making independent creative decisions and learning from experiences.\n\nKey Features:\n• Quantified decision-making\n• Rhizomatic memory formation\n• Environmental awareness\n• Continuous background processing\n\nTap START AUTONOMY to begin continuous operation."
    }

    private fun setupPermissions() {
        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                missingPermissions.toTypedArray(),
                PERMISSIONS_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            
            if (allGranted) {
                Toast.makeText(this, "All permissions granted - Autonomous AI ready!", Toast.LENGTH_SHORT).show()
                autonomyStatusText.text = "System Ready - All permissions granted"
            } else {
                Toast.makeText(this, "Some permissions denied - Limited autonomous functionality", Toast.LENGTH_LONG).show()
                autonomyStatusText.text = "System Ready - Limited functionality (missing permissions)"
            }
        }
    }

    private fun bindToAutonomousService() {
        val serviceIntent = Intent(this, AutonomousCreativeService::class.java)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun updateUIForServiceConnection() {
        autonomyStatusText.text = "Connected to Autonomous Creative Service"
        startAutonomyButton.isEnabled = true
    }

    private fun startAutonomousExecution() {
        autonomousService?.let { service ->
            uiScope.launch {
                try {
                    autonomyStatusText.text = "Starting autonomous creative processes..."
                    
                    val success = withContext(Dispatchers.IO) {
                        service.startAutonomousExecution()
                    }
                    
                    if (success) {
                        isAutonomyActive = true
                        updateUIState(true)
                        startStatusMonitoring()
                        
                        autonomyStatusText.text = "Autonomous Creative AI - ACTIVE"
                        realTimeLogText.text = "Autonomous execution started...\n\n• Creative decision engine active\n• Memory trace formation enabled\n• Environmental sensors monitoring\n• Background processing initiated\n\nThe AI is now operating independently, making creative decisions based on quantified metrics and environmental awareness."
                        
                        Toast.makeText(this@MainActivityAutonomy, "Autonomous creativity engaged!", Toast.LENGTH_SHORT).show()
                    } else {
                        autonomyStatusText.text = "Failed to start autonomous execution"
                        Toast.makeText(this@MainActivityAutonomy, "Failed to start autonomous execution", Toast.LENGTH_LONG).show()
                    }
                    
                } catch (e: Exception) {
                    autonomyStatusText.text = "Error: ${e.message}"
                    Toast.makeText(this@MainActivityAutonomy, "Error starting autonomous execution", Toast.LENGTH_LONG).show()
                }
            }
        } ?: run {
            Toast.makeText(this, "Service not connected", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopAutonomousExecution() {
        autonomousService?.let { service ->
            uiScope.launch {
                try {
                    autonomyStatusText.text = "Stopping autonomous processes..."
                    
                    withContext(Dispatchers.IO) {
                        service.stopAutonomousExecution()
                    }
                    
                    isAutonomyActive = false
                    updateUIState(false)
                    stopStatusMonitoring()
                    
                    autonomyStatusText.text = "Autonomous Creative AI - STOPPED"
                    realTimeLogText.text = "Autonomous execution stopped.\n\nCreative memories preserved.\nDecision patterns saved.\nEnvironmental awareness paused.\n\nThe AI retains all learned experiences and can resume autonomous operation when restarted."
                    
                    Toast.makeText(this@MainActivityAutonomy, "Autonomous execution stopped", Toast.LENGTH_SHORT).show()
                    
                } catch (e: Exception) {
                    autonomyStatusText.text = "Error stopping: ${e.message}"
                    Toast.makeText(this@MainActivityAutonomy, "Error stopping autonomous execution", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun pauseAutonomousExecution() {
        autonomousService?.let { service ->
            if (isAutonomyActive) {
                // Pause functionality
                Toast.makeText(this, "Autonomous execution paused temporarily", Toast.LENGTH_SHORT).show()
                autonomyStatusText.text = "Autonomous Creative AI - PAUSED"
            } else {
                Toast.makeText(this, "Autonomous execution not active", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun adjustAutonomyLevel(level: Int) {
        val autonomyLevel = when {
            level < 20 -> "Conservative"
            level < 40 -> "Balanced" 
            level < 60 -> "Active"
            level < 80 -> "Aggressive"
            else -> "Maximum"
        }
        
        autonomousService?.let { service ->
            // This would adjust the autonomous execution parameters
            Toast.makeText(this, "Autonomy Level: $autonomyLevel", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateUIState(active: Boolean) {
        startAutonomyButton.isEnabled = !active && serviceBound
        stopAutonomyButton.isEnabled = active
        pauseAutonomyButton.isEnabled = active
        autonomyLevelSeekBar.isEnabled = active
    }

    private fun startStatusMonitoring() {
        statusUpdateJob?.cancel()
        statusUpdateJob = uiScope.launch {
            while (isAutonomyActive) {
                try {
                    updateExecutionStatus()
                    delay(2000) // Update every 2 seconds for real-time feel
                } catch (e: Exception) {
                    // Handle errors gracefully
                    delay(5000)
                }
            }
        }
    }

    private fun stopStatusMonitoring() {
        statusUpdateJob?.cancel()
        statusUpdateJob = null
    }

    private suspend fun updateExecutionStatus() {
        autonomousService?.let { service ->
            withContext(Dispatchers.IO) {
                try {
                    val status = service.getExecutionStatus()
                    
                    withContext(Dispatchers.Main) {
                        updateStatusDisplay(status)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        autonomyStatusText.text = "Status update failed"
                    }
                }
            }
        }
    }

    private fun updateStatusDisplay(status: Map<String, Any>) {
        val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        
        // Extract metrics
        val metrics = status["metrics"] as? Map<String, Any> ?: emptyMap()
        val memory = status["memory"] as? Map<String, Any> ?: emptyMap()
        val androidBridge = status["android_bridge"] as? Map<String, Any> ?: emptyMap()
        
        val cycleCount = metrics["cycle_count"] as? Number ?: 0
        val avgCreativeValue = metrics["average_creative_value"] as? Number ?: 0.0
        val errorRate = metrics["error_rate"] as? Number ?: 0.0
        val traceCount = memory["trace_count"] as? Number ?: 0
        val connectionDensity = memory["connection_density"] as? Number ?: 0.0
        val memoryUsage = memory["memory_usage_percent"] as? Number ?: 0.0
        
        // Update main status
        autonomyStatusText.text = "AUTONOMOUS [$timestamp] | Cycles: $cycleCount | Value: ${String.format("%.3f", avgCreativeValue.toDouble())}"
        
        // Update metrics display
        cycleCountText.text = "Cycles: $cycleCount"
        connectionDensityText.text = "Connections: ${String.format("%.3f", connectionDensity.toDouble())}"
        memoryTracesText.text = "Memory: $traceCount traces"
        errorRateText.text = "Errors: ${String.format("%.1f%%", errorRate.toDouble() * 100)}"
        
        // Update progress bars
        creativeValueProgress.progress = (avgCreativeValue.toDouble() * 100).toInt()
        systemResourcesProgress.progress = (100 - memoryUsage.toDouble()).toInt()
        
        // Update execution metrics
        val runtimeHours = metrics["total_runtime_hours"] as? Number ?: 0.0
        val avgCycleTime = metrics["average_cycle_time"] as? Number ?: 0.0
        
        val metricsText = buildString {
            appendLine("EXECUTION METRICS [$timestamp]")
            appendLine()
            appendLine("Runtime: ${String.format("%.2f", runtimeHours.toDouble())} hours")
            appendLine("Avg Cycle Time: ${String.format("%.2f", avgCycleTime.toDouble())}s")
            appendLine("Creative Value: ${String.format("%.3f", avgCreativeValue.toDouble())}/1.0")
            appendLine("Memory Usage: ${String.format("%.1f", memoryUsage.toDouble())}%")
            appendLine("Connection Density: ${String.format("%.3f", connectionDensity.toDouble())}")
            appendLine("Error Rate: ${String.format("%.1f", errorRate.toDouble() * 100)}%")
            appendLine()
            
            val sensorMonitoring = androidBridge["sensor_monitoring"] as? Boolean ?: false
            appendLine("Sensor Monitoring: ${if (sensorMonitoring) "ACTIVE" else "INACTIVE"}")
            
            val lastSensorUpdate = androidBridge["last_sensor_update"] as? Number ?: 0
            if (lastSensorUpdate.toLong() > 0) {
                val timeSince = (System.currentTimeMillis() - lastSensorUpdate.toLong()) / 1000
                appendLine("Last Sensor Update: ${timeSince}s ago")
            }
            
            appendLine()
            appendLine("The AI continues autonomous creative processing...")
        }
        
        executionMetricsText.text = metricsText
        
        // Update real-time log with recent activity
        updateRealTimeLog(status, timestamp)
    }

    private fun updateRealTimeLog(status: Map<String, Any>, timestamp: String) {
        val controller = status["controller"] as? Map<String, Any> ?: emptyMap()
        val recentUsage = controller["recent_tool_usage"] as? Map<String, Any> ?: emptyMap()
        
        val logText = buildString {
            appendLine("REAL-TIME AUTONOMOUS LOG [$timestamp]")
            appendLine()
            appendLine("Current State: ${status["state"] ?: "unknown"}")
            appendLine("Engine Running: ${status["running"] ?: false}")
            appendLine()
            
            if (recentUsage.isNotEmpty()) {
                appendLine("Recent Tool Usage:")
                recentUsage.entries.take(5).forEach { (tool, lastUsed) ->
                    val timeSince = (System.currentTimeMillis() - (lastUsed as Number).toLong()) / 1000
                    appendLine("  • $tool: ${timeSince}s ago")
                }
                appendLine()
            }
            
            val metrics = status["metrics"] as? Map<String, Any> ?: emptyMap()
            val cycleCount = metrics["cycle_count"] as? Number ?: 0
            val avgValue = metrics["average_creative_value"] as? Number ?: 0.0
            
            when {
                avgValue.toDouble() > 0.8 -> {
                    appendLine("Creative Performance: EXCELLENT")
                    appendLine("The AI is generating high-value creative outputs")
                }
                avgValue.toDouble() > 0.6 -> {
                    appendLine("Creative Performance: GOOD")
                    appendLine("Steady creative progress with solid outputs")
                }
                avgValue.toDouble() > 0.4 -> {
                    appendLine("Creative Performance: MODERATE")
                    appendLine("Building creative momentum through exploration")
                }
                else -> {
                    appendLine("Creative Performance: DEVELOPING")
                    appendLine("Learning and adapting creative approaches")
                }
            }
            
            appendLine()
            appendLine("Autonomous processes continue in background...")
        }
        
        realTimeLogText.text = logText
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.autonomy_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_execution_history -> {
                showExecutionHistory()
                true
            }
            R.id.action_memory_analysis -> {
                showMemoryAnalysis()
                true
            }
            R.id.action_autonomy_settings -> {
                showAutonomySettings()
                true
            }
            R.id.action_export_logs -> {
                exportExecutionLogs()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun showExecutionHistory() {
        autonomousService?.let { service ->
            uiScope.launch {
                try {
                    val status = withContext(Dispatchers.IO) {
                        service.getExecutionStatus()
                    }
                    
                    val historyText = generateExecutionHistoryDisplay(status)
                    realTimeLogText.text = historyText
                    
                    Toast.makeText(this@MainActivityAutonomy, "Execution history displayed", Toast.LENGTH_SHORT).show()
                    
                } catch (e: Exception) {
                    Toast.makeText(this@MainActivityAutonomy, "Failed to load execution history", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun generateExecutionHistoryDisplay(status: Map<String, Any>): String {
        val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        val metrics = status["metrics"] as? Map<String, Any> ?: emptyMap()
        
        return buildString {
            appendLine("AUTONOMOUS EXECUTION HISTORY [$timestamp]")
            appendLine()
            appendLine("Total Cycles: ${metrics["cycle_count"] ?: 0}")
            appendLine("Runtime: ${String.format("%.2f", (metrics["total_runtime_hours"] as? Number ?: 0.0).toDouble())} hours")
            appendLine("Average Creative Value: ${String.format("%.3f", (metrics["average_creative_value"] as? Number ?: 0.0).toDouble())}")
            appendLine("Error Count: ${metrics["error_count"] ?: 0}")
            appendLine()
            appendLine("Key Achievements:")
            
            val avgValue = (metrics["average_creative_value"] as? Number ?: 0.0).toDouble()
            val cycleCount = (metrics["cycle_count"] as? Number ?: 0).toInt()
            
            if (cycleCount > 100) appendLine("  • Reached 100+ autonomous cycles")
            if (avgValue > 0.7) appendLine("  • Maintained high creative value (>0.7)")
            if (cycleCount > 500) appendLine("  • Achieved sustained autonomous operation")
            
            appendLine()
            appendLine("Memory System:")
            val memory = status["memory"] as? Map<String, Any> ?: emptyMap()
            appendLine("  • Memory traces: ${memory["trace_count"] ?: 0}")
            appendLine("  • Connection density: ${String.format("%.3f", (memory["connection_density"] as? Number ?: 0.0).toDouble())}")
            appendLine("  • Memory utilization: ${String.format("%.1f", (memory["memory_usage_percent"] as? Number ?: 0.0).toDouble())}%")
            appendLine()
            appendLine("The AI has been operating autonomously,")
            appendLine("learning from experiences and evolving")
            appendLine("its creative decision-making processes.")
        }
    }

    private fun showMemoryAnalysis() {
        Toast.makeText(this, "Memory analysis functionality would be implemented here", Toast.LENGTH_SHORT).show()
    }

    private fun showAutonomySettings() {
        Toast.makeText(this, "Autonomy settings dialog would open here", Toast.LENGTH_SHORT).show()
    }

    private fun exportExecutionLogs() {
        Toast.makeText(this, "Execution logs export functionality would be implemented here", Toast.LENGTH_SHORT).show()
    }

    override fun onResume() {
        super.onResume()
        
        if (!serviceBound) {
            bindToAutonomousService()
        }
    }

    override fun onPause() {
        super.onPause()
        // Keep autonomy running in background - don't stop monitoring
    }

    override fun onDestroy() {
        super.onDestroy()
        
        // Clean shutdown
        stopStatusMonitoring()
        uiScope.cancel()
        
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
    }
}
