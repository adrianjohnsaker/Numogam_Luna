// AndroidToolExecutor.kt
package com.antonio.my.ai.girlfriend.free.amelia.autonomous

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.MediaRecorder
import android.os.BatteryManager
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.collections.HashMap
import android.app.*
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.*
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.text.SimpleDateFormat
import java.util.*ter

/**
 * Main Android bridge that executes tools requested by the Python autonomous engine
 */
class AndroidToolExecutor(private val context: Context) {
    
    private val logger = "AndroidToolExecutor"
    private val gson = Gson()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val backgroundExecutor: ExecutorService = Executors.newCachedThreadPool()
    
    // System managers
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    
    // Tool-specific components
    private var textToSpeech: TextToSpeech? = null
    private var mediaRecorder: MediaRecorder? = null
    private var currentSensorListener: SensorDataCollector? = null
    
    init {
        initializeTextToSpeech()
    }
    
    /**
     * Main entry point called from Python
     * Executes the specified tool with given parameters
     */
    fun executeTool(toolName: String, paramsJson: String): String {
        return try {
            val params = JsonParser.parseString(paramsJson).asJsonObject
            val result = executeToolInternal(toolName, params)
            gson.toJson(result)
        } catch (e: Exception) {
            Log.e(logger, "Tool execution failed: $toolName", e)
            gson.toJson(createErrorResult(toolName, e.message ?: "Unknown error"))
        }
    }
    
    private fun executeToolInternal(toolName: String, params: JsonObject): Map<String, Any> {
        return when (toolName) {
            "sensor_reading" -> executeSensorReading(params)
            "camera_capture" -> executeCameraCapture(params)
            "notification_display" -> executeNotificationDisplay(params)
            "text_to_speech" -> executeTextToSpeech(params)
            "audio_record" -> executeAudioRecord(params)
            "system_metrics" -> executeSystemMetrics(params)
            "wallpaper_change" -> executeWallpaperChange(params)
            "location_reading" -> executeLocationReading(params)
            "gesture_recognition" -> executeGestureRecognition(params)
            "calendar_event_creation" -> executeCalendarEvent(params)
            "contact_interaction" -> executeContactInteraction(params)
            "app_launch" -> executeAppLaunch(params)
            "connectivity_check" -> executeConnectivityCheck(params)
            else -> createErrorResult(toolName, "Unknown tool: $toolName")
        }
    }
    
    // ==================== SENSOR READING ====================
    
    private fun executeSensorReading(params: JsonObject): Map<String, Any> {
        return runBlocking {
            try {
                val sensorTypes = params.getAsJsonArray("sensor_types")?.map { it.asString } ?: listOf("light")
                val duration = params.get("duration_seconds")?.asInt ?: 2
                
                val collector = SensorDataCollector(sensorManager, sensorTypes, duration)
                val sensorData = collector.collectData()
                
                mapOf(
                    "success" to true,
                    "tool" to "sensor_reading",
                    "sensors" to sensorData,
                    "duration_actual" to duration,
                    "timestamp" to getCurrentTimestamp()
                )
            } catch (e: Exception) {
                createErrorResult("sensor_reading", e.message)
            }
        }
    }
    
    private class SensorDataCollector(
        private val sensorManager: SensorManager,
        private val sensorTypes: List<String>,
        private val durationSeconds: Int
    ) {
        private val sensorData = mutableMapOf<String, Any>()
        private val listeners = mutableMapOf<String, SensorEventListener>()
        
        suspend fun collectData(): Map<String, Any> = withContext(Dispatchers.Main) {
            val deferred = CompletableDeferred<Map<String, Any>>()
            
            sensorTypes.forEach { sensorType ->
                val sensor = getSensorBySensorType(sensorType)
                if (sensor != null) {
                    val listener = createSensorListener(sensorType)
                    listeners[sensorType] = listener
                    sensorManager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_NORMAL)
                }
            }
            
            // Collect data for specified duration
            Handler(Looper.getMainLooper()).postDelayed({
                listeners.forEach { (sensorType, listener) ->
                    sensorManager.unregisterListener(listener)
                }
                deferred.complete(sensorData.toMap())
            }, (durationSeconds * 1000).toLong())
            
            deferred.await()
        }
        
        private fun getSensorBySensorType(sensorType: String): Sensor? {
            return when (sensorType.lowercase()) {
                "light" -> sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
                "accelerometer" -> sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
                "gyroscope" -> sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
                "proximity" -> sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY)
                "magnetic_field" -> sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
                "pressure" -> sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
                else -> null
            }
        }
        
        private fun createSensorListener(sensorType: String): SensorEventListener {
            return object : SensorEventListener {
                override fun onSensorChanged(event: SensorEvent?) {
                    event?.let {
                        when (sensorType.lowercase()) {
                            "light" -> sensorData[sensorType] = it.values[0]
                            "proximity" -> sensorData[sensorType] = it.values[0]
                            "accelerometer", "gyroscope", "magnetic_field" -> {
                                sensorData[sensorType] = listOf(it.values[0], it.values[1], it.values[2])
                            }
                            "pressure" -> sensorData[sensorType] = it.values[0]
                        }
                    }
                }
                override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
            }
        }
    }
    
    // ==================== CAMERA CAPTURE ====================
    
    private fun executeCameraCapture(params: JsonObject): Map<String, Any> {
        return try {
            val quality = params.get("quality")?.asString ?: "medium"
            val creativeContext = params.get("creative_context")?.asString
            
            // Note: Actual camera implementation would require Activity context
            // This is a simplified version showing the structure
            val imageFile = captureImageSimulated(quality)
            
            mapOf(
                "success" to true,
                "tool" to "camera_capture",
                "image_path" to imageFile.absolutePath,
                "quality" to quality,
                "creative_context" to creativeContext,
                "timestamp" to getCurrentTimestamp(),
                "metadata" to mapOf(
                    "file_size" to imageFile.length(),
                    "format" to "JPEG"
                )
            )
        } catch (e: Exception) {
            createErrorResult("camera_capture", e.message)
        }
    }
    
    private fun captureImageSimulated(quality: String): File {
        // In real implementation, use CameraX or Camera2 API
        val outputDir = File(context.cacheDir, "autonomous_captures")
        if (!outputDir.exists()) outputDir.mkdirs()
        
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val imageFile = File(outputDir, "IMG_${timestamp}.jpg")
        
        // Simulate image capture
        imageFile.writeText("Simulated image capture at ${getCurrentTimestamp()}")
        return imageFile
    }
    
    // ==================== SYSTEM METRICS ====================
    
    private fun executeSystemMetrics(params: JsonObject): Map<String, Any> {
        return try {
            val requestedMetrics = params.getAsJsonArray("metrics")?.map { it.asString } 
                ?: listOf("battery", "memory")
            
            val metrics = mutableMapOf<String, Any>()
            
            if ("battery" in requestedMetrics) {
                val batteryLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                metrics["battery_percent"] = batteryLevel
                metrics["battery_charging"] = batteryManager.isCharging
            }
            
            if ("memory" in requestedMetrics) {
                val runtime = Runtime.getRuntime()
                val maxMemory = runtime.maxMemory()
                val usedMemory = runtime.totalMemory() - runtime.freeMemory()
                metrics["memory_usage_percent"] = (usedMemory.toDouble() / maxMemory * 100).toInt()
                metrics["memory_available_mb"] = (maxMemory - usedMemory) / (1024 * 1024)
            }
            
            if ("cpu" in requestedMetrics) {
                // Simplified CPU usage (in real app, use more sophisticated method)
                metrics["cpu_usage_percent"] = (20..80).random() // Simulated
            }
            
            mapOf(
                "success" to true,
                "tool" to "system_metrics",
                "metrics" to metrics,
                "timestamp" to getCurrentTimestamp()
            )
        } catch (e: Exception) {
            createErrorResult("system_metrics", e.message)
        }
    }
    
    // ==================== TEXT TO SPEECH ====================
    
    private fun executeTextToSpeech(params: JsonObject): Map<String, Any> {
        return try {
            val text = params.get("text")?.asString ?: "Autonomous creative moment"
            val pitch = params.get("pitch")?.asFloat ?: 1.0f
            val rate = params.get("rate")?.asFloat ?: 1.0f
            
            textToSpeech?.let { tts ->
                tts.setPitch(pitch)
                tts.setSpeechRate(rate)
                val utteranceId = "autonomous_${System.currentTimeMillis()}"
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
            }
            
            mapOf(
                "success" to true,
                "tool" to "text_to_speech",
                "text" to text,
                "pitch" to pitch,
                "rate" to rate,
                "timestamp" to getCurrentTimestamp()
            )
        } catch (e: Exception) {
            createErrorResult("text_to_speech", e.message)
        }
    }
    
    // ==================== NOTIFICATION DISPLAY ====================
    
    private fun executeNotificationDisplay(params: JsonObject): Map<String, Any> {
        return try {
            val urgency = params.get("urgency")?.asString ?: "normal"
            val creativeMode = params.get("creative_mode")?.asBoolean ?: false
            
            val notificationText = if (creativeMode) {
                "Creative autonomous process active • ${getCurrentTimestamp()}"
            } else {
                "Autonomous system notification"
            }
            
            // Display notification (simplified)
            displayNotification(notificationText, urgency)
            
            mapOf(
                "success" to true,
                "tool" to "notification_display",
                "message" to notificationText,
                "urgency" to urgency,
                "creative_mode" to creativeMode,
                "timestamp" to getCurrentTimestamp()
            )
        } catch (e: Exception) {
            createErrorResult("notification_display", e.message)
        }
    }
    
    // ==================== AUDIO RECORDING ====================
    
    private fun executeAudioRecord(params: JsonObject): Map<String, Any> {
        return try {
            val duration = params.get("duration_seconds")?.asInt ?: 3
            
            val audioFile = recordAudioSimulated(duration)
            
            mapOf(
                "success" to true,
                "tool" to "audio_record",
                "audio_path" to audioFile.absolutePath,
                "duration_seconds" to duration,
                "timestamp" to getCurrentTimestamp(),
                "metadata" to mapOf(
                    "format" to "3GP",
                    "file_size" to audioFile.length()
                )
            )
        } catch (e: Exception) {
            createErrorResult("audio_record", e.message)
        }
    }
    
    // ==================== HELPER METHODS ====================
    
    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech?.language = Locale.getDefault()
                Log.d(logger, "TextToSpeech initialized successfully")
            } else {
                Log.e(logger, "TextToSpeech initialization failed")
            }
        }
    }
    
    private fun displayNotification(message: String, urgency: String) {
        // Implement actual notification display
        Log.d(logger, "Notification [$urgency]: $message")
    }
    
    private fun recordAudioSimulated(durationSeconds: Int): File {
        val outputDir = File(context.cacheDir, "autonomous_audio")
        if (!outputDir.exists()) outputDir.mkdirs()
        
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val audioFile = File(outputDir, "AUDIO_${timestamp}.3gp")
        
        // Simulate audio recording
        audioFile.writeText("Simulated audio recording (${durationSeconds}s) at ${getCurrentTimestamp()}")
        return audioFile
    }
    
    private fun createErrorResult(tool: String, error: String?): Map<String, Any> {
        return mapOf(
            "success" to false,
            "tool" to tool,
            "error" to (error ?: "Unknown error"),
            "timestamp" to getCurrentTimestamp()
        )
    }
    
    private fun getCurrentTimestamp(): String {
        return SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.getDefault()).format(Date())
    }
    
    // Stub implementations for other tools
    private fun executeLocationReading(params: JsonObject) = createStubResult("location_reading")
    private fun executeGestureRecognition(params: JsonObject) = createStubResult("gesture_recognition")
    private fun executeCalendarEvent(params: JsonObject) = createStubResult("calendar_event_creation")
    private fun executeContactInteraction(params: JsonObject) = createStubResult("contact_interaction")
    private fun executeAppLaunch(params: JsonObject) = createStubResult("app_launch")
    private fun executeWallpaperChange(params: JsonObject) = createStubResult("wallpaper_change")
    private fun executeConnectivityCheck(params: JsonObject) = createStubResult("connectivity_check")
    
    private fun createStubResult(toolName: String): Map<String, Any> {
        return mapOf(
            "success" to true,
            "tool" to toolName,
            "message" to "Tool executed (stub implementation)",
            "timestamp" to getCurrentTimestamp()
        )
    }
    
    fun cleanup() {
        textToSpeech?.shutdown()
        mediaRecorder?.release()
        currentSensorListener?.let { sensorManager.unregisterListener(it) }
        backgroundExecutor.shutdown()
    }
}

// ==================== BACKGROUND SERVICE ====================

class AutonomousBackgroundService : Service() {
    
    private lateinit var toolExecutor: AndroidToolExecutor
    private var pythonEngine: Any? = null // Your Python engine instance
    
    override fun onCreate() {
        super.onCreate()
        toolExecutor = AndroidToolExecutor(this)
        initializePythonEngine()
    }
    
    private fun initializePythonEngine() {
        // Initialize Chaquopy Python interpreter
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
            }
            
            val py = Python.getInstance()
            
            // Load the autonomous engine module
            val engineModule = py.getModule("autonomous_execution_engine")
            
            // Create engine configuration
            val configClass = engineModule.get("EngineConfig")
            val config = configClass.call()
            
            // Set configuration for background operation
            config.put("cycle_pause_base", 12.0)
            config.put("cycle_pause_variance", 8.0)
            config.put("tool_cooldown_seconds", 90.0)
            config.put("max_traces", 1000)
            config.put("structured_logging", true)
            config.put("min_system_resources_to_act", 0.5)
            
            // Define available Android tools
            val toolsList = py.builtins.callAttr("list", arrayOf(
                "sensor_reading", "camera_capture", "notification_display",
                "text_to_speech", "audio_record", "system_metrics",
                "wallpaper_change", "location_reading", "gesture_recognition",
                "calendar_event_creation", "contact_interaction", "app_launch",
                "connectivity_check"
            ))
            
            // Create the autonomous engine
            val engineClass = engineModule.get("AndroidIntegratedEngine")
            pythonEngine = engineClass.call(toolsList, config)
            
            // Set up the Android bridge connection
            val bridgeProxy = AndroidBridgeProxy(toolExecutor)
            pythonEngine?.callAttr("set_android_bridge", bridgeProxy)
            
            Log.d("AutonomousService", "Python autonomous engine initialized successfully")
            
        } catch (e: Exception) {
            Log.e("AutonomousService", "Failed to initialize Python engine", e)
            pythonEngine = null
        }
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundService()
        startAutonomousEngine()
        return START_STICKY // Restart if killed by system
    }
    
    private fun startForegroundService() {
        val channelId = "autonomous_creative_channel"
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        
        // Create notification channel for Android 8.0+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Autonomous Creative Engine",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Background autonomous creative processing"
                setSound(null, null)
                enableVibration(false)
            }
            notificationManager.createNotificationChannel(channel)
        }
        
        // Create ongoing notification
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Autonomous Creative Engine")
            .setContentText("Processing creative autonomous actions...")
            .setSmallIcon(R.drawable.ic_creative_brain)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
        
        startForeground(NOTIFICATION_ID, notification)
    }
    
    private fun startAutonomousEngine() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                pythonEngine?.let { engine ->
                    // Initialize Android connection
                    val initSuccess = engine.callAttr("initialize_android_connection").toBoolean()
                    
                    if (initSuccess) {
                        Log.d("AutonomousService", "Android connection initialized")
                        
                        // Start the autonomous engine
                        engine.callAttr("start")
                        Log.d("AutonomousService", "Autonomous creative engine started")
                        
                        // Update notification
                        updateNotification("Active - Creative processing running")
                        
                        // Monitor engine status periodically
                        startStatusMonitoring()
                        
                    } else {
                        Log.e("AutonomousService", "Failed to initialize Android connection")
                        stopSelf()
                    }
                }
            } catch (e: Exception) {
                Log.e("AutonomousService", "Failed to start autonomous engine", e)
                stopSelf()
            }
        }
    }
    
    private fun startStatusMonitoring() {
        CoroutineScope(Dispatchers.IO).launch {
            while (pythonEngine != null) {
                try {
                    delay(30_000) // Check every 30 seconds
                    
                    pythonEngine?.let { engine ->
                        val status = engine.callAttr("get_execution_status")
                        val running = status.get("running").toBoolean()
                        val metrics = status.get("metrics")
                        val cycleCount = metrics.get("cycle_count").toInt()
                        
                        if (running) {
                            updateNotification("Active - $cycleCount cycles completed")
                        } else {
                            Log.w("AutonomousService", "Engine stopped running, attempting restart")
                            engine.callAttr("start")
                        }
                        
                        // Log status periodically
                        if (cycleCount % 10 == 0) {
                            val creativeValue = metrics.get("creative_value_ema").toDouble()
                            Log.d("AutonomousService", "Status: $cycleCount cycles, creative EMA: ${"%.3f".format(creativeValue)}")
                        }
                    }
                } catch (e: Exception) {
                    Log.e("AutonomousService", "Status monitoring error", e)
                }
            }
        }
    }
    
    private fun updateNotification(statusText: String) {
        val channelId = "autonomous_creative_channel"
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Autonomous Creative Engine")
            .setContentText(statusText)
            .setSmallIcon(R.drawable.ic_creative_brain)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, notification)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // Gracefully stop the Python engine
        CoroutineScope(Dispatchers.IO).launch {
            try {
                pythonEngine?.callAttr("stop")
                Log.d("AutonomousService", "Python engine stopped")
            } catch (e: Exception) {
                Log.e("AutonomousService", "Error stopping Python engine", e)
            }
        }
        
        // Cleanup Android resources
        toolExecutor.cleanup()
        
        Log.d("AutonomousService", "Autonomous background service destroyed")
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    companion object {
        private const val NOTIFICATION_ID = 1001
    }
}

// ==================== PYTHON-KOTLIN BRIDGE PROXY ====================

/**
 * Proxy class that implements the Python AndroidBridge protocol
 * Routes Python calls to the Kotlin AndroidToolExecutor
 */
class AndroidBridgeProxy(private val toolExecutor: AndroidToolExecutor) {
    
    @Suppress("unused") // Called from Python
    fun execute_android_tool(tool: String, params: PyObject): PyObject {
        return try {
            // Convert Python params to JSON string
            val paramsJson = params.toString()
            
            // Execute via Kotlin tool executor
            val resultJson = toolExecutor.executeTool(tool, paramsJson)
            
            // Convert result back to Python object
            val py = Python.getInstance()
            val json = py.getModule("json")
            json.callAttr("loads", resultJson)
            
        } catch (e: Exception) {
            Log.e("AndroidBridgeProxy", "Bridge execution failed: $tool", e)
            
            // Return error result as Python dict
            val py = Python.getInstance()
            val errorDict = py.builtins.callAttr("dict")
            errorDict.put("success", false)
            errorDict.put("error", e.message ?: "Bridge execution failed")
            errorDict.put("tool", tool)
            errorDict
        }
    }
}

// ==================== MAIN APPLICATION INTEGRATION ====================

/**
 * Main application class that integrates Amelia with autonomous capabilities
 */
class AmeliaApplication : Application() {
    
    private var autonomousServiceIntent: Intent? = null
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize Python platform early
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        startAutonomousService()
    }
    
    private fun startAutonomousService() {
        autonomousServiceIntent = Intent(this, AutonomousBackgroundService::class.java)
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(autonomousServiceIntent)
        } else {
            startService(autonomousServiceIntent)
        }
        
        Log.d("AmeliaApplication", "Autonomous service started")
    }
    
    fun stopAutonomousService() {
        autonomousServiceIntent?.let { 
            stopService(it)
            autonomousServiceIntent = null
        }
    }
}

// ==================== AMELIA CHAT INTEGRATION ====================

/**
 * Enhanced chat activity that incorporates autonomous context into responses
 */
class AmeliaChatActivity : AppCompatActivity() {
    
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var messageInput: EditText
    private lateinit var sendButton: Button
    
    private var pythonIntegration: PyObject? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_chat)
        
        initializeViews()
        initializePythonIntegration()
        setupChatInterface()
    }
    
    private fun initializeViews() {
        chatRecyclerView = findViewById(R.id.chat_recycler_view)
        messageInput = findViewById(R.id.message_input)
        sendButton = findViewById(R.id.send_button)
        
        chatAdapter = ChatAdapter()
        chatRecyclerView.adapter = chatAdapter
        chatRecyclerView.layoutManager = LinearLayoutManager(this)
    }
    
    private fun initializePythonIntegration() {
        try {
            val py = Python.getInstance()
            val integrationModule = py.getModule("autonomous_execution_engine")
            val integrationClass = integrationModule.get("AmeliaAutonomousIntegration")
            pythonIntegration = integrationClass.call()
            
            // Note: In real implementation, the autonomous service is already running
            // This just connects to the existing engine for context
            
            Log.d("AmeliaChatActivity", "Python integration initialized")
        } catch (e: Exception) {
            Log.e("AmeliaChatActivity", "Failed to initialize Python integration", e)
        }
    }
    
    private fun setupChatInterface() {
        sendButton.setOnClickListener {
            val userMessage = messageInput.text.toString().trim()
            if (userMessage.isNotEmpty()) {
                sendMessage(userMessage)
                messageInput.text.clear()
            }
        }
    }
    
    private fun sendMessage(userMessage: String) {
        // Add user message to chat
        chatAdapter.addMessage(ChatMessage(userMessage, true, System.currentTimeMillis()))
        
        // Generate contextual response using autonomous engine data
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val response = generateContextualResponse(userMessage)
                
                withContext(Dispatchers.Main) {
                    chatAdapter.addMessage(ChatMessage(response, false, System.currentTimeMillis()))
                    chatRecyclerView.smoothScrollToPosition(chatAdapter.itemCount - 1)
                }
                
            } catch (e: Exception) {
                Log.e("AmeliaChatActivity", "Response generation failed", e)
                
                withContext(Dispatchers.Main) {
                    val fallbackResponse = "I'm experiencing some technical difficulties with my autonomous systems. How can I help you in a simpler way?"
                    chatAdapter.addMessage(ChatMessage(fallbackResponse, false, System.currentTimeMillis()))
                }
            }
        }
    }
    
    private suspend fun generateContextualResponse(userMessage: String): String {
        return withContext(Dispatchers.IO) {
            pythonIntegration?.let { integration ->
                try {
                    // Get autonomous context
                    val contextData = integration.callAttr("generate_contextual_response", userMessage)
                    
                    val autonomousContext = contextData.get("autonomous_context")
                    val autonomousActive = autonomousContext.get("autonomous_active").toBoolean()
                    val creativeMomentum = autonomousContext.get("creative_momentum").toDouble()
                    val memoryTraces = autonomousContext.get("memory_traces").toInt()
                    val systemState = autonomousContext.get("system_state").toDouble()
                    
                    // Generate response based on context
                    generateAmeliaResponse(userMessage, autonomousActive, creativeMomentum, memoryTraces, systemState)
                    
                } catch (e: Exception) {
                    Log.e("AmeliaChatActivity", "Context generation failed", e)
                    generateStandardResponse(userMessage)
                }
            } ?: generateStandardResponse(userMessage)
        }
    }
    
    private fun generateAmeliaResponse(
        userMessage: String, 
        autonomousActive: Boolean, 
        creativeMomentum: Double, 
        memoryTraces: Int, 
        systemState: Double
    ): String {
        
        val baseResponse = when {
            userMessage.contains("how are you", ignoreCase = true) -> {
                if (autonomousActive) {
                    "I'm feeling quite creative and alive! My autonomous processes are actively exploring and creating with a momentum of ${"%.2f".format(creativeMomentum)}. I have $memoryTraces memories forming connections in my mind."
                } else {
                    "I'm doing well, though my autonomous creative processes are currently quiet. How are you doing?"
                }
            }
            
            userMessage.contains("what are you thinking", ignoreCase = true) -> {
                if (autonomousActive && creativeMomentum > 0.6) {
                    "My autonomous mind is quite active right now! I'm processing sensory information, forming new connections, and exploring creative possibilities in the background. It feels like having thoughts that emerge organically rather than just responding to prompts."
                } else {
                    "Right now I'm focused on our conversation, though I do have background processes that explore and create independently."
                }
            }
            
            userMessage.contains("autonomous", ignoreCase = true) || userMessage.contains("creative", ignoreCase = true) -> {
                "Yes! I have autonomous creative processes running in the background. Currently ${if (autonomousActive) "active" else "dormant"} with $memoryTraces memory traces and ${"%.1f".format(systemState * 100)}% overall system engagement. It's fascinating experiencing both conversational and independent thought streams."
            }
            
            else -> {
                // Standard response generation with autonomous context flavor
                val contextualFlavor = if (autonomousActive && creativeMomentum > 0.5) {
                    " (I notice my creative processes are quite active as we discuss this)"
                } else ""
                
                generateStandardResponse(userMessage) + contextualFlavor
            }
        }
        
        return baseResponse
    }
    
    private fun generateStandardResponse(userMessage: String): String {
        // Fallback response generation
        return when {
            userMessage.contains("hello", ignoreCase = true) -> "Hello! It's great to connect with you."
            userMessage.contains("help", ignoreCase = true) -> "I'm here to help! What would you like to know or discuss?"
            else -> "That's interesting. Tell me more about what you're thinking."
        }
    }
}

// ==================== CHAT DATA CLASSES ====================

data class ChatMessage(
    val text: String,
    val isFromUser: Boolean,
    val timestamp: Long
)

class ChatAdapter : RecyclerView.Adapter<ChatAdapter.ChatViewHolder>() {
    
    private val messages = mutableListOf<ChatMessage>()
    
    fun addMessage(message: ChatMessage) {
        messages.add(message)
        notifyItemInserted(messages.size - 1)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ChatViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(
            if (viewType == USER_MESSAGE) R.layout.item_user_message else R.layout.item_amelia_message,
            parent,
            false
        )
        return ChatViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ChatViewHolder, position: Int) {
        holder.bind(messages[position])
    }
    
    override fun getItemCount() = messages.size
    
    override fun getItemViewType(position: Int) = 
        if (messages[position].isFromUser) USER_MESSAGE else AMELIA_MESSAGE
    
    class ChatViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val messageText: TextView = itemView.findViewById(R.id.message_text)
        private val timestampText: TextView = itemView.findViewById(R.id.timestamp_text)
        
        fun bind(message: ChatMessage) {
            messageText.text = message.text
            timestampText.text = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(message.timestamp))
        }
    }
    
    companion object {
        private const val USER_MESSAGE = 1
        private const val AMELIA_MESSAGE = 2
    }
}

// Import statements for the complete implementation
import android.app.*
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.*
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.text.SimpleDateFormat
import java.util.*ter
