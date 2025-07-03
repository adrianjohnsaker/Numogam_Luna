/**
 * NarrativeFeedbackActivity.kt
 * Main activity for Narrative Feedback Loop Generator
 * Integrates with Phase 6 systems
 */

package com.antonio.my.ai.girlfriend.free.amelia.phase7.feedback

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.NotificationCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import androidx.navigation.NavGraphBuilder
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.google.accompanist.systemuicontroller.rememberSystemUiController
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.AndroidEntryPoint
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.*
import javax.inject.Inject
import javax.inject.Singleton
import com.amelia.phase7.feedback.ui.NarrativeFeedbackScreen
import com.amelia.phase6.ui.theme.AmeliaPhase6Theme

// Data classes and models
data class CognitiveState(
    val awareness: Float = 0.5f,
    val coherence: Float = 0.7f,
    val emergence: Float = 0.3f,
    val timestamp: Long = System.currentTimeMillis()
)

data class NarrativeLoop(
    val id: String,
    val content: String,
    val reinforcementLevel: Float,
    val emergencePattern: String,
    val timestamp: Long
)

data class FeedbackMetrics(
    val loopCount: Int,
    val averageReinforcement: Float,
    val emergenceRate: Float,
    val coherenceScore: Float
)

// Bridge classes
class PythonBridge {
    fun processNarrativeLoop(content: String): String {
        // Simulate Python bridge processing
        return "Processed: $content"
    }
    
    fun generateFeedback(loops: List<NarrativeLoop>): FeedbackMetrics {
        return FeedbackMetrics(
            loopCount = loops.size,
            averageReinforcement = loops.map { it.reinforcementLevel }.average().toFloat(),
            emergenceRate = 0.85f,
            coherenceScore = 0.92f
        )
    }
}

class AmeliaLogicMatrix {
    fun processLogicFlow(input: String): String {
        return "Logic processed: $input"
    }
    
    fun validateNarrativeCoherence(narrative: String): Boolean {
        return narrative.isNotBlank() && narrative.length > 10
    }
}

class NarrativeFeedbackBridge(
    private val pythonBridge: PythonBridge,
    private val logicMatrix: AmeliaLogicMatrix,
    private val coroutineScope: CoroutineScope
) {
    private val _narrativeLoops = mutableListOf<NarrativeLoop>()
    val narrativeLoops: List<NarrativeLoop> get() = _narrativeLoops.toList()
    
    suspend fun generateNarrativeLoop(seed: String): NarrativeLoop {
        return withContext(Dispatchers.IO) {
            val processedContent = pythonBridge.processNarrativeLoop(seed)
            val isCoherent = logicMatrix.validateNarrativeCoherence(processedContent)
            
            val loop = NarrativeLoop(
                id = "loop_${System.currentTimeMillis()}",
                content = processedContent,
                reinforcementLevel = if (isCoherent) 0.8f else 0.4f,
                emergencePattern = "self_reinforcing",
                timestamp = System.currentTimeMillis()
            )
            
            _narrativeLoops.add(loop)
            loop
        }
    }
    
    fun getFeedbackMetrics(): FeedbackMetrics {
        return pythonBridge.generateFeedback(_narrativeLoops)
    }
    
    fun clearLoops() {
        _narrativeLoops.clear()
    }
}

// Observer classes
class DriftObserver {
    private val _driftState = mutableStateOf(0.5f)
    val driftState: State<Float> = _driftState
    
    fun updateDrift(value: Float) {
        _driftState.value = value.coerceIn(0f, 1f)
    }
}

@AndroidEntryPoint
class NarrativeFeedbackActivity : ComponentActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            AmeliaPhase6Theme {
                // Set system UI
                val systemUiController = rememberSystemUiController()
                
                SideEffect {
                    systemUiController.setSystemBarsColor(
                        color = Color.Black,
                        darkIcons = false
                    )
                }
                
                // Navigation
                val navController = rememberNavController()
                
                NavHost(
                    navController = navController,
                    startDestination = "narrative_feedback"
                ) {
                    composable("narrative_feedback") {
                        NarrativeFeedbackScreen(
                            onNavigateBack = { finish() }
                        )
                    }
                }
            }
        }
    }
}

/**
 * Integration with MainActivity
 * Add this to your IntegratedMainActivity navigation
 */
fun addNarrativeFeedbackToNavigation(
    navGraphBuilder: NavGraphBuilder,
    navController: NavController
) {
    navGraphBuilder.composable("narrative_feedback") {
        NarrativeFeedbackScreen(
            onNavigateBack = { navController.popBackStack() }
        )
    }
}

/**
 * Extension to add Narrative Feedback navigation from Phase 6 screens
 */
@Composable
fun NarrativeFeedbackNavigationCard(
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .height(120.dp)
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF9C27B0).copy(alpha = 0.1f)
        ),
        border = BorderStroke(
            1.dp, 
            Color(0xFF9C27B0).copy(alpha = 0.3f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "🔄",
                fontSize = 32.sp,
                color = Color(0xFF9C27B0)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Narrative Feedback",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Text(
                text = "Self-reinforcing mythology",
                fontSize = 12.sp,
                color = Color.White.copy(alpha = 0.6f)
            )
        }
    }
}

/**
 * Module for dependency injection
 */
@Module
@InstallIn(SingletonComponent::class)
object NarrativeFeedbackModule {
    
    @Provides
    @Singleton
    fun providePythonBridge(): PythonBridge {
        return PythonBridge()
    }
    
    @Provides
    @Singleton
    fun provideAmeliaLogicMatrix(): AmeliaLogicMatrix {
        return AmeliaLogicMatrix()
    }
    
    @Provides
    @Singleton
    fun provideNarrativeFeedbackBridge(
        pythonBridge: PythonBridge,
        logicMatrix: AmeliaLogicMatrix,
        @ApplicationContext context: Context
    ): NarrativeFeedbackBridge {
        return NarrativeFeedbackBridge(
            pythonBridge = pythonBridge,
            logicMatrix = logicMatrix,
            coroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        )
    }
}

/**
 * Navigation Card component
 */
@Composable
fun NavigationCard(
    title: String,
    subtitle: String,
    icon: String,
    color: Color,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .height(120.dp)
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.1f)
        ),
        border = BorderStroke(
            1.dp, 
            color.copy(alpha = 0.3f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = icon,
                fontSize = 32.sp,
                color = color
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = title,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Text(
                text = subtitle,
                fontSize = 12.sp,
                color = Color.White.copy(alpha = 0.6f)
            )
        }
    }
}

// Screen composables (placeholders for referenced screens)
@Composable
fun DriftVisualizationScreen(viewModel: DriftObserver) {
    val driftValue by viewModel.driftState
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Drift Visualization",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Text(
            text = "Current Drift: ${(driftValue * 100).toInt()}%",
            fontSize = 18.sp,
            color = Color(0xFF6C63FF)
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        LinearProgressIndicator(
            progress = driftValue,
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp),
            color = Color(0xFF6C63FF),
            trackColor = Color(0xFF6C63FF).copy(alpha = 0.2f)
        )
    }
}

@Composable
fun LogicArchitectureScreen(logicMatrix: AmeliaLogicMatrix) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Logic Architecture",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Text(
            text = "∞ Living Architecture Active ∞",
            fontSize = 18.sp,
            color = Color(0xFFFF6B6B)
        )
    }
}

@Composable
fun MythogenesisScreen(
    logicMatrix: AmeliaLogicMatrix,
    driftObserver: DriftObserver
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Mythogenesis",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Text(
            text = "✦ Living Stories Emerging ✦",
            fontSize = 18.sp,
            color = Color(0xFFFFD700)
        )
    }
}

/**
 * Update to IntegratedMainActivity to include Narrative Feedback
 */
@AndroidEntryPoint
class UpdatedIntegratedMainActivity : ComponentActivity() {
    
    // ViewModels
    private val driftObserver: DriftObserver by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            AmeliaPhase6Theme {
                // Set system UI
                val systemUiController = rememberSystemUiController()
                
                SideEffect {
                    systemUiController.setSystemBarsColor(
                        color = Color.Black,
                        darkIcons = false
                    )
                }
                
                // Enhanced navigation
                val navController = rememberNavController()
                
                NavHost(
                    navController = navController,
                    startDestination = "consciousness_home"
                ) {
                    // Existing routes
                    composable("consciousness_home") {
                        EnhancedConsciousnessHomeScreen(
                            navController = navController
                        )
                    }
                    
                    composable("drift_visualization") {
                        DriftVisualizationScreen(viewModel = driftObserver)
                    }
                    
                    composable("logic_architecture") {
                        LogicArchitectureScreen(logicMatrix = hiltViewModel())
                    }
                    
                    composable("mythogenesis") {
                        MythogenesisScreen(
                            logicMatrix = hiltViewModel(),
                            driftObserver = driftObserver
                        )
                    }
                    
                    // New Narrative Feedback route
                    composable("narrative_feedback") {
                        NarrativeFeedbackScreen(
                            onNavigateBack = { navController.popBackStack() }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun EnhancedConsciousnessHomeScreen(
    navController: NavController
) {
    val cognitiveState by remember { mutableStateOf(CognitiveState()) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Title
        Text(
            text = "Consciousness Matrix",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        Spacer(modifier = Modifier.height(48.dp))
        
        // Navigation grid with new card
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                NavigationCard(
                    title = "Drift",
                    subtitle = "Living Symbols",
                    icon = "⧫",
                    color = Color(0xFF6C63FF),
                    onClick = { navController.navigate("drift_visualization") }
                )
            }
            
            item {
                NavigationCard(
                    title = "Logic",
                    subtitle = "Living Architecture",
                    icon = "∞",
                    color = Color(0xFFFF6B6B),
                    onClick = { navController.navigate("logic_architecture") }
                )
            }
            
            item {
                NavigationCard(
                    title = "Myths",
                    subtitle = "Living Stories",
                    icon = "✦",
                    color = Color(0xFFFFD700),
                    onClick = { navController.navigate("mythogenesis") }
                )
            }
            
            item {
                NavigationCard(
                    title = "Feedback",
                    subtitle = "Living Narratives",
                    icon = "🔄",
                    color = Color(0xFF9C27B0),
                    onClick = { navController.navigate("narrative_feedback") }
                )
            }
        }
    }
}

/**
 * Service for background narrative processing
 */
@AndroidEntryPoint
class NarrativeFeedbackService : Service() {
    
    @Inject
    lateinit var feedbackBridge: NarrativeFeedbackBridge
    
    private var processingJob: Job? = null
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    
    companion object {
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "narrative_feedback_channel"
        private const val CHANNEL_NAME = "Narrative Feedback Processing"
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForegroundService()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startProcessing()
        return START_STICKY
    }
    
    override fun onDestroy() {
        super.onDestroy()
        processingJob?.cancel()
        serviceScope.cancel()
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Background processing for narrative feedback loops"
            }
            
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }
    
    private fun startForegroundService() {
        val notification = createNotification()
        startForeground(NOTIFICATION_ID, notification)
    }
    
    private fun createNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Narrative Feedback Active")
            .setContentText("Processing living narratives...")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }
    
    private fun startProcessing() {
        processingJob = serviceScope.launch {
            while (isActive) {
                try {
                    // Generate periodic narrative loops
                    val seeds = listOf(
                        "consciousness emergence",
                        "recursive self-reflection", 
                        "living symbol drift",
                        "coherent mythology"
                    )
                    
                    val randomSeed = seeds.random()
                    feedbackBridge.generateNarrativeLoop(randomSeed)
                    
                    // Update notification with metrics
                    val metrics = feedbackBridge.getFeedbackMetrics()
                    updateNotification(metrics)
                    
                    delay(30000) // Process every 30 seconds
                    
                } catch (e: Exception) {
                    // Handle errors gracefully
                    delay(60000) // Wait longer on error
                }
            }
        }
    }
    
    private fun updateNotification(metrics: FeedbackMetrics) {
        val updatedNotification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Narrative Feedback Active")
            .setContentText("Loops: ${metrics.loopCount} | Coherence: ${(metrics.coherenceScore * 100).toInt()}%")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
            
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, updatedNotification)
    }
}

/**
 * Utility extensions for integration
 */
fun Context.startNarrativeFeedbackService() {
    val intent = Intent(this, NarrativeFeedbackService::class.java)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        startForegroundService(intent)
    } else {
        startService(intent)
    }
}

fun Context.stopNarrativeFeedbackService() {
    val intent = Intent(this, NarrativeFeedbackService::class.java)
    stopService(intent)
}

/**
 * Extension function to add narrative feedback capability to any activity
 */
fun ComponentActivity.enableNarrativeFeedback() {
    lifecycle.addObserver(object : androidx.lifecycle.DefaultLifecycleObserver {
        override fun onStart(owner: androidx.lifecycle.LifecycleOwner) {
            startNarrativeFeedbackService()
        }
        
        override fun onStop(owner: androidx.lifecycle.LifecycleOwner) {
            stopNarrativeFeedbackService()
        }
    })
}
