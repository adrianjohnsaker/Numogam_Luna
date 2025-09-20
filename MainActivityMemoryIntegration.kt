// MainActivityMemoryIntegration.kt
// Main Activity integrating Amelia's memory system with Android UI

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader

class MainActivityMemoryIntegration : ComponentActivity() {
    
    companion object {
        private const val TAG = "AmeliaMemoryActivity"
    }
    
    private lateinit var memoryBridge: ChaquopyMemoryBridge
    
    // State variables for UI
    private var isInitialized by mutableStateOf(false)
    private var currentResponseContext by mutableStateOf<ResponseContext?>(null)
    private var memoryClusters by mutableStateOf<List<MemoryCluster>>(emptyList())
    private var rhizomaticConnections by mutableStateOf<Map<String, List<String>>>(emptyMap())
    private var becomingTrajectories by mutableStateOf<List<Pair<String, Double>>>(emptyList())
    private var userInput by mutableStateOf("")
    private var isProcessing by mutableStateOf(false)
    private var statusMessage by mutableStateOf("Initializing Amelia Memory System...")
    
    // File picker for transcript import
    private val documentPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { processTranscriptFile(it) }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize memory bridge
        memoryBridge = ChaquopyMemoryBridge(this)
        
        // Initialize memory system asynchronously
        lifecycleScope.launch {
            initializeAmeliaSystem()
        }
        
        setContent {
            AmeliaMemoryTheme {
                AmeliaMemoryInterface()
            }
        }
    }
    
    private suspend fun initializeAmeliaSystem() {
        withContext(Dispatchers.IO) {
            try {
                statusMessage = "Initializing Python environment..."
                val success = memoryBridge.initializeMemorySystem()
                
                if (success) {
                    statusMessage = "Loading memory clusters..."
                    memoryClusters = memoryBridge.getMemoryClusters()
                    
                    statusMessage = "Analyzing rhizomatic connections..."
                    rhizomaticConnections = memoryBridge.analyzeRhizomaticConnections()
                    
                    statusMessage = "Calculating becoming trajectories..."
                    becomingTrajectories = memoryBridge.getBecomingTrajectories()
                    
                    isInitialized = true
                    statusMessage = "Amelia Memory System Ready"
                    
                } else {
                    statusMessage = "Failed to initialize memory system"
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Initialization error", e)
                statusMessage = "Initialization error: ${e.message}"
            }
        }
    }
    
    private fun processTranscriptFile(uri: Uri) {
        lifecycleScope.launch {
            isProcessing = true
            statusMessage = "Processing transcript file..."
            
            try {
                val content = readFileContent(uri)
                val success = memoryBridge.processTranscriptContent(content)
                
                if (success) {
                    statusMessage = "Transcript integrated successfully"
                    // Refresh memory clusters and connections
                    memoryClusters = memoryBridge.getMemoryClusters()
                    rhizomaticConnections = memoryBridge.analyzeRhizomaticConnections()
                    becomingTrajectories = memoryBridge.getBecomingTrajectories()
                    
                    // Save memory state
                    memoryBridge.saveMemoryState()
                    
                } else {
                    statusMessage = "Failed to process transcript"
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing transcript", e)
                statusMessage = "Error: ${e.message}"
            } finally {
                isProcessing = false
            }
        }
    }
    
    private suspend fun readFileContent(uri: Uri): String = withContext(Dispatchers.IO) {
        contentResolver.openInputStream(uri)?.use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                reader.readText()
            }
        } ?: throw Exception("Unable to read file")
    }
    
    private fun processUserInput() {
        if (userInput.isBlank() || !isInitialized) return
        
        lifecycleScope.launch {
            isProcessing = true
            statusMessage = "Generating memory-informed response context..."
            
            try {
                currentResponseContext = memoryBridge.getResponseContext(userInput)
                
                // Add to conversation memory
                memoryBridge.addConversationMemory("User: $userInput")
                
                statusMessage = "Response context generated"
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing user input", e)
                statusMessage = "Error: ${e.message}"
            } finally {
                isProcessing = false
            }
        }
    }
    
    @Composable
    fun AmeliaMemoryInterface() {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            // Header
            Text(
                text = "Amelia Memory Integration",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Status
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (isInitialized) Color(0xFF4CAF50) else Color(0xFFFF9800)
                )
            ) {
                Text(
                    text = statusMessage,
                    modifier = Modifier.padding(16.dp),
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            if (isInitialized) {
                // Control Panel
                ControlPanel()
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Memory Visualization
                MemoryVisualization()
            }
        }
    }
    
    @Composable
    fun ControlPanel() {
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Deleuzian Memory Interface",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Transcript Import
                Button(
                    onClick = {
                        documentPickerLauncher.launch(arrayOf("text/*", "application/*"))
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isProcessing
                ) {
                    Text("Import Transcript File")
                }
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // User Input
                OutlinedTextField(
                    value = userInput,
                    onValueChange = { userInput = it },
                    label = { Text("Enter dialogue context") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isProcessing
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = { processUserInput() },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isProcessing && userInput.isNotBlank()
                ) {
                    Text("Generate Response Context")
                }
                
                if (isProcessing) {
                    Spacer(modifier = Modifier.height(8.dp))
                    LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }
    }
    
    @Composable
    fun MemoryVisualization() {
        LazyColumn {
            // Response Context Section
            currentResponseContext?.let { context ->
                item {
                    ResponseContextCard(context)
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
            
            // Memory Clusters Section
            item {
                MemoryClustersCard()
                Spacer(modifier = Modifier.height(16.dp))
            }
            
            // Rhizomatic Connections Section
            item {
                RhizomaticConnectionsCard()
                Spacer(modifier = Modifier.height(16.dp))
            }
            
            // Becoming Trajectories Section
            item {
                BecomingTrajectoriesCard()
            }
        }
    }
    
    @Composable
    fun ResponseContextCard(context: ResponseContext) {
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Current Response Context",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF3F51B5)
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Persona Modulation
                Text("Persona Modulation:", fontWeight = FontWeight.Medium)
                Text("• Creativity: ${String.format("%.2f", context.personaModulation.creativityEnhancement)}")
                Text("• Aesthetic: ${String.format("%.2f", context.personaModulation.aestheticSensitivity)}")
                Text("• Philosophical: ${String.format("%.2f", context.personaModulation.philosophicalDepth)}")
                Text("• Emotional: ${String.format("%.2f", context.personaModulation.emotionalResonance)}")
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Active Themes
                if (context.suggestedThemes.isNotEmpty()) {
                    Text("Active Themes:", fontWeight = FontWeight.Medium)
                    Text("${context.suggestedThemes.joinToString(", ")}")
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Contextual Keywords
                if (context.contextualKeywords.isNotEmpty()) {
                    Text("Resonant Keywords:", fontWeight = FontWeight.Medium)
                    Text("${context.contextualKeywords.take(10).joinToString(", ")}")
                }
            }
        }
    }
    
    @Composable
    fun MemoryClustersCard() {
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Memory Clusters (${memoryClusters.size})",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF4CAF50)
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                memoryClusters.forEach { cluster ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = cluster.theme,
                                fontWeight = FontWeight.Medium
                            )
                            Text(
                                text = "Resonance: ${String.format("%.3f", cluster.resonanceLevel)}",
                                fontSize = 12.sp,
                                color = Color.Gray
                            )
                        }
                        Text(
                            text = "${cluster.associatedMemories.size} memories",
                            fontSize = 12.sp,
                            color = Color.Gray
                        )
                    }
                    
                    // Progress bar for resonance level
                    LinearProgressIndicator(
                        progress = cluster.resonanceLevel.toFloat(),
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        color = Color(0xFF2196F3)
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
    
    @Composable
    fun RhizomaticConnectionsCard() {
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Rhizomatic Connections",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFFF5722)
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                rhizomaticConnections.forEach { (theme, connections) ->
                    Text(
                        text = theme,
                        fontWeight = FontWeight.Medium,
                        color = Color(0xFF3F51B5)
                    )
                    connections.forEach { connection ->
                        Text(
                            text = "  → $connection",
                            fontSize = 12.sp,
                            color = Color.Gray
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
    
    @Composable
    fun BecomingTrajectoriesCard() {
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Becoming Trajectories",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF9C27B0)
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                becomingTrajectories.forEach { (theme, intensity) ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = theme,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = String.format("%.3f", intensity),
                            fontSize = 12.sp,
                            color = Color.Gray
                        )
                    }
                    
                    LinearProgressIndicator(
                        progress = (intensity / becomingTrajectories.maxOfOrNull { it.second } ?: 1.0).toFloat(),
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        color = Color(0xFF9C27B0)
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                }
            }
        }
    }
    
    @Composable
    fun AmeliaMemoryTheme(content: @Composable () -> Unit) {
        MaterialTheme(
            colorScheme = lightColorScheme(
                primary = Color(0xFF3F51B5),
                onPrimary = Color.White,
                primaryContainer = Color(0xFF3F51B5),
                onPrimaryContainer = Color.White,
                secondary = Color(0xFF2196F3),
                onSecondary = Color.White,
                secondaryContainer = Color(0xFFE3F2FD),
                onSecondaryContainer = Color(0xFF1976D2),
                tertiary = Color(0xFF9C27B0),
                onTertiary = Color.White,
                tertiaryContainer = Color(0xFFF3E5F5),
                onTertiaryContainer = Color(0xFF7B1FA2),
                error = Color(0xFFD32F2F),
                onError = Color.White,
                errorContainer = Color(0xFFFFEBEE),
                onErrorContainer = Color(0xFFD32F2F),
                background = Color(0xFFF5F5F5),
                onBackground = Color(0xFF212121),
                surface = Color.White,
                onSurface = Color(0xFF212121),
                surfaceVariant = Color(0xFFF5F5F5),
                onSurfaceVariant = Color(0xFF424242),
                outline = Color(0xFFBDBDBD),
                outlineVariant = Color(0xFFE0E0E0),
                scrim = Color(0x80000000),
                inverseSurface = Color(0xFF303030),
                inverseOnSurface = Color(0xFFFAFAFA),
                inversePrimary = Color(0xFF7986CB),
                surfaceDim = Color(0xFFE8E8E8),
                surfaceBright = Color.White,
                surfaceContainerLowest = Color.White,
                surfaceContainerLow = Color(0xFFFCFCFC),
                surfaceContainer = Color(0xFFF6F6F6),
                surfaceContainerHigh = Color(0xFFF0F0F0),
                surfaceContainerHighest = Color(0xFFEBEBEB)
            ),
            typography = Typography(
                displayLarge = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 57.sp,
                    lineHeight = 64.sp,
                    letterSpacing = (-0.25).sp,
                ),
                displayMedium = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 45.sp,
                    lineHeight = 52.sp,
                    letterSpacing = 0.sp,
                ),
                displaySmall = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 36.sp,
                    lineHeight = 44.sp,
                    letterSpacing = 0.sp,
                ),
                headlineLarge = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 32.sp,
                    lineHeight = 40.sp,
                    letterSpacing = 0.sp,
                ),
                headlineMedium = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 28.sp,
                    lineHeight = 36.sp,
                    letterSpacing = 0.sp,
                ),
                headlineSmall = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 24.sp,
                    lineHeight = 32.sp,
                    letterSpacing = 0.sp,
                ),
                titleLarge = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Medium,
                    fontSize = 22.sp,
                    lineHeight = 28.sp,
                    letterSpacing = 0.sp,
                ),
                titleMedium = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Medium,
                    fontSize = 16.sp,
                    lineHeight = 24.sp,
                    letterSpacing = 0.1.sp,
                ),
                titleSmall = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Medium,
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                    letterSpacing = 0.1.sp,
                ),
                bodyLarge = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 16.sp,
                    lineHeight = 24.sp,
                    letterSpacing = 0.5.sp,
                ),
                bodyMedium = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                    letterSpacing = 0.25.sp,
                ),
                bodySmall = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Normal,
                    fontSize = 12.sp,
                    lineHeight = 16.sp,
                    letterSpacing = 0.4.sp,
                ),
                labelLarge = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Medium,
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                    letterSpacing = 0.1.sp,
                ),
                labelMedium = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Medium,
                    fontSize = 12.sp,
                    lineHeight = 16.sp,
                    letterSpacing = 0.5.sp,
                ),
                labelSmall = androidx.compose.ui.text.TextStyle(
                    fontWeight = FontWeight.Medium,
                    fontSize = 11.sp,
                    lineHeight = 16.sp,
                    letterSpacing = 0.5.sp,
                )
            ),
            content = content
        )
    }
    
    override fun onDestroy() {
        super.onDestroy()
        memoryBridge.cleanup()
    }
}
