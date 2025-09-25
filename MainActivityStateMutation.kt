// MainActivityStateMutation.kt
package com.antonio.my.ai.girlfriend.free.amelia

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.yourapp.amelia.ui.theme.*
import kotlinx.coroutines.delay
import kotlin.math.*

class MainActivityStateMutation : ComponentActivity() {
    
    companion object {
        private const val TAG = "MainActivityStateMutation"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        Log.d(TAG, "Initializing Amelia State Mutation Activity")
        
        // Initialize Chaquopy Python
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
            Log.d(TAG, "Python environment started")
        }
        
        setContent {
            AmeliaTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val stateManager: AmeliaStateManager = viewModel()
                    AmeliaStateMutationApp(stateManager)
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Amelia State Mutation Activity destroyed")
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AmeliaStateMutationApp(stateManager: AmeliaStateManager) {
    val state by stateManager.currentState.collectAsState()
    val mutationResults by stateManager.mutationResults.collectAsState()
    val suggestions by stateManager.suggestions.collectAsState()
    val isLoading by stateManager.isLoading.collectAsState()
    
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("State", "Mutations", "Analytics", "Settings")
    
    LaunchedEffect(Unit) {
        stateManager.refreshSuggestions()
        // Auto-refresh every 30 seconds
        while (true) {
            delay(30000)
            stateManager.refreshState()
        }
    }
    
    Scaffold(
        topBar = {
            AmeliaTopBar(
                title = "Amelia State Mutation",
                isLoading = isLoading,
                onRefresh = { stateManager.refreshState() }
            )
        },
        bottomBar = {
            AmeliaBottomNavigation(
                selectedTab = selectedTab,
                tabs = tabs,
                onTabSelected = { selectedTab = it }
            )
        },
        floatingActionButton = {
            if (selectedTab == 1) { // Mutations tab
                FloatingActionButton(
                    onClick = { stateManager.boostCreativity() },
                    containerColor = AmeliaStatusColors.Creative
                ) {
                    Icon(
                        Icons.Default.Auto,
                        contentDescription = "Quick Creativity Boost",
                        tint = Color.White
                    )
                }
            }
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when (selectedTab) {
                0 -> StateOverviewScreen(state, stateManager)
                1 -> MutationsScreen(suggestions, mutationResults, stateManager)
                2 -> AnalyticsScreen(state, mutationResults)
                3 -> SettingsScreen(stateManager)
            }
            
            // Loading overlay
            if (isLoading) {
                LoadingOverlay()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AmeliaTopBar(
    title: String,
    isLoading: Boolean,
    onRefresh: () -> Unit
) {
    TopAppBar(
        title = {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.headlineSmall
                )
                if (isLoading) {
                    Spacer(modifier = Modifier.width(8.dp))
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp
                    )
                }
            }
        },
        actions = {
            IconButton(onClick = onRefresh) {
                Icon(
                    Icons.Default.Refresh,
                    contentDescription = "Refresh State"
                )
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
        )
    )
}

@Composable
fun AmeliaBottomNavigation(
    selectedTab: Int,
    tabs: List<String>,
    onTabSelected: (Int) -> Unit
) {
    val icons = listOf(
        Icons.Default.Dashboard,
        Icons.Default.Science,
        Icons.Default.Analytics,
        Icons.Default.Settings
    )
    
    NavigationBar {
        tabs.forEachIndexed { index, tab ->
            NavigationBarItem(
                icon = {
                    Icon(
                        icons[index],
                        contentDescription = tab
                    )
                },
                label = { Text(tab) },
                selected = selectedTab == index,
                onClick = { onTabSelected(index) }
            )
        }
    }
}

@Composable
fun StateOverviewScreen(
    state: AmeliaState?,
    stateManager: AmeliaStateManager
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            WelcomeCard()
        }
        
        state?.let { currentState ->
            item {
                CreativityVisualizationCard(currentState.creativityMetrics)
            }
            
            item {
                ParametersOverviewCard(currentState.parameters)
            }
            
            item {
                SystemHealthCard(currentState)
            }
            
            item {
                QuickActionsCard(
                    onBoostCreativity = { stateManager.boostCreativity() },
                    onIncreaseExploration = { stateManager.increaseExploration() },
                    onReset = { stateManager.resetParameters() }
                )
            }
        } ?: item {
            LoadingStateCard()
        }
    }
}

@Composable
fun MutationsScreen(
    suggestions: List<MutationSuggestion>,
    mutationResults: List<MutationResult>,
    stateManager: AmeliaStateManager
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        if (suggestions.isNotEmpty()) {
            item {
                Text(
                    text = "AI Suggestions",
                    style = MaterialTheme.typography.headlineMedium,
                    color = AmeliaStatusColors.Creative
                )
            }
            
            items(suggestions) { suggestion ->
                AnimatedSuggestionCard(
                    suggestion = suggestion,
                    onApply = {
                        stateManager.executeMutation(
                            MutationType.valueOf(suggestion.type.uppercase()),
                            suggestion.target,
                            suggestion.suggestedValue,
                            suggestion.reason,
                            "Applied AI suggestion"
                        )
                    }
                )
            }
        }
        
        item {
            Divider(modifier = Modifier.padding(vertical = 8.dp))
        }
        
        item {
            AdvancedMutationCard(stateManager)
        }
        
        if (mutationResults.isNotEmpty()) {
            item {
                Text(
                    text = "Recent Mutations",
                    style = MaterialTheme.typography.headlineMedium,
                    modifier = Modifier.padding(top = 16.dp)
                )
            }
            
            items(mutationResults.take(10)) { result ->
                AnimatedMutationResultCard(result)
            }
        }
    }
}

@Composable
fun AnalyticsScreen(
    state: AmeliaState?,
    mutationResults: List<MutationResult>
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Text(
                text = "Analytics Dashboard",
                style = MaterialTheme.typography.headlineMedium
            )
        }
        
        state?.let { currentState ->
            item {
                CreativityTrendChart(currentState.creativityMetrics, mutationResults)
            }
            
            item {
                MutationSuccessRateCard(mutationResults)
            }
            
            item {
                ParameterEvolutionChart(mutationResults)
            }
        }
    }
}

@Composable
fun SettingsScreen(stateManager: AmeliaStateManager) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Text(
                text = "Settings",
                style = MaterialTheme.typography.headlineMedium
            )
        }
        
        item {
            SystemControlsCard(stateManager)
        }
        
        item {
            SafetySettingsCard()
        }
        
        item {
            AboutCard()
        }
    }
}

@Composable
fun WelcomeCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                text = "🧠 Amelia State Mutation",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onPrimaryContainer
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Monitor and enhance Amelia's autonomous creativity through real-time parameter mutation and feedback loops.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
            )
        }
    }
}

@Composable
fun CreativityVisualizationCard(metrics: CreativityMetrics) {
    val animatedCreativity by animateFloatAsState(
        targetValue = metrics.currentCreativity.toFloat(),
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        )
    )
    
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                text = "Creativity Visualization",
                style = MaterialTheme.typography.titleLarge,
                color = AmeliaStatusColors.Creative
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Circular creativity meter
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp),
                contentAlignment = Alignment.Center
            ) {
                CreativityMeter(
                    value = animatedCreativity,
                    modifier = Modifier.size(180.dp)
                )
                
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "${(animatedCreativity * 100).toInt()}%",
                        style = MaterialTheme.typography.headlineLarge,
                        fontWeight = FontWeight.Bold,
                        color = AmeliaStatusColors.Creative
                    )
                    Text(
                        text = "Creativity Level",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Secondary metrics
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                MetricChip("Risk", metrics.riskTaking, AmeliaStatusColors.Warning)
                MetricChip("Coherence", metrics.coherenceScore, AmeliaStatusColors.Success)
                MetricChip("Novelty", metrics.noveltyScore, AmeliaStatusColors.Info)
            }
        }
    }
}

@Composable
fun CreativityMeter(
    value: Float,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        val center = Offset(size.width / 2f, size.height / 2f)
        val radius = minOf(size.width, size.height) / 2f - 20.dp.toPx()
        
        // Background arc
        drawArc(
            color = Color.Gray.copy(alpha = 0.2f),
            startAngle = 135f,
            sweepAngle = 270f,
            useCenter = false,
            style = Stroke(width = 12.dp.toPx()),
            size = androidx.compose.ui.geometry.Size(radius * 2, radius * 2),
            topLeft = Offset(center.x - radius, center.y - radius)
        )
        
        // Progress arc
        val sweepAngle = 270f * value
        drawArc(
            brush = Brush.sweepGradient(
                colors = listOf(
                    AmeliaStatusColors.Error,
                    AmeliaStatusColors.Warning,
                    AmeliaStatusColors.Success
                ),
                center = center
            ),
            startAngle = 135f,
            sweepAngle = sweepAngle,
            useCenter = false,
            style = Stroke(width = 12.dp.toPx()),
            size = androidx.compose.ui.geometry.Size(radius * 2, radius * 2),
            topLeft = Offset(center.x - radius, center.y - radius)
        )
    }
}

@Composable
fun MetricChip(
    label: String,
    value: Double,
    color: Color
) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = color.copy(alpha = 0.1f),
        border = ButtonDefaults.outlinedButtonBorder.copy(brush = Brush.linearGradient(listOf(color, color)))
    ) {
        Column(
            modifier = Modifier.padding(12.dp, 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = String.format("%.2f", value),
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun AnimatedSuggestionCard(
    suggestion: MutationSuggestion,
    onApply: () -> Unit
) {
    var isExpanded by remember { mutableStateOf(false) }
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = AmeliaStatusColors.Creative.copy(alpha = 0.05f)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "💡 ${suggestion.target}",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "→ ${String.format("%.3f", suggestion.suggestedValue)}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = AmeliaStatusColors.Creative
                    )
                }
                
                Row {
                    IconButton(onClick = { isExpanded = !isExpanded }) {
                        Icon(
                            if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = if (isExpanded) "Collapse" else "Expand"
                        )
                    }
                    
                    Button(
                        onClick = onApply,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = AmeliaStatusColors.Creative
                        )
                    ) {
                        Text("Apply")
                    }
                }
            }
            
            AnimatedVisibility(visible = isExpanded) {
                Column(modifier = Modifier.padding(top = 12.dp)) {
                    Divider(modifier = Modifier.padding(vertical = 8.dp))
                    Text(
                        text = "Reasoning:",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = suggestion.reason,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun LoadingOverlay() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.3f)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator(
                    color = AmeliaStatusColors.Creative
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Processing mutation...",
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

// Additional helper composables would continue here...
// (SystemControlsCard, CreativityTrendChart, etc.)

@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    AmeliaTheme {
        WelcomeCard()
    }
}
