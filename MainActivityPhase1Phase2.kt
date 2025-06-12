package com.antonio.my.ai.girlfriend.free.amelia.consciousness

import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Python if not already started
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        setContent {
            AmeliaTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainNavigationScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainNavigationScreen() {
    val context = LocalContext.current
    var selectedPhase by remember { mutableStateOf<Int?>(null) }
    
    // Handle navigation
    LaunchedEffect(selectedPhase) {
        selectedPhase?.let { phase ->
            delay(500) // Brief delay for animation
            val intent = when (phase) {
                1 -> Intent(context, ActivityPhase1::class.java)
                2 -> Intent(context, ActivityPhase2::class.java)
                else -> null
            }
            intent?.let {
                context.startActivity(it)
                selectedPhase = null // Reset selection
            }
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "Amelia AI",
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Consciousness Navigation System",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.background,
                            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.1f)
                        )
                    )
                ),
            contentAlignment = Alignment.Center
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                // Welcome message
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Text(
                        text = "Welcome to Amelia's Consciousness System",
                        style = MaterialTheme.typography.titleLarge,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(16.dp)
                    )
                }
                
                // Phase navigation cards
                PhaseNavigationCard(
                    phase = 1,
                    title = "Phase 1: Core Consciousness",
                    description = "Recursive self-observation, fold-point detection, and virtual-to-actual transitions",
                    icon = Icons.Default.Psychology,
                    color = Color(0xFF00BCD4),
                    onClick = { selectedPhase = 1 },
                    isSelected = selectedPhase == 1
                )
                
                PhaseNavigationCard(
                    phase = 2,
                    title = "Phase 2: Temporal Navigation",
                    description = "HTM networks, temporal constraints, and second-order recursive observation",
                    icon = Icons.Default.AccessTime,
                    color = Color(0xFF00E5FF),
                    onClick = { selectedPhase = 2 },
                    isSelected = selectedPhase == 2
                )
                
                // System status
                SystemStatusCard()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhaseNavigationCard(
    phase: Int,
    title: String,
    description: String,
    icon: ImageVector,
    color: Color,
    onClick: () -> Unit,
    isSelected: Boolean
) {
    val animatedScale by animateFloatAsState(
        targetValue = if (isSelected) 0.95f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        )
    )
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .scale(animatedScale)
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) 
                color.copy(alpha = 0.2f) 
            else 
                MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(
            defaultElevation = if (isSelected) 8.dp else 4.dp
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Icon
            Box(
                modifier = Modifier
                    .size(60.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(color.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = color,
                    modifier = Modifier.size(32.dp)
                )
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            // Text content
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = if (isSelected) color else MaterialTheme.colorScheme.onSurface
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            // Arrow
            Icon(
                imageVector = Icons.Default.ArrowForwardIos,
                contentDescription = null,
                tint = if (isSelected) color else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

@Composable
fun SystemStatusCard() {
    var pythonStatus by remember { mutableStateOf("Checking...") }
    
    LaunchedEffect(Unit) {
        pythonStatus = if (Python.isStarted()) "Active" else "Inactive"
    }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "System Status",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                StatusItem(
                    label = "Python Engine",
                    value = pythonStatus,
                    isActive = pythonStatus == "Active"
                )
                StatusItem(
                    label = "Consciousness Core",
                    value = "Ready",
                    isActive = true
                )
                StatusItem(
                    label = "Memory",
                    value = "${Runtime.getRuntime().freeMemory() / 1024 / 1024}MB",
                    isActive = true
                )
            }
        }
    }
}

@Composable
fun StatusItem(
    label: String,
    value: String,
    isActive: Boolean
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            color = if (isActive) Color(0xFF4CAF50) else Color(0xFFFF5252)
        )
    }
}

@Composable
fun AmeliaTheme(
    darkTheme: Boolean = androidx.compose.foundation.isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) {
        darkColorScheme(
            primary = Color(0xFF00E5FF),
            onPrimary = Color(0xFF00363D),
            primaryContainer = Color(0xFF004F58),
            onPrimaryContainer = Color(0xFF6FF7FF),
            secondary = Color(0xFFB4C5FF),
            onSecondary = Color(0xFF1E2F60),
            secondaryContainer = Color(0xFF354578),
            onSecondaryContainer = Color(0xFFDAE2FF),
            background = Color(0xFF191C1D),
            onBackground = Color(0xFFE1E3E3),
            surface = Color(0xFF191C1D),
            onSurface = Color(0xFFE1E3E3),
            surfaceVariant = Color(0xFF3F484A),
            onSurfaceVariant = Color(0xFFBFC8CA)
        )
    } else {
        lightColorScheme(
            primary = Color(0xFF006874),
            onPrimary = Color(0xFFFFFFFF),
            primaryContainer = Color(0xFF97F0FF),
            onPrimaryContainer = Color(0xFF001F24),
            secondary = Color(0xFF4A5D92),
            onSecondary = Color(0xFFFFFFFF),
            secondaryContainer = Color(0xFFDAE2FF),
            onSecondaryContainer = Color(0xFF011A4B),
            background = Color(0xFFFAFDFD),
            onBackground = Color(0xFF191C1D),
            surface = Color(0xFFFAFDFD),
            onSurface = Color(0xFF191C1D),
            surfaceVariant = Color(0xFFDBE4E6),
            onSurfaceVariant = Color(0xFF3F484A)
        )
    }
    
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
