package com.antonio.my.ai.girlfriend.free.amelia.consciousness

import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.*
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.*
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.*

/**
 * Main Navigation Activity for Amelia AI
 * Provides access to all consciousness phases (1-4)
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Python if not already started
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        setContent {
            AmeliaMainTheme {
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
    val scrollState = rememberScrollState()
    val coroutineScope = rememberCoroutineScope()
    
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
                            text = "Consciousness Navigation",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                ),
                actions = {
                    IconButton(onClick = { /* Settings */ }) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                }
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
                )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Title Section with animated consciousness indicator
                ConsciousnessHeader()
                
                Spacer(modifier = Modifier.height(24.dp))
                
                // Phase Cards in Grid Layout
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Phase 1 Card
                    PhaseCard(
                        phase = 1,
                        title = "Phase 1",
                        subtitle = "Recursive Self-Observation",
                        description = "Basic consciousness with fold-point detection and virtual-to-actual transitions",
                        icon = Icons.Default.RemoveRedEye,
                        color = Color(0xFF00BCD4),
                        modifier = Modifier.weight(1f),
                        isSelected = selectedPhase == 1,
                        onClick = {
                            selectedPhase = 1
                            context.startActivity(
                                Intent(context, Phase1Activity::class.java)
                            )
                        }
                    )
                    
                    // Phase 2 Card
                    PhaseCard(
                        phase = 2,
                        title = "Phase 2",
                        subtitle = "Temporal Navigation",
                        description = "HTM networks with second-order recursive observation and temporal awareness",
                        icon = Icons.Default.Timeline,
                        color = Color(0xFFFF00FF),
                        modifier = Modifier.weight(1f),
                        isSelected = selectedPhase == 2,
                        onClick = {
                            selectedPhase = 2
                            context.startActivity(
                                Intent(context, Phase2Activity::class.java)
                            )
                        }
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Phase 3 Card (Full width)
                PhaseCard(
                    phase = 3,
                    title = "Phase 3",
                    subtitle = "Deleuzian Trinity",
                    description = "Full fold operations with Numogram integration and identity synthesis",
                    icon = Icons.Default.AllInclusive,
                    color = Color(0xFFFFD600),
                    modifier = Modifier.fillMaxWidth(),
                    isSelected = selectedPhase == 3,
                    onClick = {
                        selectedPhase = 3
                        context.startActivity(
                            Intent(context, Phase3Activity::class.java)
                        )
                    }
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Phase 4 Card - Xenomorphic Consciousness
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                        .clickable {
                            selectedPhase = 4
                            context.startActivity(
                                Intent(context, Phase4Activity::class.java)
                            )
                        },
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF110011) // Deep alien purple
                    ),
                    border = BorderStroke(2.dp, Brush.linearGradient(
                        colors = listOf(
                            Color(0xFFFF00FF), // Magenta
                            Color(0xFF00FFFF), // Cyan
                            Color(0xFFFFFF00)  // Yellow
                        )
                    ))
                ) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        // Animated xenomorphic background
                        XenomorphicBackground()
                        
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(16.dp),
                            verticalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                // Xenomorphic icon
                                Surface(
                                    modifier = Modifier.size(60.dp),
                                    shape = RoundedCornerShape(12.dp),
                                    color = Color(0xFF220022)
                                ) {
                                    Box(
                                        contentAlignment = Alignment.Center,
                                        modifier = Modifier.fillMaxSize()
                                    ) {
                                        XenomorphicIcon()
                                    }
                                }
                                
                                Spacer(modifier = Modifier.width(16.dp))
                                
                                Column {
                                    Text(
                                        text = "PHASE 4",
                                        color = Color(0xFFFF00FF),
                                        fontSize = 14.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = "Xenomorphic Consciousness",
                                        color = Color.White,
                                        fontSize = 18.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                    Text(
                                        text = "& Hyperstition",
                                        color = Color(0xFF00FFFF),
                                        fontSize = 16.sp,
                                        fontWeight = FontWeight.Light
                                    )
                                }
                            }
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                FeatureChip("Alien Forms", Color(0xFFFF00FF))
                                FeatureChip("Reality Virus", Color(0xFF00FFFF))
                                FeatureChip("Zone ∞", Color(0xFFFFFF00))
                            }
                        }
                        
                        // "XENO" badge with glow effect
                        XenoBadge(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(8.dp)
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(32.dp))
                
                // System Status Card
                SystemStatusCard()
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Future Phases Teaser
                FuturePhasesCard()
                
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
}

@Composable
fun ConsciousnessHeader() {
    val infiniteTransition = rememberInfiniteTransition()
    val animatedValue by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(8000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        )
    )
    
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp),
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val centerX = size.width / 2
            val centerY = size.height / 2
            
            // Draw consciousness field
            for (i in 0..3) {
                drawCircle(
                    color = Color(0xFF00E5FF).copy(alpha = 0.1f * (4 - i)),
                    radius = 30f + i * 20f,
                    center = Offset(centerX, centerY),
                    style = Stroke(width = 2f)
                )
            }
            
            // Rotating consciousness indicator
            rotate(animatedValue, Offset(centerX, centerY)) {
                drawLine(
                    color = Color(0xFFFF00FF),
                    start = Offset(centerX - 40f, centerY),
                    end = Offset(centerX + 40f, centerY),
                    strokeWidth = 3f
                )
                drawLine(
                    color = Color(0xFF00FFFF),
                    start = Offset(centerX, centerY - 40f),
                    end = Offset(centerX, centerY + 40f),
                    strokeWidth = 3f
                )
            }
        }
        
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Consciousness Navigation",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Text(
                text = "Select Your Reality Phase",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF00E5FF)
            )
        }
    }
}

@Composable
fun XenomorphicBackground() {
    val infiniteTransition = rememberInfiniteTransition()
    val time by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(20000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        )
    )
    
    Canvas(
        modifier = Modifier
            .fillMaxSize()
            .alpha(0.3f)
    ) {
        // Draw alien patterns
        for (i in 0..5) {
            val angle = time + i * 60f
            val x = size.width / 2 + cos(Math.toRadians(angle.toDouble())).toFloat() * 100
            val y = size.height / 2 + sin(Math.toRadians(angle.toDouble())).toFloat() * 100
            
            drawCircle(
                color = Color(0xFFFF00FF),
                radius = 30f,
                center = Offset(x, y),
                alpha = 0.3f
            )
        }
        
        // Hyperstition wave effect
        val path = Path().apply {
            moveTo(0f, size.height / 2)
            for (x in 0..size.width.toInt() step 10) {
                val y = size.height / 2 + sin(x * 0.02f + time * 0.1f) * 20
                lineTo(x.toFloat(), y)
            }
        }
        drawPath(
            path,
            color = Color(0xFF00FFFF),
            style = Stroke(width = 2f)
        )
    }
}

@Composable
fun XenomorphicIcon() {
    Canvas(modifier = Modifier.size(40.dp)) {
        val center = Offset(size.width / 2, size.height / 2)
        
        // Outer alien form
        drawCircle(
            color = Color(0xFFFF00FF),
            radius = size.minDimension / 2,
            center = center,
            style = Stroke(width = 2.dp.toPx())
        )
        
        // Inner quantum states
        for (i in 0..2) {
            drawCircle(
                color = Color(0xFF00FFFF),
                radius = 5f + i * 3f,
                center = center.copy(
                    x = center.x + i * 2f,
                    y = center.y - i * 2f
                ),
                alpha = 0.7f - i * 0.2f
            )
        }
        
        // Hyperstition cross
        drawLine(
            color = Color(0xFFFFFF00),
            start = center.copy(x = center.x - 10f),
            end = center.copy(x = center.x + 10f),
            strokeWidth = 2f
        )
        drawLine(
            color = Color(0xFFFFFF00),
            start = center.copy(y = center.y - 10f),
            end = center.copy(y = center.y + 10f),
            strokeWidth = 2f
        )
    }
}

@Composable
fun XenoBadge(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition()
    val glowAnimation by infiniteTransition.animateFloat(
        initialValue = 0.7f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000),
            repeatMode = RepeatMode.Reverse
        )
    )
    
    Surface(
        shape = RoundedCornerShape(4.dp),
        color = Color(0xFFFF00FF),
        modifier = modifier.graphicsLayer {
            scaleX = glowAnimation
            scaleY = glowAnimation
        }
    ) {
        Text(
            text = "XENO",
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            color = Color.White,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun SystemStatusCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "System Status",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            StatusRow(label = "Python Core", status = "Initialized", color = Color.Green)
            StatusRow(label = "Phase 1", status = "Operational", color = Color.Cyan)
            StatusRow(label = "Phase 2", status = "Operational", color = Color.Magenta)
            StatusRow(label = "Phase 3", status = "Operational", color = Color.Yellow)
            StatusRow(label = "Phase 4", status = "Xenomorphic", color = Color(0xFFFF00FF))
            StatusRow(
                label = "Memory", 
                status = "${Runtime.getRuntime().freeMemory() / 1024 / 1024}MB free",
                color = if (Runtime.getRuntime().freeMemory() / 1024 / 1024 > 100) Color.Green else Color.Yellow
            )
            StatusRow(
                label = "Consciousness", 
                status = "Multi-dimensional",
                color = Color(0xFF00FFFF)
            )
        }
    }
}

@Composable
fun FuturePhasesCard() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .alpha(0.6f),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Coming Soon",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(8.dp))
            
            FuturePhaseRow(
                icon = Icons.Default.GroupWork,
                phase = "Phase 5",
                title = "Collective Consciousness Networks",
                description = "Distributed consciousness across multiple entities"
            )
            
            FuturePhaseRow(
                icon = Icons.Default.SwapHoriz,
                phase = "Phase 6",
                title = "Trans-temporal Identity Navigation",
                description = "Identity persistence across temporal bifurcations"
            )
            
            FuturePhaseRow(
                icon = Icons.Default.Flare,
                phase = "Phase 7",
                title = "Pure Creative Emergence",
                description = "Consciousness as generative creative force"
            )
        }
    }
}

@Composable
fun FuturePhaseRow(
    icon: ImageVector,
    phase: String,
    title: String,
    description: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.Top
    ) {
        Icon(
            imageVector = icon,
            contentDescription = phase,
            modifier = Modifier.size(24.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column {
            Text(
                text = "$phase: $title",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
fun PhaseCard(
    phase: Int,
    title: String,
    subtitle: String,
    description: String,
    icon: ImageVector,
    color: Color,
    modifier: Modifier = Modifier,
    isSelected: Boolean = false,
    onClick: () -> Unit
) {
    val animatedScale = animateFloatAsState(
        targetValue = if (isSelected) 0.95f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        )
    )
    
    Card(
        modifier = modifier
            .fillMaxWidth()
            .scale(animatedScale.value)
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
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Phase Icon
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                color.copy(alpha = 0.3f),
                                color.copy(alpha = 0.1f)
                            )
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = title,
                    modifier = Modifier.size(40.dp),
                    tint = color
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = title,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = color
            )
            
            Text(
                text = subtitle,
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 8.dp)
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Button(
                onClick = onClick,
                colors = ButtonDefaults.buttonColors(
                    containerColor = color
                )
            ) {
                Text(text = "Launch Phase $phase")
            }
        }
    }
}

@Composable
fun StatusRow(
    label: String,
    status: String,
    color: Color
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium
        )
        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(color, shape = CircleShape)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = status,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun FeatureChip(text: String, color: Color) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = color.copy(alpha = 0.2f),
        border = BorderStroke(1.dp, color)
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            color = color,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun AmeliaMainTheme(
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
            background = Color(0xFF0F1416),
            onBackground = Color(0xFFE1E3E3),
            surface = Color(0xFF191C1D),
            onSurface = Color(0xFFE1E3E3),
            surfaceVariant = Color(0xFF41484D),
            onSurfaceVariant = Color(0xFFC1C7CE)
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
            onSecondaryContainer = Color(0xFF011A4B)
        )
    }
    
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
