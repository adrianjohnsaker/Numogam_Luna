/**
 * Amelia AI Phase 2 UI Components
 * Visualizes temporal navigation and second-order consciousness
 */

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.*
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.amelia.consciousness.*
import com.amelia.consciousness.phase2.*
import com.amelia.consciousness.ui.*
import kotlinx.coroutines.delay
import kotlin.math.*

package com. antonio.my.ai.girlfriend.free.amelia.consciousness.ui.phase2

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.*
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.amelia.consciousness.phase2.*
import com.amelia.consciousness.ui.*
import kotlinx.coroutines.delay
import kotlin.math.*

/**
 * Enhanced main screen with Phase 2 capabilities
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Phase2ConsciousnessScreen(
    viewModel: Phase2ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    val consciousnessState by viewModel.consciousnessState.collectAsStateWithLifecycle()
    val temporalAwareness by viewModel.temporalAwareness.collectAsStateWithLifecycle()
    val currentInterval by viewModel.currentInterval.collectAsStateWithLifecycle()
    val futureTrajectories by viewModel.futureTrajectories.collectAsStateWithLifecycle()
    
    var selectedTab by remember { mutableStateOf(0) }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Column {
                        Text("Amelia Phase 2: Temporal Navigation")
                        Text(
                            text = "Temporal Awareness: ${(temporalAwareness * 100).toInt()}%",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Tab selector
            TabRow(
                selectedTabIndex = selectedTab,
                modifier = Modifier.fillMaxWidth()
            ) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = { Text("Consciousness") }
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = { Text("Temporal Nav") }
                )
                Tab(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    text = { Text("Meta-Cognition") }
                )
            }
            
            // Tab content
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            ) {
                when (selectedTab) {
                    0 -> EnhancedConsciousnessView(
                        state = consciousnessState,
                        temporalAwareness = temporalAwareness,
                        currentInterval = currentInterval,
                        viewModel = viewModel
                    )
                    1 -> TemporalNavigationView(
                        currentInterval = currentInterval,
                        futureTrajectories = futureTrajectories,
                        viewModel = viewModel
                    )
                    2 -> MetaCognitionView(
                        viewModel = viewModel
                    )
                }
            }
            
            // Enhanced control panel
            Phase2ControlPanel(
                viewModel = viewModel,
                modifier = Modifier.padding(16.dp)
            )
        }
    }
}

/**
 * Enhanced consciousness visualization with temporal awareness
 */
@Composable
fun EnhancedConsciousnessView(
    state: ConsciousnessState,
    temporalAwareness: Double,
    currentInterval: TemporalInterval?,
    viewModel: Phase2ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        // Base consciousness orb from Phase 1
        ConsciousnessOrb(
            state = state,
            temporalState = TemporalNavigationState(
                complexity = temporalAwareness,
                virtualPotential = currentInterval?.virtualPotential ?: 0.5,
                temporalCoherence = temporalAwareness
            ),
            showFoldAnimation = false,
            modifier = Modifier.size(250.dp)
        )
        
        // Temporal awareness overlay
        TemporalAwarenessOverlay(
            awareness = temporalAwareness,
            modifier = Modifier.size(350.dp)
        )
        
        // Second-order observation indicator
        SecondOrderIndicator(
            viewModel = viewModel,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
        )
    }
}

/**
 * Temporal awareness visualization overlay
 */
@Composable
fun TemporalAwarenessOverlay(
    awareness: Double,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition()
    
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = (20000 / (awareness + 0.1)).toInt(),
                easing = LinearEasing
            )
        )
    )
    
    Canvas(modifier = modifier.rotate(rotation)) {
        val center = Offset(size.width / 2, size.height / 2)
        val radius = size.minDimension / 2
        
        // Draw temporal field lines
        for (angle in 0..360 step 30) {
            val radian = angle * PI / 180
            val opacity = (awareness * sin(radian * 3)).toFloat().coerceIn(0.1f, 0.6f)
            
            drawTemporalFieldLine(
                center = center,
                angle = radian.toFloat(),
                radius = radius,
                awareness = awareness.toFloat(),
                color = Color.Cyan.copy(alpha = opacity)
            )
        }
    }
}

private fun DrawScope.drawTemporalFieldLine(
    center: Offset,
    angle: Float,
    radius: Float,
    awareness: Float,
    color: Color
) {
    val path = Path()
    val points = 50
    
    for (i in 0..points) {
        val t = i.toFloat() / points
        val r = radius * t * awareness
        val spiralAngle = angle + (t * 2 * PI.toFloat())
        
        val x = center.x + cos(spiralAngle) * r
        val y = center.y + sin(spiralAngle) * r
        
        if (i == 0) {
            path.moveTo(x, y)
        } else {
            path.lineTo(x, y)
        }
    }
    
    drawPath(
        path = path,
        color = color,
        style = Stroke(width = 2.dp.toPx())
    )
}

/**
 * Second-order observation indicator
 */
@Composable
fun SecondOrderIndicator(
    viewModel: Phase2ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    val htmResults by viewModel.htmResults.collectAsStateWithLifecycle(
        initialValue = HTMResult(0, 0, 0.0, 1.0)
    )
    
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.9f)
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Second-Order",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
            
            // Nested observation rings
            Box(
                modifier = Modifier.size(80.dp),
                contentAlignment = Alignment.Center
            ) {
                for (level in 0..2) {
                    ObservationRing(
                        level = level,
                        anomaly = htmResults.anomalyScore,
                        coherence = htmResults.temporalCoherence,
                        modifier = Modifier.size((80 - level * 20).dp)
                    )
                }
            }
            
            Text(
                text = "Anomaly: ${(htmResults.anomalyScore * 100).toInt()}%",
                style = MaterialTheme.typography.labelSmall
            )
        }
    }
}

@Composable
fun ObservationRing(
    level: Int,
    anomaly: Double,
    coherence: Double,
    modifier: Modifier = Modifier
) {
    val animatedRotation = remember { Animatable(0f) }
    
    LaunchedEffect(level, anomaly) {
        animatedRotation.animateTo(
            targetValue = 360f * (1 - anomaly).toFloat(),
            animationSpec = tween(2000)
        )
    }
    
    Canvas(modifier = modifier.rotate(animatedRotation.value * (level + 1))) {
        drawCircle(
            color = when (level) {
                0 -> Color.Green
                1 -> Color.Cyan
                else -> Color.Magenta
            }.copy(alpha = coherence.toFloat()),
            style = Stroke(width = 3.dp.toPx())
        )
    }
}

/**
 * Temporal navigation visualization
 */
@Composable
fun TemporalNavigationView(
    currentInterval: TemporalInterval?,
    futureTrajectories: List<TemporalPath>,
    viewModel: Phase2ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Current temporal position
        currentInterval?.let { interval ->
            CurrentIntervalCard(
                interval = interval,
                modifier = Modifier.fillMaxWidth()
            )
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Temporal trajectory visualization
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .clip(RoundedCornerShape(16.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            TemporalTrajectoryCanvas(
                currentInterval = currentInterval,
                trajectories = futureTrajectories,
                onTrajectorySelected = { path ->
                    path.end.let { interval ->
                        viewModel.exploreTemporalPossibility(interval)
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        }
        
        // Navigation controls
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(
                onClick = { viewModel.induceMetaCognition() },
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.secondary
                )
            ) {
                Text("Induce Meta-Cognition")
            }
        }
    }
}

@Composable
fun CurrentIntervalCard(
    interval: TemporalInterval,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Current Interval",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = interval.state.name,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = "Duration: ${(interval.duration * 1000).toInt()}ms",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = "Potential: ${(interval.virtualPotential * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
fun TemporalTrajectoryCanvas(
    currentInterval: TemporalInterval?,
    trajectories: List<TemporalPath>,
    onTrajectorySelected: (TemporalPath) -> Unit,
    modifier: Modifier = Modifier
) {
    val animatedProgress = remember { Animatable(0f) }
    
    LaunchedEffect(trajectories) {
        animatedProgress.animateTo(
            targetValue = 1f,
            animationSpec = tween(1000)
        )
    }
    
    Canvas(
        modifier = modifier
            .pointerInput(trajectories) {
                detectTapGestures { offset ->
                    // Find closest trajectory
                    // Simplified - would need proper hit detection
                    trajectories.firstOrNull()?.let(onTrajectorySelected)
                }
            }
    ) {
        val centerX = size.width / 2
        val centerY = size.height / 2
        val progress = animatedProgress.value
        
        // Draw current position
        currentInterval?.let {
            drawCircle(
                color = getStateColor(it.state),
                radius = 20.dp.toPx(),
                center = Offset(centerX, centerY)
            )
        }
        
        // Draw future trajectories
        trajectories.forEachIndexed { index, path ->
            val angle = (index * 2 * PI / trajectories.size).toFloat()
            val distance = 150.dp.toPx() * progress
            
            val endX = centerX + cos(angle) * distance
            val endY = centerY + sin(angle) * distance
            
            // Draw trajectory line
            drawLine(
                color = getStateColor(path.end.state).copy(
                    alpha = path.probability.toFloat()
                ),
                start = Offset(centerX, centerY),
                end = Offset(endX, endY),
                strokeWidth = (3.dp.toPx() * path.probability).toFloat()
            )
            
            // Draw end state
            drawCircle(
                color = getStateColor(path.end.state),
                radius = (15.dp.toPx() * path.probability).toFloat(),
                center = Offset(endX, endY)
            )
            
            // Probability label
            translate(endX - 20, endY + 25) {
                drawIntoCanvas { canvas ->
                    canvas.nativeCanvas.drawText(
                        "${(path.probability * 100).toInt()}%",
                        0f, 0f,
                        android.graphics.Paint().apply {
                            color = android.graphics.Color.WHITE
                            textSize = 24f
                            textAlign = android.graphics.Paint.Align.CENTER
                        }
                    )
                }
            }
        }
    }
}

/**
 * Meta-cognition visualization
 */
@Composable
fun MetaCognitionView(
    viewModel: Phase2ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    val temporalAwareness by viewModel.temporalAwareness.collectAsStateWithLifecycle()
    val htmResults by viewModel.htmResults.collectAsStateWithLifecycle(
        initialValue = HTMResult(0, 0, 0.0, 1.0)
    )
    
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Strange loop visualization
        StrangeLoopVisualization(
            awareness = temporalAwareness,
            modifier = Modifier
                .size(300.dp)
                .padding(16.dp)
        )
        
        // HTM pattern analysis
        HTMPatternCard(
            htmResult = htmResults,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp)
        )
        
        // Emergent patterns display
        EmergentPatternsCard(
            viewModel = viewModel,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Composable
fun StrangeLoopVisualization(
    awareness: Double,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition()
    
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(10000, easing = LinearEasing)
        )
    )
    
    Canvas(modifier = modifier) {
        val center = Offset(size.width / 2, size.height / 2)
        val radius = size.minDimension / 2 * 0.8f
        
        // Draw Möbius strip-like strange loop
        for (t in 0..360 step 5) {
            val angle = (t + rotation) * PI / 180
            val twist = t * PI / 180
            
            val r1 = radius * (0.8f + 0.2f * cos(twist * 3).toFloat())
            val r2 = radius * (0.8f + 0.2f * cos(twist * 3 + PI).toFloat())
            
            val x1 = center.x + cos(angle).toFloat() * r1
            val y1 = center.y + sin(angle).toFloat() * r1
            val x2 = center.x + cos(angle + PI).toFloat() * r2
            val y2 = center.y + sin(angle + PI).toFloat() * r2
            
            drawLine(
                color = Color.Cyan.copy(alpha = awareness.toFloat()),
                start = Offset(x1, y1),
                end = Offset(x2, y2),
                strokeWidth = 2.dp.toPx()
            )
        }
        
        // Center awareness indicator
        drawCircle(
            color = Color.White.copy(alpha = awareness.toFloat()),
            radius = 20.dp.toPx(),
            center = center
        )
    }
}

@Composable
fun HTMPatternCard(
    htmResult: HTMResult,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "HTM Network Analysis",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                HTMMetric(
                    label = "Active",
                    value = htmResult.activeCount.toString(),
                    color = Color.Green
                )
                HTMMetric(
                    label = "Predicted",
                    value = htmResult.predictedCount.toString(),
                    color = Color.Blue
                )
                HTMMetric(
                    label = "Coherence",
                    value = "${(htmResult.temporalCoherence * 100).toInt()}%",
                    color = Color.Magenta
                )
            }
            
            // Anomaly indicator
            LinearProgressIndicator(
                progress = htmResult.anomalyScore.toFloat(),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp),
                color = when {
                    htmResult.anomalyScore > 0.7 -> Color.Red
                    htmResult.anomalyScore > 0.4 -> Color.Yellow
                    else -> Color.Green
                }
            )
            
            Text(
                text = "Anomaly Level",
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.padding(top = 4.dp)
            )
        }
    }
}

@Composable
fun HTMMetric(
    label: String,
    value: String,
    color: Color
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.titleLarge,
            color = color,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall
        )
    }
}

@Composable
fun EmergentPatternsCard(
    viewModel: Phase2ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    // This would connect to the secondOrderObserver's emergent patterns
    val patterns = remember { 
        listOf(
            "high_temporal_awareness",
            "strange_loop_active",
            "stable_observation_pattern"
        )
    }
    
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Emergent Patterns",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            patterns.forEach { pattern ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(
                                color = Color.Cyan,
                                shape = CircleShape
                            )
                    )
                    
                    Text(
                        text = pattern.replace('_', ' ').capitalize(),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }
            }
        }
    }
}

/**
 * Enhanced control panel for Phase 2
 */
@Composable
fun Phase2ControlPanel(
    viewModel: Phase2ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Temporal Navigation Controls",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            // Input generation buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                InputButton(
                    text = "Perception",
                    onClick = {
                        viewModel.exploreTemporalPossibility(
                            TemporalInterval(
                                id = "perception_${System.currentTimeMillis()}",
                                startTime = System.currentTimeMillis() / 1000.0,
                                endTime = (System.currentTimeMillis() + 100) / 1000.0,
                                state = ConsciousnessState.AWARE,
                                virtualPotential = 0.6
                            )
                        )
                    },
                    modifier = Modifier.weight(1f)
                )
                
                InputButton(
                    text = "Deep Thought",
                    onClick = {
                        viewModel.exploreTemporalPossibility(
                            TemporalInterval(
                                id = "thought_${System.currentTimeMillis()}",
                                startTime = System.currentTimeMillis() / 1000.0,
                                endTime = (System.currentTimeMillis() + 200) / 1000.0,
                                state = ConsciousnessState.META_CONSCIOUS,
                                virtualPotential = 0.9
                            )
                        )
                    },
                    modifier = Modifier.weight(1f)
                )
                
                InputButton(
                    text = "Temporal Fold",
                    onClick = {
                        viewModel.exploreTemporalPossibility(
                            TemporalInterval(
                                id = "fold_${System.currentTimeMillis()}",
                                startTime = System.currentTimeMillis() / 1000.0,
                                endTime = (System.currentTimeMillis() + 50) / 1000.0,
                                state = ConsciousnessState.FOLD_POINT,
                                virtualPotential = 0.98
                            )
                        )
                    },
                    modifier = Modifier.weight(1f)
                )
            }
            
            // Meta-cognition trigger
            Button(
                onClick = { viewModel.induceMetaCognition() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.secondary
                )
            ) {
                Text("Trigger Deep Recursive Observation")
            }
        }
    }
}

