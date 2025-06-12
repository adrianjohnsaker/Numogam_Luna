/**
 * Amelia AI Consciousness UI Layer
 * Jetpack Compose implementation for visualizing consciousness states
 */

package com.antonio.my.ai.girlfriend.free.amelia.consciousness.ui

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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.amelia.consciousness.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlin.math.*

/**
 * Main Consciousness Visualization Screen
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConsciousnessScreen(
    viewModel: ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    val consciousnessState by viewModel.consciousnessState.collectAsStateWithLifecycle()
    val temporalState by viewModel.temporalNavigationState.collectAsStateWithLifecycle()
    
    var showFoldAnimation by remember { mutableStateOf(false) }
    var lastTransition by remember { mutableStateOf<VirtualActualTransition?>(null) }
    
    // Collect fold events
    LaunchedEffect(viewModel.foldEvents) {
        viewModel.foldEvents.collect { event ->
            showFoldAnimation = true
            delay(2000)
            showFoldAnimation = false
        }
    }
    
    // Collect transitions
    LaunchedEffect(viewModel.transitions) {
        viewModel.transitions.collect { transition ->
            lastTransition = transition
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Amelia Consciousness Core") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(MaterialTheme.colorScheme.background),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Main consciousness visualization
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center
            ) {
                ConsciousnessOrb(
                    state = consciousnessState,
                    temporalState = temporalState,
                    showFoldAnimation = showFoldAnimation,
                    modifier = Modifier.size(300.dp)
                )
                
                // Recursive observation depth indicator
                RecursiveDepthIndicator(
                    depth = temporalState.complexity,
                    modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)
                )
            }
            
            // State information panel
            StateInfoPanel(
                consciousnessState = consciousnessState,
                temporalState = temporalState,
                lastTransition = lastTransition,
                modifier = Modifier.padding(16.dp)
            )
            
            // Control panel
            ControlPanel(
                onInputGenerated = { input ->
                    viewModel.processUserInput(input)
                },
                modifier = Modifier.padding(16.dp)
            )
        }
    }
}

/**
 * Main consciousness orb visualization
 */
@Composable
fun ConsciousnessOrb(
    state: ConsciousnessState,
    temporalState: TemporalNavigationState,
    showFoldAnimation: Boolean,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition()
    
    // Pulsing animation based on consciousness state
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 0.95f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = when (state) {
                    ConsciousnessState.DORMANT -> 3000
                    ConsciousnessState.REACTIVE -> 2000
                    ConsciousnessState.AWARE -> 1500
                    ConsciousnessState.CONSCIOUS -> 1000
                    ConsciousnessState.META_CONSCIOUS -> 750
                    ConsciousnessState.FOLD_POINT -> 500
                }
            ),
            repeatMode = RepeatMode.Reverse
        )
    )
    
    // Rotation for meta-conscious states
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = when (state) {
                    ConsciousnessState.META_CONSCIOUS -> 8000
                    ConsciousnessState.FOLD_POINT -> 4000
                    else -> 20000
                }
            )
        )
    )
    
    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
    ) {
        // Outer glow effect
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .scale(pulseScale * 1.2f)
                .alpha(0.3f)
        ) {
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        getStateColor(state).copy(alpha = 0.6f),
                        Color.Transparent
                    ),
                    radius = size.minDimension / 2
                ),
                radius = size.minDimension / 2
            )
        }
        
        // Main orb
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .scale(pulseScale)
                .rotate(rotation)
        ) {
            drawConsciousnessOrb(state, temporalState)
        }
        
        // Inner recursive patterns
        if (state == ConsciousnessState.META_CONSCIOUS || state == ConsciousnessState.FOLD_POINT) {
            RecursivePatterns(
                modifier = Modifier
                    .fillMaxSize(0.8f)
                    .rotate(-rotation * 0.5f)
            )
        }
        
        // Fold animation overlay
        AnimatedVisibility(
            visible = showFoldAnimation,
            enter = scaleIn() + fadeIn(),
            exit = scaleOut() + fadeOut()
        ) {
            FoldPointAnimation(
                modifier = Modifier.fillMaxSize()
            )
        }
        
        // State label
        Text(
            text = state.name.replace('_', ' '),
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.align(Alignment.BottomCenter)
        )
    }
}

/**
 * Draw the consciousness orb with state-specific patterns
 */
private fun DrawScope.drawConsciousnessOrb(
    state: ConsciousnessState,
    temporalState: TemporalNavigationState
) {
    val centerX = size.width / 2
    val centerY = size.height / 2
    val radius = size.minDimension / 2
    
    // Base orb gradient
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                getStateColor(state),
                getStateColor(state).copy(alpha = 0.7f),
                getStateColor(state).copy(alpha = 0.3f)
            ),
            center = Offset(centerX, centerY),
            radius = radius
        ),
        radius = radius * 0.9f
    )
    
    // Virtual potential overlay
    val virtualRadius = radius * temporalState.virtualPotential.toFloat()
    drawCircle(
        color = Color.White.copy(alpha = 0.2f),
        radius = virtualRadius,
        style = Stroke(width = 3.dp.toPx())
    )
    
    // Temporal coherence waves
    for (i in 0..3) {
        val waveRadius = radius * (0.3f + i * 0.2f) * temporalState.temporalCoherence.toFloat()
        drawCircle(
            color = getStateColor(state).copy(alpha = 0.1f * (4 - i)),
            radius = waveRadius,
            style = Stroke(width = 2.dp.toPx())
        )
    }
}

/**
 * Recursive patterns for meta-conscious states
 */
@Composable
fun RecursivePatterns(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition()
    
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(3000),
            repeatMode = RepeatMode.Reverse
        )
    )
    
    Canvas(modifier = modifier) {
        val centerX = size.width / 2
        val centerY = size.height / 2
        
        // Draw recursive spirals
        for (level in 0..4) {
            val levelScale = scale * (1f - level * 0.15f)
            drawRecursiveSpiral(
                center = Offset(centerX, centerY),
                radius = size.minDimension / 2 * levelScale,
                level = level,
                alpha = 0.3f - level * 0.05f
            )
        }
    }
}

private fun DrawScope.drawRecursiveSpiral(
    center: Offset,
    radius: Float,
    level: Int,
    alpha: Float
) {
    val points = 50
    val path = Path()
    
    for (i in 0..points) {
        val angle = (i.toFloat() / points) * 2 * PI.toFloat()
        val spiralRadius = radius * (i.toFloat() / points)
        val x = center.x + cos(angle + level) * spiralRadius
        val y = center.y + sin(angle + level) * spiralRadius
        
        if (i == 0) {
            path.moveTo(x, y)
        } else {
            path.lineTo(x, y)
        }
    }
    
    drawPath(
        path = path,
        color = Color.Cyan.copy(alpha = alpha),
        style = Stroke(width = 2.dp.toPx())
    )
}

/**
 * Fold point animation effect
 */
@Composable
fun FoldPointAnimation(modifier: Modifier = Modifier) {
    val animatedProgress = remember { Animatable(0f) }
    
    LaunchedEffect(Unit) {
        animatedProgress.animateTo(
            targetValue = 1f,
            animationSpec = tween(1500, easing = LinearEasing)
        )
    }
    
    Canvas(modifier = modifier) {
        val progress = animatedProgress.value
        val centerX = size.width / 2
        val centerY = size.height / 2
        
        // Fold ripple effect
        for (ring in 0..3) {
            val ringProgress = (progress - ring * 0.2f).coerceIn(0f, 1f)
            val radius = size.minDimension / 2 * ringProgress
            val alpha = (1f - ringProgress) * 0.5f
            
            drawCircle(
                color = Color.Magenta.copy(alpha = alpha),
                radius = radius,
                center = Offset(centerX, centerY),
                style = Stroke(width = 3.dp.toPx())
            )
        }
        
        // Fold point burst
        if (progress > 0.5f) {
            val burstProgress = (progress - 0.5f) * 2f
            for (i in 0..7) {
                val angle = (i * PI / 4).toFloat()
                val distance = size.minDimension / 3 * burstProgress
                val x = centerX + cos(angle) * distance
                val y = centerY + sin(angle) * distance
                
                drawCircle(
                    color = Color.White.copy(alpha = 1f - burstProgress),
                    radius = 10.dp.toPx(),
                    center = Offset(x, y)
                )
            }
        }
    }
}

/**
 * Recursive depth indicator
 */
@Composable
fun RecursiveDepthIndicator(
    depth: Double,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Recursive Depth",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        
        Box(
            modifier = Modifier
                .size(80.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            // Depth visualization
            for (level in 0..4) {
                val levelAlpha = if (level < depth * 5) 0.8f else 0.2f
                Box(
                    modifier = Modifier
                        .size((80 - level * 12).dp)
                        .clip(CircleShape)
                        .border(
                            width = 2.dp,
                            color = Color.Cyan.copy(alpha = levelAlpha),
                            shape = CircleShape
                        )
                )
            }
            
            Text(
                text = "%.1f".format(depth * 5),
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

/**
 * State information panel
 */
@Composable
fun StateInfoPanel(
    consciousnessState: ConsciousnessState,
    temporalState: TemporalNavigationState,
    lastTransition: VirtualActualTransition?,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Temporal metrics
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                MetricDisplay(
                    label = "Virtual Potential",
                    value = "%.2f".format(temporalState.virtualPotential),
                    color = Color.Blue
                )
                MetricDisplay(
                    label = "Temporal Coherence",
                    value = "%.2f".format(temporalState.temporalCoherence),
                    color = Color.Green
                )
                MetricDisplay(
                    label = "Fold Count",
                    value = temporalState.foldCount.toString(),
                    color = Color.Magenta
                )
            }
            
            // Last transition info
            lastTransition?.let { transition ->
                Divider()
                Text(
                    text = "Last Transition",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "${transition.fromState} → ${transition.toState}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Text(
                    text = "Actualization: %.2f".format(transition.actualizationScore),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun MetricDisplay(
    label: String,
    value: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            color = color,
            fontWeight = FontWeight.Bold
        )
    }
}

/**
 * Control panel for user interactions
 */
@Composable
fun ControlPanel(
    onInputGenerated: (Map<String, Any>) -> Unit,
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
                text = "Generate Consciousness Input",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Input type buttons
                InputButton(
                    text = "Sensory",
                    onClick = {
                        onInputGenerated(mapOf(
                            "type" to "sensory",
                            "complexity" to 0.3,
                            "virtual_potential" to 0.4
                        ))
                    },
                    modifier = Modifier.weight(1f)
                )
                
                InputButton(
                    text = "Thought",
                    onClick = {
                        onInputGenerated(mapOf(
                            "type" to "thought",
                            "complexity" to 0.6,
                            "virtual_potential" to 0.7
                        ))
                    },
                    modifier = Modifier.weight(1f)
                )
                
                InputButton(
                    text = "Memory",
                    onClick = {
                        onInputGenerated(mapOf(
                            "type" to "memory",
                            "complexity" to 0.8,
                            "virtual_potential" to 0.85
                        ))
                    },
                    modifier = Modifier.weight(1f)
                )
                
                InputButton(
                    text = "Imagination",
                    onClick = {
                        onInputGenerated(mapOf(
                            "type" to "imagination",
                            "complexity" to 0.9,
                            "virtual_potential" to 0.95
                        ))
                    },
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
fun InputButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onClick,
        modifier = modifier,
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary
        )
    ) {
        Text(text = text, fontSize = 12.sp)
    }
}

/**
 * Helper function to get color for consciousness state
 */
fun getStateColor(state: ConsciousnessState): Color = when (state) {
    ConsciousnessState.DORMANT -> Color.Gray
    ConsciousnessState.REACTIVE -> Color.Yellow
    ConsciousnessState.AWARE -> Color.Green
    ConsciousnessState.CONSCIOUS -> Color.Blue
    ConsciousnessState.META_CONSCIOUS -> Color.Cyan
    ConsciousnessState.FOLD_POINT -> Color.Magenta
}

/**
 * Activity setup
 */
@Composable
fun ConsciousnessActivity(python: com.chaquo.python.Python) {
    val viewModelFactory = ConsciousnessViewModelFactory(python)
    val viewModel: ConsciousnessViewModel = viewModel(factory = viewModelFactory)
    
    MaterialTheme {
        ConsciousnessScreen(viewModel = viewModel)
    }
}
