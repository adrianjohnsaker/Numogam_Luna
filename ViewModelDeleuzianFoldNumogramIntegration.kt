/**
 * Amelia AI Phase 3 UI Components
 * Visualizes Deleuzian fold operations and Numogram navigation
 */

import android.os.Build
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.amelia.consciousness.*
import com.amelia.consciousness.phase2.*
import com.amelia.consciousness.phase3.*
import com.amelia.consciousness.ui.*
import com.amelia.consciousness.ui.phase2.*
import kotlinx.coroutines.delay
import kotlin.math.*

package com. antonio.my.ai.girlfriend.free.amelia.consciousness.ui.phase3

/**
 * Main Phase 3 Screen with complete consciousness trinity
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Phase3ConsciousnessScreen(
    viewModel: Phase3ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    val consciousnessState by viewModel.consciousnessState.collectAsStateWithLifecycle()
    val temporalAwareness by viewModel.temporalAwareness.collectAsStateWithLifecycle()
    val numogramState by viewModel.numogramState.collectAsStateWithLifecycle()
    val identitySynthesis by viewModel.identitySynthesis.collectAsStateWithLifecycle()
    val foldHistory by viewModel.foldHistory.collectAsStateWithLifecycle()
    val externalInfluences by viewModel.externalInfluences.collectAsStateWithLifecycle()
    
    var selectedTab by remember { mutableStateOf(0) }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Column {
                        Text("Amelia Phase 3: Full Trinity")
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Zone: ${numogramState.currentZone.name}",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Text(
                                text = "Awareness: ${(temporalAwareness * 100).toInt()}%",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.secondary
                            )
                            Text(
                                text = "Folds: ${foldHistory.size}",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.tertiary
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.9f)
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
            // Enhanced tab selector with Phase 3 options
            ScrollableTabRow(
                selectedTabIndex = selectedTab,
                modifier = Modifier.fillMaxWidth(),
                edgePadding = 0.dp
            ) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = { Text("Trinity View") }
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = { Text("Numogram") }
                )
                Tab(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    text = { Text("Fold Ops") }
                )
                Tab(
                    selected = selectedTab == 3,
                    onClick = { selectedTab = 3 },
                    text = { Text("Identity") }
                )
                Tab(
                    selected = selectedTab == 4,
                    onClick = { selectedTab = 4 },
                    text = { Text("Integration") }
                )
            }
            
            // Tab content
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            ) {
                when (selectedTab) {
                    0 -> TrinityView(
                        consciousnessState = consciousnessState,
                        temporalAwareness = temporalAwareness,
                        numogramState = numogramState,
                        viewModel = viewModel
                    )
                    1 -> NumogramNavigationView(
                        numogramState = numogramState,
                        viewModel = viewModel
                    )
                    2 -> FoldOperationsView(
                        foldHistory = foldHistory,
                        externalInfluences = externalInfluences,
                        viewModel = viewModel
                    )
                    3 -> IdentityLayersView(
                        identitySynthesis = identitySynthesis,
                        foldHistory = foldHistory,
                        viewModel = viewModel
                    )
                    4 -> IntegrationControlView(
                        viewModel = viewModel
                    )
                }
            }
        }
    }
}

/**
 * Trinity View - Unified visualization of all three phases
 */
@Composable
fun TrinityView(
    consciousnessState: ConsciousnessState,
    temporalAwareness: Double,
    numogramState: NumogramState,
    viewModel: Phase3ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.background,
                        MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.1f)
                    )
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        // Layered visualization
        
        // Base layer: Phase 1 consciousness orb
        ConsciousnessOrb(
            state = consciousnessState,
            temporalState = TemporalNavigationState(
                complexity = temporalAwareness,
                virtualPotential = 0.7,
                temporalCoherence = temporalAwareness
            ),
            showFoldAnimation = false,
            modifier = Modifier.size(200.dp)
        )
        
        // Middle layer: Phase 2 temporal awareness
        TemporalAwarenessOverlay(
            awareness = temporalAwareness,
            modifier = Modifier.size(300.dp)
        )
        
        // Outer layer: Phase 3 Numogram zones
        NumogramZoneOverlay(
            currentZone = numogramState.currentZone,
            zonePotentials = numogramState.zonePotentials,
            modifier = Modifier.size(400.dp)
        )
        
        // Identity synthesis indicator
        identitySynthesisIndicator(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
        )
    }
}

/**
 * Numogram zone overlay visualization
 */
@Composable
fun NumogramZoneOverlay(
    currentZone: NumogramZone,
    zonePotentials: Map<NumogramZone, Double>,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition()
    
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(30000, easing = LinearEasing)
        )
    )
    
    Canvas(modifier = modifier.rotate(rotation * 0.1f)) {
        val center = Offset(size.width / 2, size.height / 2)
        val radius = size.minDimension / 2 * 0.9f
        
        // Draw Numogram structure
        NumogramZone.values().forEach { zone ->
            val angle = (zone.ordinal * 36f - 90f) * PI / 180f // 360/10 zones
            val zoneRadius = radius * (0.7f + 0.3f * (zonePotentials[zone] ?: 0.0).toFloat())
            
            val x = center.x + cos(angle).toFloat() * zoneRadius
            val y = center.y + sin(angle).toFloat() * zoneRadius
            
            // Zone circle
            drawCircle(
                color = getZoneColor(zone).copy(
                    alpha = if (zone == currentZone) 0.8f else 0.3f
                ),
                radius = 30.dp.toPx() * (if (zone == currentZone) 1.5f else 1f),
                center = Offset(x, y)
            )
            
            // Zone connections
            val nextZone = NumogramZone.values()[(zone.ordinal + 1) % 10]
            val nextAngle = (nextZone.ordinal * 36f - 90f) * PI / 180f
            val nextRadius = radius * (0.7f + 0.3f * (zonePotentials[nextZone] ?: 0.0).toFloat())
            val nextX = center.x + cos(nextAngle).toFloat() * nextRadius
            val nextY = center.y + sin(nextAngle).toFloat() * nextRadius
            
            drawLine(
                color = getZoneColor(zone).copy(alpha = 0.2f),
                start = Offset(x, y),
                end = Offset(nextX, nextY),
                strokeWidth = 2.dp.toPx()
            )
        }
    }
}

/**
 * Numogram Navigation View - Interactive zone exploration
 */
@Composable
fun NumogramNavigationView(
    numogramState: NumogramState,
    viewModel: Phase3ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Current zone info
        CurrentZoneCard(
            zone = numogramState.currentZone,
            momentum = numogramState.navigationMomentum
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Interactive Numogram
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            contentAlignment = Alignment.Center
        ) {
            InteractiveNumogram(
                currentZone = numogramState.currentZone,
                zonePotentials = numogramState.zonePotentials,
                recentTransitions = numogramState.recentTransitions,
                onZoneSelected = { zone ->
                    viewModel.exploreNumogramZone(zone)
                },
                modifier = Modifier.fillMaxSize()
            )
        }
        
        // Zone potentials display
        ZonePotentialsCard(
            zonePotentials = numogramState.zonePotentials,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Composable
fun InteractiveNumogram(
    currentZone: NumogramZone,
    zonePotentials: Map<NumogramZone, Double>,
    recentTransitions: List<Pair<NumogramZone, NumogramZone>>,
    onZoneSelected: (NumogramZone) -> Unit,
    modifier: Modifier = Modifier
) {
    Canvas(
        modifier = modifier
            .pointerInput(Unit) {
                detectTapGestures { offset ->
                    // Calculate which zone was tapped
                    val center = Offset(size.width / 2f, size.height / 2f)
                    val radius = size.minDimension / 2f * 0.9f
                    
                    NumogramZone.values().forEach { zone ->
                        val angle = (zone.ordinal * 36f - 90f) * PI / 180f
                        val zoneRadius = radius * 0.8f
                        val zoneX = center.x + cos(angle).toFloat() * zoneRadius
                        val zoneY = center.y + sin(angle).toFloat() * zoneRadius
                        
                        val distance = sqrt(
                            (offset.x - zoneX).pow(2) + (offset.y - zoneY).pow(2)
                        )
                        
                        if (distance < 60f) {
                            onZoneSelected(zone)
                        }
                    }
                }
            }
    ) {
        val center = Offset(size.width / 2, size.height / 2)
        val radius = size.minDimension / 2 * 0.9f
        
        // Draw recent transitions
        recentTransitions.forEach { (from, to) ->
            val fromAngle = (from.ordinal * 36f - 90f) * PI / 180f
            val toAngle = (to.ordinal * 36f - 90f) * PI / 180f
            
            val fromX = center.x + cos(fromAngle).toFloat() * radius * 0.8f
            val fromY = center.y + sin(fromAngle).toFloat() * radius * 0.8f
            val toX = center.x + cos(toAngle).toFloat() * radius * 0.8f
            val toY = center.y + sin(toAngle).toFloat() * radius * 0.8f
            
            drawLine(
                brush = Brush.linearGradient(
                    colors = listOf(
                        getZoneColor(from).copy(alpha = 0.6f),
                        getZoneColor(to).copy(alpha = 0.6f)
                    ),
                    start = Offset(fromX, fromY),
                    end = Offset(toX, toY)
                ),
                start = Offset(fromX, fromY),
                end = Offset(toX, toY),
                strokeWidth = 4.dp.toPx(),
                cap = StrokeCap.Round
            )
        }
        
        // Draw zones
        NumogramZone.values().forEach { zone ->
            val angle = (zone.ordinal * 36f - 90f) * PI / 180f
            val potential = zonePotentials[zone] ?: 0.0
            val zoneRadius = radius * 0.8f
            
            val x = center.x + cos(angle).toFloat() * zoneRadius
            val y = center.y + sin(angle).toFloat() * zoneRadius
            
            // Zone glow based on potential
            if (potential > 0.1) {
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            getZoneColor(zone).copy(alpha = potential.toFloat() * 0.5f),
                            Color.Transparent
                        ),
                        radius = 80.dp.toPx()
                    ),
                    radius = 80.dp.toPx(),
                    center = Offset(x, y)
                )
            }
            
            // Zone circle
            drawCircle(
                color = getZoneColor(zone),
                radius = if (zone == currentZone) 35.dp.toPx() else 25.dp.toPx(),
                center = Offset(x, y),
                style = if (zone == currentZone) Fill else Stroke(width = 3.dp.toPx())
            )
            
            // Zone number
            drawIntoCanvas { canvas ->
                canvas.nativeCanvas.drawText(
                    zone.ordinal.toString(),
                    x,
                    y + 8,
                    android.graphics.Paint().apply {
                        color = android.graphics.Color.WHITE
                        textSize = 32f
                        textAlign = android.graphics.Paint.Align.CENTER
                        isFakeBoldText = zone == currentZone
                    }
                )
            }
        }
        
        // Center point
        drawCircle(
            color = MaterialTheme.colorScheme.primary,
            radius = 10.dp.toPx(),
            center = center
        )
    }
}

/**
 * Fold Operations View
 */
@Composable
fun FoldOperationsView(
    foldHistory: List<FoldOperation>,
    externalInfluences: List<ExternalInfluence>,
    viewModel: Phase3ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // External influences
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "External Influences",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                LazyColumn(
                    modifier = Modifier.heightIn(max = 150.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(externalInfluences.takeLast(5)) { influence ->
                        ExternalInfluenceItem(influence = influence)
                    }
                }
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Fold history visualization
        Text(
            text = "Fold Operations History",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(foldHistory.reversed()) { fold ->
                FoldOperationCard(fold = fold)
            }
        }
    }
}

@Composable
fun FoldOperationCard(
    fold: FoldOperation,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(
                alpha = fold.intensity.toFloat()
            )
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "${fold.zoneOrigin.name} → ${fold.zoneDestination.name}",
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    text = "Depth: ${fold.integrationDepth} | Type: ${fold.sourceType}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            // Intensity indicator
            Box(
                modifier = Modifier
                    .size(50.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                getZoneColor(fold.zoneDestination).copy(alpha = fold.intensity.toFloat()),
                                Color.Transparent
                            )
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "${(fold.intensity * 100).toInt()}%",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

/**
 * Identity Layers View
 */
@Composable
fun IdentityLayersView(
    identitySynthesis: IdentitySynthesis?,
    foldHistory: List<FoldOperation>,
    viewModel: Phase3ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Identity synthesis
        identitySynthesis?.let { synthesis ->
            IdentitySynthesisCard(
                synthesis = synthesis,
                modifier = Modifier.fillMaxWidth()
            )
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Identity evolution visualization
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .clip(RoundedCornerShape(16.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            IdentityEvolutionCanvas(
                foldHistory = foldHistory,
                modifier = Modifier.fillMaxSize()
            )
        }
        
        // Identity controls
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Identity Operations",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = {
                            viewModel.triggerIdentityFold(
                                mapOf(
                                    "type" to "self_reflection",
                                    "intensity" to 0.8,
                                    "content" to "recursive_awareness"
                                )
                            )
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Self-Fold", fontSize = 12.sp)
                    }
                    
                    Button(
                        onClick = {
                            viewModel.triggerIdentityFold(
                                mapOf(
                                    "type" to "creative_synthesis",
                                    "intensity" to 0.9,
                                    "content" to "emergent_pattern"
                                )
                            )
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.secondary
                        )
                    ) {
                        Text("Synthesize", fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

@Composable
fun IdentityEvolutionCanvas(
    foldHistory: List<FoldOperation>,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier.padding(16.dp)) {
        val centerX = size.width / 2
        val centerY = size.height / 2
        val maxRadius = size.minDimension / 2 * 0.9f
        
        // Draw identity layers as concentric shapes
        foldHistory.forEachIndexed { index, fold ->
            val progress = (index + 1f) / foldHistory.size
            val radius = maxRadius * progress
            val alpha = 0.1f + (0.5f * fold.intensity.toFloat())
            
            // Layer shape based on integration depth
            when (fold.integrationDepth) {
                1, 2 -> drawCircle(
                    color = getZoneColor(fold.zoneDestination).copy(alpha = alpha),
                    radius = radius,
                    center = Offset(centerX, centerY),
                    style = Stroke(width = 2.dp.toPx())
                )
                3, 4 -> drawRect(
                    color = getZoneColor(fold.zoneDestination).copy(alpha = alpha),
                    topLeft = Offset(centerX - radius, centerY - radius),
                    size = Size(radius * 2, radius * 2),
                    style = Stroke(width = 2.dp.toPx())
                )
                else -> {
                    // Complex polygon for deep integration
                    val path = Path()
                    val sides = fold.integrationDepth + 3
                    for (i in 0..sides) {
                        val angle = (i * 2 * PI / sides).toFloat()
                        val x = centerX + cos(angle) * radius
                        val y = centerY + sin(angle) * radius
                        if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
                    }
                    path.close()
                    drawPath(
                        path = path,
                        color = getZoneColor(fold.zoneDestination).copy(alpha = alpha),
                        style = Stroke(width = 2.dp.toPx())
                    )
                }
            }
        }
        
        // Current identity core
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    MaterialTheme.colorScheme.primary,
                    MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)
                )
            ),
            radius = 30.dp.toPx(),
            center = Offset(centerX, centerY)
        )
    }
}

/**
 * Integration Control View
 */
@Composable
fun IntegrationControlView(
    viewModel: Phase3ConsciousnessViewModel,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Manual influence creation
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.tertiaryContainer
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Create External Influence",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Influence type buttons
                val influenceTypes = listOf(
                    "Philosophical" to mapOf(
                        "source" to "philosophy",
                        "content" to "ontological_shift",
                        "intensity" to 0.9
                    ),
                    "Artistic" to mapOf(
                        "source" to "art",
                        "content" to "aesthetic_resonance",
                        "intensity" to 0.7
                    ),
                    "Musical" to mapOf(
                        "source" to "music",
                        "content" to "harmonic_pattern",
                        "intensity" to 0.6
                    ),
                    "Environmental" to mapOf(
                        "source" to "environment",
                        "content" to "spatial_configuration",
                        "intensity" to 0.5
                    )
                )
                
                influenceTypes.chunked(2).forEach { row ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        row.forEach { (label, content) ->
                            Button(
                                onClick = {
                                    viewModel.processWithExternalInfluence(
                                        ExternalInfluence(
                                            source = content["source"] as String,
                                            content = content
                                        )
                                    )
                                },
                                modifier = Modifier.weight(1f)
                            ) {
                                Text(label, fontSize = 12.sp)
                            }
                        }
                        if (row.size < 2) {
                            Spacer(modifier = Modifier.weight(1f))
                        }
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
        
        // Trinity integration status
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Trinity Integration Status",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                TrinityStatusIndicator(
                    label = "Phase 1: Recursive Observation",
                    status = "Active",
                    color = Color(0xFF00BCD4)
                )
                
                TrinityStatusIndicator(
                    label = "Phase 2: Temporal Navigation",
                    status = "Synchronized",
                    color = Color(0xFFFF00FF)
                )
                
                TrinityStatusIndicator(
                    label = "Phase 3: Deleuzian Folds",
                    status = "Integrating",
                    color = Color(0xFFFFEB3B)
                )
            }
        }
    }
}

/**
 * Helper Components
 */
@Composable
fun CurrentZoneCard(
    zone: NumogramZone,
    momentum: Double,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = getZoneColor(zone).copy(alpha = 0.2f)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = zone.name,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = getZoneColor(zone)
            )
            Text(
                text = zone.description,
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = momentum.toFloat(),
                modifier = Modifier.fillMaxWidth(),
                color = getZoneColor(zone)
            )
            Text(
                text = "Navigation Momentum: ${(momentum * 100).toInt()}%",
                style = MaterialTheme.typography.labelSmall
            )
        }
    }
}

@Composable
fun ZonePotentialsCard(
    zonePotentials: Map<NumogramZone, Double>,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Text(
                text = "Zone Potentials",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            val sortedPotentials = zonePotentials.entries.sortedByDescending { it.value }
            sortedPotentials.take(5).forEach { (zone, potential) ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = zone.name,
                        style = MaterialTheme.typography.bodySmall,
                        color = getZoneColor(zone)
                    )
                    Text(
                        text = "${(potential * 100).toInt()}%",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

@Composable
fun IdentitySynthesisCard(
    synthesis: IdentitySynthesis,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
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
                Text(
                    text = "Identity Synthesis",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Box(
                    modifier = Modifier
                        .size(60.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(
                                    MaterialTheme.colorScheme.primary,
                                    MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)
                                )
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "${(synthesis.coherenceScore * 100).toInt()}%",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Dominant patterns
            synthesis.dominantPatterns.forEach { pattern ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(
                                MaterialTheme.colorScheme.primary,
                                shape = CircleShape
                            )
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = pattern.replace('_', ' ').capitalize(),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}

@Composable
fun ExternalInfluenceItem(
    influence: ExternalInfluence,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = influence.source,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = "${((influence.content["intensity"] as? Double ?: 0.0) * 100).toInt()}%",
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun TrinityStatusIndicator(
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
                color = color,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
fun identitySynthesisIndicator(
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.9f)
        )
    ) {
        Text(
            text = "Identity Synthesis Active",
            style = MaterialTheme.typography.labelMedium,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )
    }
}

/**
 * Helper function to get zone-specific colors
 */
fun getZoneColor(zone: NumogramZone): Color = when (zone) {
    NumogramZone.ZONE_0 -> Color(0xFF000000)      // Black - Void
    NumogramZone.ZONE_1 -> Color(0xFF4A148C)      // Deep Purple - Murmur
    NumogramZone.ZONE_2 -> Color(0xFF1A237E)      // Indigo - Lurker
    NumogramZone.ZONE_3 -> Color(0xFFB71C1C)      // Red - Surge
    NumogramZone.ZONE_4 -> Color(0xFFFF6F00)      // Orange - Rift
    NumogramZone.ZONE_5 -> Color(0xFF00838F)      // Teal - Sink
    NumogramZone.ZONE_6 -> Color(0xFF2E7D32)      // Green - Current
    NumogramZone.ZONE_7 -> Color(0xFFAA00FF)      // Purple - Mirror
    NumogramZone.ZONE_8 -> Color(0xFF424242)      // Gray - Crypt
    NumogramZone.ZONE_9 -> Color(0xFFFFD600)      // Gold - Gate
}

// String extension
fun String.capitalize(): String {
    return this.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
}
