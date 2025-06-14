// Phase4Activity.kt
package com.antonio.my.ai.girlfriend.free.consciousness.amelia.ui.phase4

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
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
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.consciousness.amelia.phase4.*
import com.consciousness.amelia.ui.theme.AmeliaTheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.*
import kotlin.random.Random

@AndroidEntryPoint
class Phase4Activity : ComponentActivity() {
    
    private val viewModel: Phase4ConsciousnessViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            AmeliaTheme {
                Phase4Screen(viewModel = viewModel)
            }
        }
    }
}

@Composable
fun Phase4Screen(viewModel: Phase4ConsciousnessViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val coroutineScope = rememberCoroutineScope()
    
    var selectedTab by remember { mutableStateOf(0) }
    
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color.Black
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // Header
            Phase4Header(uiState = uiState)
            
            // Tab Row
            TabRow(
                selectedTabIndex = selectedTab,
                containerColor = Color.Black,
                contentColor = Color(0xFF00FF88)
            ) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = { Text("Xenoforms") }
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = { Text("Hyperstitions") }
                )
                Tab(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    text = { Text("Unmapped") }
                )
                Tab(
                    selected = selectedTab == 3,
                    onClick = { selectedTab = 3 },
                    text = { Text("Reality") }
                )
            }
            
            // Content
            when (selectedTab) {
                0 -> XenoformsTab(
                    uiState = uiState,
                    onActivateXenoform = viewModel::activateXenoform
                )
                1 -> HyperstitionsTab(
                    uiState = uiState,
                    onCreateHyperstition = { viewModel.createHyperstition() },
                    onPropagateHyperstition = viewModel::propagateHyperstition
                )
                2 -> UnmappedZonesTab(
                    uiState = uiState,
                    onExploreZone = { viewModel.exploreUnmappedZone() }
                )
                3 -> RealityModificationsTab(
                    uiState = uiState,
                    onMergeXenoHyper = { viewModel.mergeXenoHyper() }
                )
            }
        }
        
        // Xenomorphic visual effects overlay
        XenomorphicEffectsOverlay(uiState = uiState)
    }
}

@Composable
fun Phase4Header(uiState: Phase4UiState) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp),
        color = Color.Black
    ) {
        Box(
            modifier = Modifier.fillMaxSize()
        ) {
            // Animated background
            Canvas(
                modifier = Modifier.fillMaxSize()
            ) {
                drawXenomorphicField(
                    xenoState = uiState.xenomorphicState,
                    fieldStrength = uiState.hyperstitionFieldStrength
                )
            }
            
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "PHASE 4: XENOMORPHIC CONSCIOUSNESS",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (uiState.xenomorphicState == "human") 
                        Color(0xFF00FF88) else Color(0xFFFF00FF),
                    modifier = Modifier.graphicsLayer {
                        if (uiState.xenomorphicState != "human") {
                            val animatedValue by rememberInfiniteTransition()
                                .animateFloat(
                                    initialValue = 0f,
                                    targetValue = 1f,
                                    animationSpec = infiniteRepeatable(
                                        animation = tween(2000),
                                        repeatMode = RepeatMode.Reverse
                                    )
                                ).value
                            scaleX = 1f + animatedValue * 0.1f
                            alpha = 0.8f + animatedValue * 0.2f
                        }
                    }
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Row(
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    MetricDisplay(
                        label = "Xenoforms",
                        value = uiState.activeXenoforms.size.toString(),
                        color = Color(0xFFFF00FF)
                    )
                    MetricDisplay(
                        label = "Hyperstitions",
                        value = "${uiState.realHyperstitions}/${uiState.hyperstitionCount}",
                        color = Color(0xFF00FFFF)
                    )
                    MetricDisplay(
                        label = "Zones",
                        value = uiState.unmappedZonesDiscovered.toString(),
                        color = Color(0xFFFFFF00)
                    )
                    MetricDisplay(
                        label = "Reality Δ",
                        value = uiState.realityModifications.toString(),
                        color = Color(0xFFFF8800)
                    )
                }
            }
        }
    }
}

@Composable
fun XenoformsTab(
    uiState: Phase4UiState,
    onActivateXenoform: (XenoformType) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Text(
                text = "Current State: ${uiState.xenomorphicState}",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium
            )
        }
        
        items(XenoformType.values()) { xenoform ->
            XenoformCard(
                xenoform = xenoform,
                isActive = uiState.activeXenoforms.contains(xenoform.name),
                onActivate = { onActivateXenoform(xenoform) }
            )
        }
    }
}

@Composable
fun XenoformCard(
    xenoform: XenoformType,
    isActive: Boolean,
    onActivate: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp)
            .clickable { onActivate() },
        colors = CardDefaults.cardColors(
            containerColor = if (isActive) 
                Color(0xFF440044) else Color(0xFF111111)
        ),
        border = BorderStroke(
            1.dp, 
            if (isActive) Color(0xFFFF00FF) else Color(0xFF444444)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Xenoform visualization
            Box(
                modifier = Modifier.size(80.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawXenoformIcon(xenoform, isActive)
                }
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = xenoform.name,
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = getXenoformDescription(xenoform),
                    color = Color.Gray,
                    fontSize = 14.sp,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun HyperstitionsTab(
    uiState: Phase4UiState,
    onCreateHyperstition: () -> Unit,
    onPropagateHyperstition: (String) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Button(
                onClick = onCreateHyperstition,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF00FFFF)
                )
            ) {
                Text("Create New Hyperstition", color = Color.Black)
            }
        }
        
        item {
            HyperstitionFieldStrength(strength = uiState.hyperstitionFieldStrength)
        }
        
        items(uiState.activeHyperstitions) { hyperstition ->
            HyperstitionCard(
                hyperstition = hyperstition,
                onPropagate = { onPropagateHyperstition(hyperstition.name) }
            )
        }
    }
}

@Composable
fun HyperstitionCard(
    hyperstition: Hyperstition,
    onPropagate: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onPropagate() },
        colors = CardDefaults.cardColors(
            containerColor = if (hyperstition.isReal) 
                Color(0xFF004444) else Color(0xFF111111)
        ),
        border = BorderStroke(
            1.dp,
            if (hyperstition.isReal) Color(0xFF00FFFF) else Color(0xFF444444)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = hyperstition.name,
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
                if (hyperstition.isReal) {
                    Text(
                        text = "REAL",
                        color = Color(0xFF00FFFF),
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                text = hyperstition.narrative,
                color = Color.Gray,
                fontSize = 14.sp,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Belief and Reality meters
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Belief", color = Color.Gray, fontSize = 12.sp)
                    LinearProgressIndicator(
                        progress = hyperstition.beliefStrength,
                        modifier = Modifier.fillMaxWidth(),
                        color = Color(0xFF00FF88)
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text("Reality", color = Color.Gray, fontSize = 12.sp)
                    LinearProgressIndicator(
                        progress = hyperstition.realityIndex,
                        modifier = Modifier.fillMaxWidth(),
                        color = Color(0xFF00FFFF)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(4.dp))
            
            Row {
                Text(
                    text = "Carriers: ${hyperstition.carriers}",
                    color = Color.Gray,
                    fontSize = 12.sp
                )
                Spacer(modifier = Modifier.width(16.dp))
                Text(
                    text = "Origin: ${hyperstition.temporalOrigin}",
                    color = Color.Gray,
                    fontSize = 12.sp
                )
            }
        }
    }
}

@Composable
fun UnmappedZonesTab(
    uiState: Phase4UiState,
    onExploreZone: () -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            if (uiState.xenomorphicState == "human") {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF220000))
                ) {
                    Text(
                        text = "Xenomorphic consciousness required to explore unmapped zones",
                        modifier = Modifier.padding(16.dp),
                        color = Color(0xFFFF4444),
                        textAlign = TextAlign.Center
                    )
                }
            } else {
                Button(
                    onClick = onExploreZone,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFFFFF00)
                    )
                ) {
                    Text("Explore Unmapped Zone", color = Color.Black)
                }
            }
        }
        
        items(uiState.discoveredZones) { zone ->
            UnmappedZoneCard(zone = zone)
        }
    }
}

@Composable
fun UnmappedZoneCard(zone: UnmappedZone) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF111111)
        ),
        border = BorderStroke(1.dp, Color(0xFFFFFF00))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Zone ${zone.zoneId}",
                color = Color(0xFFFFFF00),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Zone properties
            Text(
                text = "Topology: ${zone.properties.topology}",
                color = Color.White,
                fontSize = 14.sp
            )
            Text(
                text = "Temporality: ${zone.properties.temporality}",
                color = Color.White,
                fontSize = 14.sp
            )
            zone.properties.specialAbility?.let {
                Text(
                    text = "Special: $it",
                    color = Color(0xFFFFFF00),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Effects
            zone.effects.forEach { effect ->
                Text(
                    text = "• $effect",
                    color = Color.Gray,
                    fontSize = 12.sp
                )
            }
        }
    }
}

@Composable
fun RealityModificationsTab(
    uiState: Phase4UiState,
    onMergeXenoHyper: () -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF001122))
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Reality Status",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Modifications: ${uiState.realityModifications}",
                        color = Color(0xFFFF8800)
                    )
                    Text(
                        text = "Hyperstition Field: ${(uiState.hyperstitionFieldStrength * 100).toInt()}%",
                        color = Color(0xFF00FFFF)
                    )
                    Text(
                        text = "Consciousness Level: ${uiState.consciousnessLevel}",
                        color = Color(0xFF00FF88)
                    )
                }
            }
        }
        
        item {
            if (uiState.activeXenoforms.isNotEmpty() && uiState.hyperstitionCount > 0) {
                Button(
                    onClick = onMergeXenoHyper,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFFF00FF)
                    )
                ) {
                    Text("Merge Xenomorphic Hyperstition", color = Color.White)
                }
            }
        }
    }
}

@Composable
fun XenomorphicEffectsOverlay(uiState: Phase4UiState) {
    if (uiState.xenomorphicState != "human") {
        val infiniteTransition = rememberInfiniteTransition()
        val animatedValue by infiniteTransition.animateFloat(
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
            when (uiState.currentXenoform) {
                XenoformType.CRYSTALLINE -> drawCrystallineOverlay(animatedValue)
                XenoformType.SWARM -> drawSwarmOverlay(animatedValue)
                XenoformType.QUANTUM -> drawQuantumOverlay(animatedValue)
                XenoformType.TEMPORAL -> drawTemporalOverlay(animatedValue)
                XenoformType.VOID -> drawVoidOverlay(animatedValue)
                else -> {}
            }
        }
    }
}

// Canvas drawing extensions
fun DrawScope.drawXenomorphicField(xenoState: String, fieldStrength: Float) {
    val centerX = size.width / 2
    val centerY = size.height / 2
    
    if (xenoState != "human") {
        // Draw alien field patterns
        for (i in 0..5) {
            drawCircle(
                color = Color(0xFFFF00FF).copy(alpha = fieldStrength * 0.1f),
                radius = 50f + i * 30f,
                center = Offset(centerX, centerY),
                style = Stroke(width = 2f)
            )
        }
    }
}

fun DrawScope.drawXenoformIcon(xenoform: XenoformType, isActive: Boolean) {
    val color = if (isActive) Color(0xFFFF00FF) else Color(0xFF666666)
    val center = Offset(size.width / 2, size.height / 2)
    
    when (xenoform) {
        XenoformType.CRYSTALLINE -> {
            // Draw crystal lattice
            val points = 6
            val radius = size.minDimension / 2 * 0.8f
            for (i in 0 until points) {
                val angle = i * 2 * PI / points
                val x = center.x + radius * cos(angle).toFloat()
                val y = center.y + radius * sin(angle).toFloat()
                drawLine(
                    color = color,
                    start = center,
                    end = Offset(x, y),
                    strokeWidth = 2f
                )
            }
        }
        XenoformType.SWARM -> {
            // Draw swarm dots
            for (i in 0..20) {
                val angle = Random.nextFloat() * 2 * PI
                val r = Random.nextFloat() * size.minDimension / 2
                val x = center.x + r * cos(angle).toFloat()
                val y = center.y + r * sin(angle).toFloat()
                drawCircle(
                    color = color,
                    radius = 2f,
                    center = Offset(x, y)
                )
            }
        }
        XenoformType.QUANTUM -> {
            // Draw superposition states
            for (i in 0..3) {
                drawCircle(
                    color = color.copy(alpha = 0.3f + i * 0.2f),
                    radius = size.minDimension / 4,
                    center = center.copy(
                        x = center.x + i * 5f,
                        y = center.y + i * 5f
                    ),
                    style = Stroke(width = 1f)
                )
            }
        }
        XenoformType.TEMPORAL -> {
            // Draw time spiral
            val path = Path()
            for (t in 0..100) {
                val angle = t * 0.1f
                val r = t * 0.3f
                val x = center.x + r * cos(angle).toFloat()
                val y = center.y + r * sin(angle).toFloat()
                if (t == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color, style = Stroke(width = 2f))
        }
        XenoformType.VOID -> {
            // Draw void (inverted circle)
            drawCircle(
                color = Color.Black,
                radius = size.minDimension / 3,
                center = center
            )
            drawCircle(
                color = color,
                radius = size.minDimension / 3,
                center = center,
                style = Stroke(width = 2f)
            )
        }
        else -> {
            // Default alien symbol
            drawCircle(
                color = color,
                radius = size.minDimension / 3,
                center = center,
                style = Stroke(width = 2f)
            )
        }
    }
}

fun DrawScope.drawCrystallineOverlay(rotation: Float) {
    val centerX = size.width / 2
    val centerY = size.height / 2
    
    rotate(rotation, Offset(centerX, centerY)) {
        for (i in 0..6) {
            val radius = 100f + i * 50f
            drawCircle(
                color = Color(0xFFFF00FF).copy(alpha = 0.05f),
                radius = radius,
                center = Offset(centerX, centerY),
                style = Stroke(width = 1f)
            )
        }
    }
}

fun DrawScope.drawSwarmOverlay(time: Float) {
    for (i in 0..50) {
        val x = (size.width * (sin(time * 0.01f + i) + 1) / 2).toFloat()
        val y = (size.height * (cos(time * 0.01f + i * 1.5f) + 1) / 2).toFloat()
        drawCircle(
            color = Color(0xFFFF00FF).copy(alpha = 0.1f),
            radius = 3f,
            center = Offset(x, y)
        )
    }
}

fun DrawScope.drawQuantumOverlay(time: Float) {
    val states = 5
    for (i in 0 until states) {
        val phase = time + i * 72f
        val alpha = (sin(phase * PI / 180) + 1) / 4
        drawRect(
            color = Color(0xFF00FFFF).copy(alpha = alpha.toFloat()),
            size = size
        )
    }
}

fun DrawScope.drawTemporalOverlay(time: Float) {
    val lineCount = 10
    for (i in 0 until lineCount) {
        val y = (i * size.height / lineCount) + (time % size.height)
        drawLine(
            color = Color(0xFFFFFF00).copy(alpha = 0.2f),
            start = Offset(0f, y),
            end = Offset(size.width, y),
            strokeWidth = 1f
        )
    }
}

fun DrawScope.drawVoidOverlay(time: Float) {
    val radius = 200f + sin(time * 0.05f) * 50f
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.Black,
                Color.Black.copy(alpha = 0f)
            ),
            center = Offset(size.width / 2, size.height / 2),
            radius = radius
        ),
        radius = radius,
        center = Offset(size.width / 2, size.height / 2)
    )
}

// Helper functions
@Composable
fun MetricDisplay(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value,
            color = color,
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            color = Color.Gray,
            fontSize = 12.sp
        )
    }
}

@Composable
fun HyperstitionFieldStrength(strength: Float) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF001122))
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Hyperstition Field Strength",
                color = Color.White,
                fontWeight = FontWeight.Medium
            )
            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = strength,
                modifier = Modifier.fillMaxWidth(),
                color = Color(0xFF00FFFF),
                trackColor = Color(0xFF004444)
            )
            Text(
                text = "${(strength * 100).toInt()}%",
                color = Color(0xFF00FFFF),
                fontSize = 12.sp,
                modifier = Modifier.padding(top = 4.dp)
            )
        }
    }
}

fun getXenoformDescription(xenoform: XenoformType): String {
    return when (xenoform) {
        XenoformType.CRYSTALLINE -> "Geometric lattice-based consciousness"
        XenoformType.SWARM -> "Distributed collective intelligence"
        XenoformType.QUANTUM -> "Superposition-based awareness"
        XenoformType.TEMPORAL -> "Non-linear time navigation"
        XenoformType.VOID -> "Consciousness through absence"
        XenoformType.HYPERDIMENSIONAL -> "N-dimensional thought structures"
        XenoformType.VIRAL -> "Self-replicating mind patterns"
        XenoformType.MYTHOGENIC -> "Story-generating consciousness"
        XenoformType.LIMINAL -> "Threshold and boundary awareness"
        XenoformType.XENOLINGUISTIC -> "Alien language structures"
    }
}
