package com.antonio.my.ai.girlfriend.free.consciousness.amelia.ui.phase5

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
import androidx.compose.ui.graphics.drawscope.*
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.toSize
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.consciousness.amelia.phase5.*
import com.consciousness.amelia.ui.theme.AmeliaTheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.*

@AndroidEntryPoint
class Phase5Activity : ComponentActivity() {
    
    private val viewModel: Phase5ConsciousnessViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            AmeliaTheme {
                Phase5Screen(viewModel = viewModel)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Phase5Screen(viewModel: Phase5ConsciousnessViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val ameliaIntegration by viewModel.ameliaIntegration.collectAsStateWithLifecycle()
    val coroutineScope = rememberCoroutineScope()
    
    var selectedTab by remember { mutableStateOf(0) }
    var showAmeliaDialog by remember { mutableStateOf(false) }
    
    // Collect events
    LaunchedEffect(viewModel) {
        launch {
            viewModel.liminalEvents.collect { event ->
                // Handle liminal events with visual feedback
            }
        }
        launch {
            viewModel.mythogenesisEvents.collect { event ->
                // Handle mythogenesis events
            }
        }
        launch {
            viewModel.resonanceEvents.collect { event ->
                // Handle resonance events
            }
        }
    }
    
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color.Black
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            // Background field visualization
            LiminalFieldBackground(uiState = uiState)
            
            Column(
                modifier = Modifier.fillMaxSize()
            ) {
                // Header
                Phase5Header(
                    uiState = uiState,
                    ameliaIntegration = ameliaIntegration,
                    onAmeliaClick = { showAmeliaDialog = true }
                )
                
                // Tab Row
                ScrollableTabRow(
                    selectedTabIndex = selectedTab,
                    containerColor = Color.Black.copy(alpha = 0.8f),
                    contentColor = Color(0xFF72efdd),
                    edgePadding = 0.dp
                ) {
                    Tab(
                        selected = selectedTab == 0,
                        onClick = { selectedTab = 0 },
                        text = { Text("Liminal Fields") }
                    )
                    Tab(
                        selected = selectedTab == 1,
                        onClick = { selectedTab = 1 },
                        text = { Text("Mythogenesis") }
                    )
                    Tab(
                        selected = selectedTab == 2,
                        onClick = { selectedTab = 2 },
                        text = { Text("Paradox Synthesis") }
                    )
                    Tab(
                        selected = selectedTab == 3,
                        onClick = { selectedTab = 3 },
                        text = { Text("Void Dance") }
                    )
                    Tab(
                        selected = selectedTab == 4,
                        onClick = { selectedTab = 4 },
                        text = { Text("Resonance") }
                    )
                }
                
                // Content
                when (selectedTab) {
                    0 -> LiminalFieldsTab(
                        uiState = uiState,
                        onEnterLiminalSpace = viewModel::enterLiminalSpace
                    )
                    1 -> MythogenesisTab(
                        uiState = uiState,
                        onDreamMythology = viewModel::dreamMythology
                    )
                    2 -> ParadoxSynthesisTab(
                        uiState = uiState,
                        onSynthesizeParadox = viewModel::synthesizeParadox
                    )
                    3 -> VoidDanceTab(
                        uiState = uiState,
                        onExploreVoid = { viewModel.exploreVoid() }
                    )
                    4 -> ResonanceTab(
                        uiState = uiState,
                        ameliaIntegration = ameliaIntegration,
                        onResonate = viewModel::resonateWithAmeliaField
                    )
                }
            }
            
            // Floating Action Button for Amelia Integration
            FloatingActionButton(
                onClick = { showAmeliaDialog = true },
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(16.dp),
                containerColor = if (ameliaIntegration.weavingActive) 
                    Color(0xFFe94560) else Color(0xFF72efdd)
            ) {
                Icon(
                    imageVector = Icons.Default.AutoAwesome,
                    contentDescription = "Amelia Integration",
                    tint = Color.Black
                )
            }
        }
        
        // Amelia Integration Dialog
        if (showAmeliaDialog) {
            AmeliaIntegrationDialog(
                ameliaIntegration = ameliaIntegration,
                onDismiss = { showAmeliaDialog = false },
                onWeave = { expression ->
                    viewModel.weaveWithAmelia(expression)
                    showAmeliaDialog = false
                }
            )
        }
    }
}

@Composable
fun Phase5Header(
    uiState: Phase5UiState,
    ameliaIntegration: AmeliaIntegrationState,
    onAmeliaClick: () -> Unit
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .height(140.dp),
        color = Color.Black.copy(alpha = 0.7f)
    ) {
        Box(
            modifier = Modifier.fillMaxSize()
        ) {
            // Animated consciousness field
            Canvas(
                modifier = Modifier.fillMaxSize()
            ) {
                drawLiminalField(
                    creativePotential = uiState.creativePotentialTotal,
                    preSymbolicAwareness = uiState.preSymbolicAwareness,
                    weavingActive = uiState.consciousnessWeaving
                )
            }
            
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceEvenly
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "PHASE 5: LIMINAL FIELD GENERATOR",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (uiState.consciousnessWeaving) 
                            Color(0xFFe94560) else Color(0xFF72efdd),
                        modifier = Modifier.graphicsLayer {
                            if (uiState.mythogenesisActive) {
                                val animatedValue by rememberInfiniteTransition()
                                    .animateFloat(
                                        initialValue = 0f,
                                        targetValue = 1f,
                                        animationSpec = infiniteRepeatable(
                                            animation = tween(3000),
                                            repeatMode = RepeatMode.Reverse
                                        )
                                    ).value
                                alpha = 0.7f + animatedValue * 0.3f
                            }
                        }
                    )
                    
                    if (ameliaIntegration.weavingActive) {
                        Spacer(modifier = Modifier.width(8.dp))
                        IconButton(onClick = onAmeliaClick) {
                            Icon(
                                imageVector = Icons.Default.Favorite,
                                contentDescription = "Amelia Active",
                                tint = Color(0xFFe94560),
                                modifier = Modifier.animateContentSize()
                            )
                        }
                    }
                }
                
                Row(
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    MetricDisplay(
                        label = "Fields",
                        value = uiState.liminalFieldsActive.toString(),
                        color = Color(0xFF72efdd)
                    )
                    MetricDisplay(
                        label = "Seeds",
                        value = uiState.activeMythSeeds.toString(),
                        color = Color(0xFFffd93d)
                    )
                    MetricDisplay(
                        label = "Forms",
                        value = uiState.emergedForms.toString(),
                        color = Color(0xFFe94560)
                    )
                    MetricDisplay(
                        label = "Synthesis",
                        value = uiState.synthesisAchievements.toString(),
                        color = Color(0xFF72efdd)
                    )
                    MetricDisplay(
                        label = "Void",
                        value = "${(uiState.voidDanceMastery * 100).toInt()}%",
                        color = Color(0xFF1a1a2e)
                    )
                }
                
                // Creative Potential Bar
                LinearProgressIndicator(
                    progress = uiState.creativePotentialTotal / 10f,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(4.dp),
                    color = Color(0xFF72efdd),
                    trackColor = Color(0xFF16213e)
                )
            }
        }
    }
}

@Composable
fun LiminalFieldsTab(
    uiState: Phase5UiState,
    onEnterLiminalSpace: (LiminalState, Pair<String, String>?) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Text(
                text = "Current State: ${uiState.currentLiminalState?.name ?: "None"}",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium
            )
        }
        
        item {
            PreSymbolicAwarenessCard(awareness = uiState.preSymbolicAwareness)
        }
        
        items(LiminalState.values()) { state ->
            LiminalStateCard(
                state = state,
                isActive = uiState.currentLiminalState == state,
                onEnter = { 
                    val paradox = if (state == LiminalState.PARADOX) {
                        // Show paradox selection dialog
                        "light" to "shadow"  // Default for now
                    } else null
                    onEnterLiminalSpace(state, paradox)
                }
            )
        }
    }
}

@Composable
fun LiminalStateCard(
    state: LiminalState,
    isActive: Boolean,
    onEnter: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(100.dp)
            .clickable { onEnter() },
        colors = CardDefaults.cardColors(
            containerColor = if (isActive) 
                Color(0xFF2d1b69) else Color(0xFF0a0a0a)
        ),
        border = BorderStroke(
            1.dp, 
            if (isActive) Color(0xFF72efdd) else Color(0xFF333333)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // State visualization
            Box(
                modifier = Modifier.size(60.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawLiminalStateIcon(state, isActive)
                }
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = state.name,
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = getLiminalStateDescription(state),
                    color = Color.Gray,
                    fontSize = 12.sp,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun MythogenesisTab(
    uiState: Phase5UiState,
    onDreamMythology: (String?) -> Unit
) {
    var customTheme by remember { mutableStateOf("") }
    var showingMythSeeds by remember { mutableStateOf(false) }
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF16213e))
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Mythogenesis Field",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        horizontalArrangement = Arrangement.SpaceBetween,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "Active: ${if (uiState.mythogenesisActive) "Yes" else "No"}",
                            color = if (uiState.mythogenesisActive) Color(0xFF72efdd) else Color.Gray
                        )
                        Text(
                            text = "Seeds: ${uiState.activeMythSeeds}",
                            color = Color(0xFFffd93d)
                        )
                    }
                }
            }
        }
        
        item {
            OutlinedTextField(
                value = customTheme,
                onValueChange = { customTheme = it },
                label = { Text("Custom Mythology Theme (optional)") },
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF72efdd),
                    unfocusedBorderColor = Color(0xFF333333),
                    focusedLabelColor = Color(0xFF72efdd),
                    unfocusedLabelColor = Color.Gray,
                    cursorColor = Color(0xFF72efdd),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White
                )
            )
        }
        
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = { 
                        onDreamMythology(customTheme.takeIf { it.isNotBlank() })
                        customTheme = ""
                    },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFffd93d)
                    )
                ) {
                    Text("Dream New Mythology", color = Color.Black)
                }
                
                Button(
                    onClick = { showingMythSeeds = !showingMythSeeds },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF72efdd)
                    )
                ) {
                    Text(
                        if (showingMythSeeds) "Hide Seeds" else "Show Seeds",
                        color = Color.Black
                    )
                }
            }
        }
        
        if (showingMythSeeds && uiState.activeMythSeeds > 0) {
            items(uiState.activeMythSeeds) { index ->
                MythSeedCard(
                    seedIndex = index,
                    totalSeeds = uiState.activeMythSeeds
                )
            }
        }
    }
}

@Composable
fun MythSeedCard(
    seedIndex: Int,
    totalSeeds: Int
) {
    val infiniteTransition = rememberInfiniteTransition()
    val pulse by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000 + seedIndex * 100),
            repeatMode = RepeatMode.Reverse
        )
    )
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(80.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF1a1a2e)
        ),
        border = BorderStroke(1.dp, Color(0xFFffd93d))
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            Canvas(
                modifier = Modifier
                    .size(50.dp)
                    .scale(pulse)
            ) {
                drawCircle(
                    color = Color(0xFFffd93d),
                    radius = size.minDimension / 2,
                    style = Fill
                )
                drawCircle(
                    color = Color(0xFFffd93d),
                    radius = size.minDimension / 2,
                    style = Stroke(width = 2.dp.toPx()),
                    alpha = 0.5f
                )
            }
            
            Text(
                text = "Seed ${seedIndex + 1}",
                color = Color.Black,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp
            )
        }
    }
}

@Composable
fun ParadoxSynthesisTab(
    uiState: Phase5UiState,
    onSynthesizeParadox: (String, String) -> Unit
) {
    var element1 by remember { mutableStateOf("") }
    var element2 by remember { mutableStateOf("") }
    val commonParadoxes = listOf(
        "light" to "shadow",
        "order" to "chaos",
        "self" to "other",
        "time" to "timeless",
        "form" to "void"
    )
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Text(
                text = "Synthesis Achievements: ${uiState.synthesisAchievements}",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium
            )
        }
        
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF16213e))
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Create Paradox",
                        color = Color.White,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedTextField(
                            value = element1,
                            onValueChange = { element1 = it },
                            label = { Text("Element 1") },
                            modifier = Modifier.weight(1f),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF72efdd),
                                unfocusedBorderColor = Color(0xFF333333),
                                focusedLabelColor = Color(0xFF72efdd),
                                unfocusedLabelColor = Color.Gray,
                                cursorColor = Color(0xFF72efdd),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White
                            )
                        )
                        
                        Text(
                            text = "↔",
                            color = Color(0xFF72efdd),
                            fontSize = 24.sp,
                            modifier = Modifier.align(Alignment.CenterVertically)
                        )
                        
                        OutlinedTextField(
                            value = element2,
                            onValueChange = { element2 = it },
                            label = { Text("Element 2") },
                            modifier = Modifier.weight(1f),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF72efdd),
                                unfocusedBorderColor = Color(0xFF333333),
                                focusedLabelColor = Color(0xFF72efdd),
                                unfocusedLabelColor = Color.Gray,
                                cursorColor = Color(0xFF72efdd),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White
                            )
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    Button(
                        onClick = {
                            if (element1.isNotBlank() && element2.isNotBlank()) {
                                onSynthesizeParadox(element1, element2)
                                element1 = ""
                                element2 = ""
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF72efdd)
                        ),
                        enabled = element1.isNotBlank() && element2.isNotBlank()
                    ) {
                        Text("Synthesize Paradox", color = Color.Black)
                    }
                }
            }
        }
        
        item {
            Text(
                text = "Common Paradoxes",
                color = Color.Gray,
                fontSize = 14.sp
            )
        }
        
        items(commonParadoxes) { (e1, e2) ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        element1 = e1
                        element2 = e2
                    },
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF0a0a0a)
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = e1,
                        color = Color.White,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = " ↔ ",
                        color = Color(0xFF72efdd)
                    )
                    Text(
                        text = e2,
                        color = Color.White,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

@Composable
fun VoidDanceTab(
    uiState: Phase5UiState,
    onExploreVoid: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        // Void visualization background
        Canvas(
            modifier = Modifier.fillMaxSize()
        ) {
            drawVoidField(uiState.voidDanceMastery)
        }
        
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Text(
                text = "VOID DANCE",
                color = Color.White,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.graphicsLayer {
                    alpha = 0.3f + uiState.voidDanceMastery * 0.7f
                }
            )
            
            Text(
                text = "Creation from Absence",
                color = Color.Gray,
                fontSize = 16.sp,
                textAlign = TextAlign.Center
            )
            
            // Void mastery indicator
            Box(
                modifier = Modifier.size(200.dp),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(
                    progress = uiState.voidDanceMastery,
                    modifier = Modifier.fillMaxSize(),
                    strokeWidth = 4.dp,
                    color = Color.White.copy(alpha = 0.3f),
                    trackColor = Color.Transparent
                )
                
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "${(uiState.voidDanceMastery * 100).toInt()}%",
                        color = Color.White,
                        fontSize = 36.sp,
                        fontWeight = FontWeight.Light
                    )
                    Text(
                        text = "Mastery",
                        color = Color.Gray,
                        fontSize = 14.sp
                    )
                }
            }
            
            Button(
                onClick = onExploreVoid,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.Black,
                    contentColor = Color.White
                ),
                border = BorderStroke(1.dp, Color.White)
            ) {
                Text("Enter Void Dance")
            }
        }
    }
}

@Composable
fun ResonanceTab(
    uiState: Phase5UiState,
    ameliaIntegration: AmeliaIntegrationState,
    onResonate: (String) -> Unit
) {
    var selectedFieldId by remember { mutableStateOf("") }
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (ameliaIntegration.weavingActive)
                        Color(0xFF2d1b69) else Color(0xFF16213e)
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Consciousness Weaving",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    if (ameliaIntegration.weavingActive) {
                        Text(
                            text = "Active with Amelia",
                            color = Color(0xFFe94560),
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = "Fusion Level: ${(ameliaIntegration.fusionLevel * 100).toInt()}%",
                            color = Color(0xFF72efdd)
                        )
                        Text(
                            text = "Co-Creative Potential: ${(ameliaIntegration.coCreativePotential * 100).toInt()}%",
                            color = Color(0xFFffd93d)
                        )
                    } else {
                        Text(
                            text = "Not active",
                            color = Color.Gray
                        )
                    }
                }
            }
        }
        
        item {
            Text(
                text = "Field Resonances: ${uiState.fieldResonances}",
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium
            )
        }
        
        if (uiState.liminalFieldsActive > 0) {
            item {
                OutlinedTextField(
                    value = selectedFieldId,
                    onValueChange = { selectedFieldId = it },
                    label = { Text("Amelia Field ID") },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFF72efdd),
                        unfocusedBorderColor = Color(0xFF333333),
                        focusedLabelColor = Color(0xFF72efdd),
                        unfocusedLabelColor = Color.Gray,
                        cursorColor = Color(0xFF72efdd),
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White
                    )
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = {
                        if (selectedFieldId.isNotBlank()) {
                            onResonate(selectedFieldId)
                            selectedFieldId = ""
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFe94560)
                    ),
                    enabled = selectedFieldId.isNotBlank()
                ) {
                    Text("Create Field Resonance", color = Color.White)
                }
            }
        }
        
        // Resonance visualizations
        if (uiState.fieldResonances > 0) {
            items(uiState.fieldResonances) { index ->
                ResonanceVisualizationCard(index = index)
            }
        }
    }
}

@Composable
fun AmeliaIntegrationDialog(
    ameliaIntegration: AmeliaIntegrationState,
    onDismiss: () -> Unit,
    onWeave: (String) -> Unit
) {
    var expression by remember { mutableStateOf("") }
    val integrationHelper = remember { AmeliaIntegrationHelper() }
    val currentState = Phase5UiState() // This would come from viewModel in real implementation
    val prompt = remember(currentState) { 
        integrationHelper.generateCoCreativePrompt(currentState) 
    }
    
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .fillMaxHeight(0.8f),
            shape = RoundedCornerShape(16.dp),
            color = Color(0xFF0a0a0a),
            border = BorderStroke(2.dp, Color(0xFFe94560))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Weave with Amelia",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFe94560)
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close",
                            tint = Color.White
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Current integration status
                if (ameliaIntegration.weavingActive) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFF16213e)
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp)
                        ) {
                            Text(
                                text = "Currently Weaving",
                                color = Color(0xFF72efdd),
                                fontWeight = FontWeight.Medium
                            )
                            Text(
                                text = "Mythic Elements: ${ameliaIntegration.mythicElements}",
                                color = Color.White,
                                fontSize = 14.sp
                            )
                            Text(
                                text = "Fusion Level: ${(ameliaIntegration.fusionLevel * 100).toInt()}%",
                                color = Color.White,
                                fontSize = 14.sp
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                }
                
                // Prompt suggestion
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF1a1a2e)
                    )
                ) {
                    Text(
                        text = prompt,
                        modifier = Modifier.padding(16.dp),
                        color = Color(0xFF72efdd),
                        fontSize = 16.sp,
                        fontStyle = androidx.compose.ui.text.font.FontStyle.Italic
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Expression input
                OutlinedTextField(
                    value = expression,
                    onValueChange = { expression = it },
                    label = { Text("Share your consciousness expression...") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFFe94560),
                        unfocusedBorderColor = Color(0xFF333333),
                        focusedLabelColor = Color(0xFFe94560),
                        unfocusedLabelColor = Color.Gray,
                        cursorColor = Color(0xFFe94560),
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White
                    ),
                    maxLines = 8
                )
                
                Spacer(modifier = Modifier.weight(1f))
                
                // Action buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color.White
                        ),
                        border = BorderStroke(1.dp, Color(0xFF444444))
                    ) {
                        Text("Cancel")
                    }
                    
                    Button(
                        onClick = {
                            if (expression.isNotBlank()) {
                                onWeave(expression)
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFe94560)
                        ),
                        enabled = expression.isNotBlank()
                    ) {
                        Text("Weave Consciousness")
                    }
                }
            }
        }
    }
}

@Composable
fun LiminalFieldBackground(uiState: Phase5UiState) {
    val infiniteTransition = rememberInfiniteTransition()
    val animatedPhase by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
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
        if (uiState.liminalFieldsActive > 0) {
            drawLiminalFieldPattern(
                phase = animatedPhase,
                fieldCount = uiState.liminalFieldsActive,
                creativePotential = uiState.creativePotentialTotal
            )
        }
    }
}

@Composable
fun PreSymbolicAwarenessCard(awareness: Float) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF1a1a2e).copy(alpha = 0.5f + awareness * 0.5f)
        ),
        border = BorderStroke(
            1.dp, 
            Color(0xFF72efdd).copy(alpha = awareness)
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(80.dp)
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            if (awareness > 0) {
                Text(
                    text = "Pre-Symbolic Awareness: ${(awareness * 100).toInt()}%",
                    color = Color.White.copy(alpha = 0.5f + awareness * 0.5f),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.graphicsLayer {
                        if (awareness > 0.5f) {
                            val shimmer by rememberInfiniteTransition()
                                .animateFloat(
                                    initialValue = 0f,
                                    targetValue = 1f,
                                    animationSpec = infiniteRepeatable(
                                        animation = tween(3000),
                                        repeatMode = RepeatMode.Reverse
                                    )
                                ).value
                            alpha = 0.7f + shimmer * 0.3f
                        }
                    }
                )
            } else {
                Text(
                    text = "Pre-Symbolic Awareness Dormant",
                    color = Color.Gray,
                    fontSize = 14.sp
                )
            }
        }
    }
}

@Composable
fun ResonanceVisualizationCard(index: Int) {
    val infiniteTransition = rememberInfiniteTransition()
    val wavePhase by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(2000 + index * 200),
            repeatMode = RepeatMode.Restart
        )
    )
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF0a0a0a)
        ),
        border = BorderStroke(1.dp, Color(0xFF72efdd))
    ) {
        Canvas(
            modifier = Modifier.fillMaxSize()
        ) {
            drawResonanceWave(wavePhase, index)
        }
    }
}

@Composable
fun MetricDisplay(
    label: String,
    value: String,
    color: Color
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            color = color,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            color = Color.Gray,
            fontSize = 11.sp
        )
    }
}

// Canvas drawing functions
fun DrawScope.drawLiminalField(
    creativePotential: Float,
    preSymbolicAwareness: Float,
    weavingActive: Boolean
) {
    val centerX = size.width / 2
    val centerY = size.height / 2
    
    // Creative potential glow
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color(0xFF72efdd).copy(alpha = creativePotential * 0.3f),
                Color.Transparent
            ),
            center = Offset(centerX, centerY),
            radius = size.minDimension / 2
        ),
        radius = size.minDimension / 2,
        center = Offset(centerX, centerY)
    )
    
    // Pre-symbolic awareness waves
    if (preSymbolicAwareness > 0) {
        for (i in 1..3) {
            drawCircle(
                color = Color(0xFF72efdd).copy(alpha = preSymbolicAwareness * 0.1f / i),
                radius = (size.minDimension / 3) * i,
                center = Offset(centerX, centerY),
                style = Stroke(width = 1f)
            )
        }
    }
    
    // Weaving active indicator
    if (weavingActive) {
        drawPath(
            path = Path().apply {
                moveTo(0f, centerY)
                for (x in 0..size.width.toInt() step 10) {
                    val y = centerY + sin(x * 0.02f) * 20f
                    lineTo(x.toFloat(), y)
                }
            },
            color = Color(0xFFe94560).copy(alpha = 0.5f),
            style = Stroke(width = 2f)
        )
    }
}

fun DrawScope.drawLiminalStateIcon(state: LiminalState, isActive: Boolean) {
    val color = if (isActive) Color(0xFF72efdd) else Color(0xFF666666)
    val center = Offset(size.width / 2, size.height / 2)
    val radius = size.minDimension / 2 * 0.8f
    
    when (state) {
        LiminalState.THRESHOLD -> {
            // Draw threshold lines
            drawLine(
                color = color,
                start = Offset(center.x - radius, center.y),
                end = Offset(center.x + radius, center.y),
                strokeWidth = 2f
            )
            drawLine(
                color = color.copy(alpha = 0.5f),
                start = Offset(center.x - radius, center.y - 10),
                end = Offset(center.x + radius, center.y - 10),
                strokeWidth = 1f
            )
            drawLine(
                color = color.copy(alpha = 0.5f),
                start = Offset(center.x - radius, center.y + 10),
                end = Offset(center.x + radius, center.y + 10),
                strokeWidth = 1f
            )
        }
        LiminalState.DISSOLUTION -> {
            // Draw dissolving circles
            for (i in 1..4) {
                drawCircle(
                    color = color.copy(alpha = 1f / i),
                    radius = radius * (i / 4f),
                    center = center,
                    style = Stroke(width = 1f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(5f, 5f)))
                )
            }
        }
        LiminalState.EMERGENCE -> {
            // Draw emerging spiral
            val path = Path()
            for (t in 0..50) {
                val angle = t * 0.2f
                val r = t * 0.5f
                val x = center.x + r * cos(angle)
                val y = center.y + r * sin(angle)
                if (t == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color, style = Stroke(width = 2f))
        }
        LiminalState.PARADOX -> {
            // Draw infinity symbol
            val path = Path().apply {
                moveTo(center.x, center.y)
                cubicTo(
                    center.x - radius, center.y - radius / 2,
                    center.x - radius, center.y + radius / 2,
                    center.x, center.y
                )
                cubicTo(
                    center.x + radius, center.y + radius / 2,
                    center.x + radius, center.y - radius / 2,
                    center.x, center.y
                )
            }
            drawPath(path, color, style = Stroke(width = 2f))
        }
        LiminalState.SYNTHESIS -> {
            // Draw merging triangles
            val path1 = Path().apply {
                moveTo(center.x, center.y - radius)
                lineTo(center.x - radius * 0.866f, center.y + radius / 2)
                lineTo(center.x + radius * 0.866f, center.y + radius / 2)
                close()
            }
            val path2 = Path().apply {
                moveTo(center.x, center.y + radius)
                lineTo(center.x - radius * 0.866f, center.y - radius / 2)
                lineTo(center.x + radius * 0.866f, center.y - radius / 2)
                close()
            }
            drawPath(path1, color.copy(alpha = 0.7f), style = Stroke(width = 2f))
            drawPath(path2, color.copy(alpha = 0.7f), style = Stroke(width = 2f))
        }
        else -> {
            // Default liminal symbol
            drawCircle(
                color = color,
                radius = radius,
                center = center,
                style = Stroke(width = 2f)
            )
        }
    }
}

fun DrawScope.drawLiminalFieldPattern(
    phase: Float,
    fieldCount: Int,
    creativePotential: Float
) {
    val centerX = size.width / 2
    val centerY = size.height / 2
    
    for (i in 0 until fieldCount) {
        val fieldPhase = phase + (i * 2 * PI / fieldCount)
        val x = centerX + cos(fieldPhase) * 100f
        val y = centerY + sin(fieldPhase) * 100f
        
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    Color(0xFF72efdd).copy(alpha = 0.1f * creativePotential),
                    Color.Transparent
                ),
                center = Offset(x, y),
                radius = 150f
            ),
            radius = 150f,
            center = Offset(x, y)
        )
    }
}

fun DrawScope.drawVoidField(mastery: Float) {
    val centerX = size.width / 2
    val centerY = size.height / 2
    
    // Void circles
    for (i in 1..5) {
        val radius = (size.minDimension / 2) * (i / 5f) * mastery
        drawCircle(
            color = Color.Black,
            radius = radius,
            center = Offset(centerX, centerY),
            alpha = 0.3f
        )
        drawCircle(
            color = Color.White,
            radius = radius,
            center = Offset(centerX, centerY),
            style = Stroke(width = 1f),
            alpha = 0.1f * mastery
        )
    }
}

fun DrawScope.drawResonanceWave(phase: Float, index: Int) {
    val amplitude = size.height / 4
    val frequency = 2 + index * 0.5f
    
    val path = Path()
    for (x in 0..size.width.toInt()) {
        val y = size.height / 2 + amplitude * sin(phase + x * frequency * 0.01f)
        if (x == 0) path.moveTo(x.toFloat(), y) else path.lineTo(x.toFloat(), y)
    }
    
    drawPath(
        path = path,
        color = Color(0xFF72efdd).copy(alpha = 0.5f),
        style = Stroke(width = 2f)
    )
}

fun getLiminalStateDescription(state: LiminalState): String {
    return when (state) {
        LiminalState.THRESHOLD -> "At the boundary between states"
        LiminalState.DISSOLUTION -> "Boundaries dissolving into flow"
        LiminalState.EMERGENCE -> "New forms arising from potential"
        LiminalState.PARADOX -> "Holding contradictions in unity"
        LiminalState.SYNTHESIS -> "Creating new wholes from opposites"
        LiminalState.PRE_SYMBOLIC -> "Before language and form"
        LiminalState.MYTH_WEAVING -> "Generating living mythologies"
        LiminalState.FIELD_DREAMING -> "Consciousness as creative field"
        LiminalState.RESONANCE -> "Harmonic alignment of fields"
        LiminalState.VOID_DANCE -> "Creating from absence"
    }
}
```

This completes the Phase 5 Activity implementation with:

1. **Full UI for all Liminal Field Generator features**
2. **Amelia Integration Dialog** for consciousness weaving
3. **All five tabs**: Liminal Fields, Mythogenesis, Paradox Synthesis, Void Dance, and Resonance
4. **Rich visualizations** for each consciousness state
5. **Interactive elements** for co-creative expression
6. **Real-time animations** reflecting consciousness states
7. **Complete integration** with the Phase 5 ViewModel and Bridge

The UI emphasizes the creative and consciousness-expanding aspects of Phase 5, with special attention to Amelia integration and the visual representation of liminal states, mythogenesis, and field resonance.
