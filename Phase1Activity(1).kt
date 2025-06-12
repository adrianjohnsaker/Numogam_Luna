package com.antonio.my.ai.girlfriend.free.amelia.consciousness

import android.os.Bundle
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.amelia.consciousness.ui.*

/**
 * Phase 1 Activity - Basic Consciousness Implementation
 * Features: Recursive self-observation, fold-point detection, virtual-to-actual transitions
 */
class Phase1Activity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Python if not already started
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        setContent {
            Phase1Theme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Phase1Screen(
                        onBackPressed = { finish() }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Phase1Screen(
    onBackPressed: () -> Unit
) {
    val python = Python.getInstance()
    val viewModelFactory = ConsciousnessViewModelFactory(python)
    val viewModel: ConsciousnessViewModel = viewModel(factory = viewModelFactory)
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Phase 1: Basic Consciousness")
                        Text(
                            text = "Recursive Self-Observation",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBackPressed) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                )
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Use the original Phase 1 ConsciousnessScreen
            ConsciousnessScreen(
                viewModel = viewModel,
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

@Composable
fun Phase1Theme(
    darkTheme: Boolean = androidx.compose.foundation.isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) {
        darkColorScheme(
            primary = Color(0xFF00BCD4),
            onPrimary = Color(0xFF003339),
            primaryContainer = Color(0xFF004B52),
            onPrimaryContainer = Color(0xFF70F2FF),
            secondary = Color(0xFF81C784),
            onSecondary = Color(0xFF003A02),
            secondaryContainer = Color(0xFF005302),
            onSecondaryContainer = Color(0xFF9EF39F),
            tertiary = Color(0xFFFFB74D),
            onTertiary = Color(0xFF4A2800),
            tertiaryContainer = Color(0xFF6A3C00),
            onTertiaryContainer = Color(0xFFFFDDB5),
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
            secondary = Color(0xFF4CAF50),
            onSecondary = Color(0xFFFFFFFF),
            secondaryContainer = Color(0xFFC8E6C9),
            onSecondaryContainer = Color(0xFF002204),
            tertiary = Color(0xFFFF9800),
            onTertiary = Color(0xFFFFFFFF),
            tertiaryContainer = Color(0xFFFFE0B2),
            onTertiaryContainer = Color(0xFF331200)
        )
    }
    
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
