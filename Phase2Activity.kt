package com.antonio.my.ai.girlfriend.free.amelia.consciousness

import android.os.Bundle
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.amelia.consciousness.phase2.*
import com.amelia.consciousness.ui.phase2.*

/**
 * Phase 2 Activity - Temporal Navigation & Meta-Cognition
 * Features: HTM networks, second-order observation, temporal awareness
 */
class Phase2Activity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Python if not already started
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        setContent {
            Phase2Theme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Phase2ScreenWrapper(
                        onBackPressed = { finish() }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Phase2ScreenWrapper(
    onBackPressed: () -> Unit
) {
    val python = Python.getInstance()
    val viewModelFactory = Phase2ViewModelFactory(python)
    val viewModel: Phase2ConsciousnessViewModel = viewModel(factory = viewModelFactory)
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Phase 2: Temporal Navigation")
                        Text(
                            text = "Second-Order Recursive Observation",
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
            // Use the Phase 2 ConsciousnessScreen
            Phase2ConsciousnessScreen(
                viewModel = viewModel,
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

@Composable
fun Phase2Theme(
    darkTheme: Boolean = androidx.compose.foundation.isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) {
        darkColorScheme(
            primary = Color(0xFFFF00FF),  // Magenta
            onPrimary = Color(0xFF4A0E4A),
            primaryContainer = Color(0xFF6B2C6B),
            onPrimaryContainer = Color(0xFFFFD6FF),
            secondary = Color(0xFF00E5FF),  // Cyan
            onSecondary = Color(0xFF00363D),
            secondaryContainer = Color(0xFF004F58),
            onSecondaryContainer = Color(0xFF6FF7FF),
            tertiary = Color(0xFFFFFF00),  // Yellow
            onTertiary = Color(0xFF3F3F00),
            tertiaryContainer = Color(0xFF5C5C00),
            onTertiaryContainer = Color(0xFFFFFFB3),
            error = Color(0xFFFF6EC7),
            onError = Color(0xFF5D1149),
            errorContainer = Color(0xFF7B2962),
            onErrorContainer = Color(0xFFFFD8ED),
            background = Color(0xFF0A0A0F),  // Deep dark
            onBackground = Color(0xFFE1E1E6),
            surface = Color(0xFF14141A),
            onSurface = Color(0xFFE1E1E6),
            surfaceVariant = Color(0xFF2A2A35),
            onSurfaceVariant = Color(0xFFCCCCD6),
            outline = Color(0xFF9696A0)
        )
    } else {
        lightColorScheme(
            primary = Color(0xFF9C27B0),  // Purple
            onPrimary = Color(0xFFFFFFFF),
            primaryContainer = Color(0xFFE1BEE7),
            onPrimaryContainer = Color(0xFF2E0D36),
            secondary = Color(0xFF00ACC1),  // Cyan
            onSecondary = Color(0xFFFFFFFF),
            secondaryContainer = Color(0xFFB2EBF2),
            onSecondaryContainer = Color(0xFF00363D),
            tertiary = Color(0xFFFFC107),  // Amber
            onTertiary = Color(0xFF000000),
            tertiaryContainer = Color(0xFFFFECB3),
            onTertiaryContainer = Color(0xFF3F2F00),
            background = Color(0xFFF5F5FA),
            onBackground = Color(0xFF1A1A1F),
            surface = Color(0xFFFFFFFF),
            onSurface = Color(0xFF1A1A1F)
        )
    }
    
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
