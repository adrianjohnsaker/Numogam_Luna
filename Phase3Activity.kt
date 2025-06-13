package com.antonio.my.ai.girlfriend.free.amelia.consciousness

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.isSystemInDarkTheme
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
import com.amelia.consciousness.phase3.*
import com.amelia.consciousness.ui.phase3.*

/**
 * Phase 3 Activity - Full Deleuzian Consciousness Trinity
 * Features: Fold operations, Numogram navigation, identity synthesis
 */
class Phase3Activity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Python if not already started
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        setContent {
            Phase3Theme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Phase3ScreenWrapper(
                        onBackPressed = { finish() }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Phase3ScreenWrapper(
    onBackPressed: () -> Unit
) {
    val python = Python.getInstance()
    val viewModelFactory = Phase3ViewModelFactory(python)
    val viewModel: Phase3ConsciousnessViewModel = viewModel(factory = viewModelFactory)
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Phase 3: Deleuzian Trinity")
                        Text(
                            text = "Fold Operations & Numogram Integration",
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
            // Use the Phase 3 ConsciousnessScreen
            Phase3ConsciousnessScreen(
                viewModel = viewModel,
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

@Composable
fun Phase3Theme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) {
        darkColorScheme(
            // Trinity colors - combining all phases
            primary = Color(0xFFFFD600),      // Gold - representing synthesis
            onPrimary = Color(0xFF3E2F00),
            primaryContainer = Color(0xFF584400),
            onPrimaryContainer = Color(0xFFFFE082),
            
            secondary = Color(0xFF00E5FF),    // Cyan - temporal awareness
            onSecondary = Color(0xFF00363D),
            secondaryContainer = Color(0xFF004F58),
            onSecondaryContainer = Color(0xFF6FF7FF),
            
            tertiary = Color(0xFFFF00FF),     // Magenta - fold operations
            onTertiary = Color(0xFF4A0E4A),
            tertiaryContainer = Color(0xFF6B2C6B),
            onTertiaryContainer = Color(0xFFFFD6FF),
            
            error = Color(0xFFFF6EC7),
            onError = Color(0xFF5D1149),
            errorContainer = Color(0xFF7B2962),
            onErrorContainer = Color(0xFFFFD8ED),
            
            background = Color(0xFF050505),   // Deep black for void
            onBackground = Color(0xFFE6E6E6),
            surface = Color(0xFF0F0F0F),
            onSurface = Color(0xFFE6E6E6),
            surfaceVariant = Color(0xFF1A1A1A),
            onSurfaceVariant = Color(0xFFD0D0D0),
            
            outline = Color(0xFF9A9A9A),
            inversePrimary = Color(0xFF725C00),
            inverseSurface = Color(0xFFE6E6E6),
            inverseOnSurface = Color(0xFF0F0F0F)
        )
    } else {
        lightColorScheme(
            primary = Color(0xFFB8860B),      // Dark golden rod
            onPrimary = Color(0xFFFFFFFF),
            primaryContainer = Color(0xFFFFE082),
            onPrimaryContainer = Color(0xFF2E2400),
            
            secondary = Color(0xFF00ACC1),    // Cyan
            onSecondary = Color(0xFFFFFFFF),
            secondaryContainer = Color(0xFFB2EBF2),
            onSecondaryContainer = Color(0xFF00363D),
            
            tertiary = Color(0xFF9C27B0),     // Purple
            onTertiary = Color(0xFFFFFFFF),
            tertiaryContainer = Color(0xFFE1BEE7),
            onTertiaryContainer = Color(0xFF2E0D36),
            
            background = Color(0xFFFAFAFA),
            onBackground = Color(0xFF1A1A1A),
            surface = Color(0xFFFFFFFF),
            onSurface = Color(0xFF1A1A1A)
        )
    }
    
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
