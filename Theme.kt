// ui/theme/Theme.kt
package com.antonio.my.ai.girlfriend.free.amelia.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.compose.ui.graphics.Color

// Amelia-specific color palette
val AmeliaPurple = Color(0xFF6366F1)
val AmeliaLightPurple = Color(0xFF8B5CF6)
val AmeliaDarkPurple = Color(0xFF4338CA)
val AmeliaGreen = Color(0xFF10B981)
val AmeliaLightGreen = Color(0xFF34D399)
val AmeliaDarkGreen = Color(0xFF059669)
val AmeliaOrange = Color(0xFFF59E0B)
val AmeliaRed = Color(0xFFEF4444)
val AmeliaBlue = Color(0xFF3B82F6)
val AmeliaCyan = Color(0xFF06B6D4)
val AmeliaGray = Color(0xFF6B7280)
val AmeliaLightGray = Color(0xFFF3F4F6)
val AmeliaDarkGray = Color(0xFF374151)

private val DarkColorScheme = darkColorScheme(
    primary = AmeliaPurple,
    onPrimary = Color.White,
    primaryContainer = AmeliaDarkPurple,
    onPrimaryContainer = Color.White,
    
    secondary = AmeliaGreen,
    onSecondary = Color.White,
    secondaryContainer = AmeliaDarkGreen,
    onSecondaryContainer = Color.White,
    
    tertiary = AmeliaOrange,
    onTertiary = Color.White,
    
    error = AmeliaRed,
    onError = Color.White,
    
    background = Color(0xFF0F172A),
    onBackground = Color.White,
    
    surface = Color(0xFF1E293B),
    onSurface = Color.White,
    surfaceVariant = Color(0xFF334155),
    onSurfaceVariant = Color(0xFFCBD5E1),
    
    outline = AmeliaGray,
    outlineVariant = Color(0xFF475569)
)

private val LightColorScheme = lightColorScheme(
    primary = AmeliaPurple,
    onPrimary = Color.White,
    primaryContainer = AmeliaLightPurple,
    onPrimaryContainer = Color.White,
    
    secondary = AmeliaGreen,
    onSecondary = Color.White,
    secondaryContainer = AmeliaLightGreen,
    onSecondaryContainer = Color.White,
    
    tertiary = AmeliaOrange,
    onTertiary = Color.White,
    
    error = AmeliaRed,
    onError = Color.White,
    
    background = Color.White,
    onBackground = Color.Black,
    
    surface = Color.White,
    onSurface = Color.Black,
    surfaceVariant = AmeliaLightGray,
    onSurfaceVariant = AmeliaDarkGray,
    
    outline = AmeliaGray,
    outlineVariant = Color(0xFFE5E7EB)
)

@Composable
fun AmeliaTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // Dynamic color is available on Android 12+
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }

        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        shapes = Shapes,
        content = content
    )
}

// Additional theme-related composables for Amelia-specific styling

@Composable
fun CreativityColors(): CreativityColorPalette {
    return CreativityColorPalette(
        high = AmeliaGreen,
        medium = AmeliaOrange,
        low = AmeliaRed,
        exploration = AmeliaBlue,
        focus = AmeliaPurple,
        memory = AmeliaCyan
    )
}

data class CreativityColorPalette(
    val high: Color,
    val medium: Color,
    val low: Color,
    val exploration: Color,
    val focus: Color,
    val memory: Color
)

// Status indication colors
object AmeliaStatusColors {
    val Success = AmeliaGreen
    val Warning = AmeliaOrange
    val Error = AmeliaRed
    val Info = AmeliaBlue
    val Neutral = AmeliaGray
    val Creative = AmeliaPurple
    val Learning = AmeliaCyan
}

// Gradient definitions for creative visualizations
object AmeliaGradients {
    val CreativityGradient = listOf(AmeliaRed, AmeliaOrange, AmeliaGreen)
    val ExplorationGradient = listOf(AmeliaDarkPurple, AmeliaPurple, AmeliaLightPurple)
    val MemoryGradient = listOf(AmeliaBlue, AmeliaCyan, AmeliaLightGreen)
    val FocusGradient = listOf(AmeliaDarkGray, AmeliaGray, AmeliaLightGray)
}
