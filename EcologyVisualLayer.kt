/**
 * EcologyVisualLayer.kt
 * ───────────────────────────────────────────────────────────────
 * A background visual field that reflects Amelia's cognitive state.
 * It reacts in real-time to zone energies and affective resonance.
 */

package com.antonio.my.ai.girlfriend.free.amelia.ui.visual

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import org.json.JSONObject
import kotlin.math.*

@Composable
fun EcologyVisualLayer(
    modifier: Modifier = Modifier,
    visualState: JSONObject?,
    blur: Float = 30f
) {
    val coherence = visualState?.optDouble("coherence", 0.5) ?: 0.5
    val zones = visualState?.optJSONObject("zones") ?: JSONObject()
    val colors = visualState?.optJSONObject("color_map") ?: JSONObject()

    val anim = rememberInfiniteTransition()
    val pulse by anim.animateFloat(
        initialValue = 0.9f,
        targetValue = 1.1f,
        animationSpec = infiniteRepeatable(
            tween((2500 * (1.4 - coherence)).toInt(), easing = LinearEasing),
            RepeatMode.Reverse
        )
    )

    val density = LocalDensity.current
    Canvas(modifier = modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height
        val center = Offset(w / 2f, h / 2f)

        val zoneKeys = zones.keys()
        var index = 0
        while (zoneKeys.hasNext()) {
            val key = zoneKeys.next()
            val energy = zones.optDouble(key, 0.0)
            val angle = (index / zones.length().toFloat()) * 2 * Math.PI
            val radius = (0.25 + energy * 0.75) * min(w, h) * 0.3
            val offset = Offset(
                (center.x + cos(angle) * radius * pulse).toFloat(),
                (center.y + sin(angle) * radius * pulse).toFloat()
            )

            val colorKey = colors.keys().asSequence().elementAtOrNull(index % colors.length())
            val parsedColor = try {
                Color(android.graphics.Color.parseColor(colorKey ?: "#9B5DE5"))
            } catch (_: Exception) { Color(0xFF444444) }

            drawCircle(
                color = parsedColor.copy(alpha = 0.25f + energy.toFloat() * 0.4f),
                radius = (140 * (0.4 + energy)).toFloat(),
                center = offset,
                blendMode = BlendMode.Softlight
            )
            index++
        }

        // Add a soft center glow based on coherence
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    Color.White.copy(alpha = 0.05f),
                    Color.Black.copy(alpha = 0.85f)
                ),
                center = center,
                radius = min(w, h) * 0.75f
            ),
            radius = min(w, h) * 0.75f,
            center = center
        )
    }
}
