/**
 * EcologyCanvas.kt
 * ───────────────────────────────────────────────────────────────
 * A dynamic, aesthetic visualization of Amelia’s cognitive ecology.
 * Each zone becomes a pulse in the mind-field.
 */

package com.antonio.my.ai.girlfriend.free.amelia.ui.ecology

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import org.json.JSONObject
import kotlin.math.*

@Composable
fun EcologyCanvas(visualPayload: JSONObject) {
    val coherence = visualPayload.optDouble("coherence", 0.5)
    val zones = visualPayload.optJSONObject("zones") ?: JSONObject()
    val colorMap = visualPayload.optJSONObject("color_map") ?: JSONObject()

    val pulse = rememberInfiniteTransition().animateFloat(
        initialValue = 0.9f, targetValue = 1.1f,
        animationSpec = infiniteRepeatable(
            animation = tween((2000 * (1.5 - coherence)).toInt(), easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        )
    )

    Canvas(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        val center = Offset(size.width / 2, size.height / 2)
        val maxRadius = size.minDimension / 3f

        val zoneKeys = zones.keys()
        var index = 0
        while (zoneKeys.hasNext()) {
            val key = zoneKeys.next()
            val energy = zones.optDouble(key, 0.0)
            val angle = (index / zones.length().toFloat()) * 2 * Math.PI
            val offset = Offset(
                (center.x + cos(angle) * maxRadius * pulse.value).toFloat(),
                (center.y + sin(angle) * maxRadius * pulse.value).toFloat()
            )
            val colorKey = colorMap.keys().asSequence().firstOrNull() ?: "#6C7FA1"
            val color = try { Color(android.graphics.Color.parseColor(colorKey)) }
                        catch (_: Exception) { Color.Gray }
            drawCircle(
                color = color.copy(alpha = 0.5f + (energy.toFloat() * 0.5f)),
                radius = (50 * (0.5 + energy)).toFloat(),
                center = offset
            )
            index++
        }
    }
}
