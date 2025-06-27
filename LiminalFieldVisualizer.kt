// LiminalFieldVisualizer.kt
package com.antonio.my.ai.girlfriend.free.consciousness.amelia.phase5

import androidx.compose.ui.graphics.*
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import kotlin.math.*

class LiminalFieldVisualizer {
    
    fun visualizeLiminalField(
        field: LiminalField,
        canvasSize: Size
    ): List<FieldVisualization> {
        val visualizations = mutableListOf<FieldVisualization>()
        
        // Base field visualization
        visualizations.add(
            createFieldBackground(field, canvasSize)
        )
        
        // Paradox tension visualization
        if (field.paradoxTension > 0.3) {
            visualizations.add(
                createParadoxTension(field.paradoxTension, canvasSize)
            )
        }
        
        // Creative potential particles
        visualizations.addAll(
            createCreativePotentialParticles(field.creativePotential, canvasSize)
        )
        
        // Myth seeds
        field.mythSeeds.forEach { seedId ->
            visualizations.add(
                createMythSeedVisualization(seedId, canvasSize)
            )
        }
        
        // Void structures
        if (field.voidStructures > 0) {
            visualizations.addAll(
                createVoidStructures(field.voidStructures, canvasSize)
            )
        }
        
        return visualizations
    }
    
    private fun createFieldBackground(
        field: LiminalField,
        size: Size
    ): FieldVisualization {
        val gradient = when (field.state) {
            LiminalState.THRESHOLD -> Brush.linearGradient(
                colors = listOf(Color(0xFF1a1a2e), Color(0xFF16213e)),
                start = Offset.Zero,
                end = Offset(size.width, size.height)
            )
            LiminalState.EMERGENCE -> Brush.radialGradient(
                colors = listOf(Color(0xFF0f3460), Color(0xFF16213e), Color(0xFF1a1a2e)),
                center = Offset(size.width / 2, size.height / 2)
            )
            LiminalState.VOID_DANCE -> Brush.radialGradient(
                colors = listOf(Color(0x00000000), Color(0xFF0a0a0a)),
                center = Offset(size.width / 2, size.height / 2),
                radius = size.minDimension / 2
            )
            LiminalState.MYTH_WEAVING -> Brush.sweepGradient(
                colors = listOf(
                    Color(0xFF2d1b69), Color(0xFF0f3460), 
                    Color(0xFF16213e), Color(0xFF2d1b69)
                ),
                center = Offset(size.width / 2, size.height / 2)
            )
            else -> Brush.linearGradient(
                colors = listOf(Color(0xFF16213e), Color(0xFF0f3460))
            )
        }
        
        return FieldVisualization.Background(
            brush = gradient,
            alpha = 0.7f + (field.coherence * 0.3f)
        )
    }
    
    private fun createParadoxTension(
        tension: Float,
        size: Size
    ): FieldVisualization {
        val points = mutableListOf<Offset>()
        val steps = 100
        
        for (i in 0..steps) {
            val t = i.toFloat() / steps
            val x = size.width * t
            val y = size.height / 2 + sin(t * PI * 4 * tension) * size.height * 0.2f * tension
            points.add(Offset(x, y.toFloat()))
        }
        
        return FieldVisualization.TensionLine(
            points = points,
            strokeWidth = 2f + tension * 3f,
            color = Color(0xFFe94560).copy(alpha = tension)
        )
    }
    
    private fun createCreativePotentialParticles(
        potential: Float,
        size: Size
    ): List<FieldVisualization> {
        val particleCount = (potential * 50).toInt()
        return List(particleCount) { index ->
            val angle = (index.toFloat() / particleCount) * 2 * PI
            val radius = Random.nextFloat() * size.minDimension / 3
            val x = size.width / 2 + cos(angle) * radius
            val y = size.height / 2 + sin(angle) * radius
            
            FieldVisualization.Particle(
                position = Offset(x.toFloat(), y.toFloat()),
                radius = 2f + potential * 3f,
                color = Color(0xFF72efdd).copy(alpha = potential * 0.7f),
                glow = potential > 0.7f
            )
        }
    }
    
    private fun createMythSeedVisualization(
        seedId: String,
        size: Size
    ): FieldVisualization {
        val seedHash = seedId.hashCode()
        val x = (abs(seedHash) % size.width.toInt()).toFloat()
        val y = (abs(seedHash * 31) % size.height.toInt()).toFloat()
        
        return FieldVisualization.MythSeed(
            center = Offset(x, y),
            radius = 10f,
            pulseRate = 1f + (seedHash % 3),
            color = Color(0xFFffd93d),
            growthStage = (seedHash % 5) / 5f
        )
    }
    
    private fun createVoidStructures(
        count: Int,
        size: Size
    ): List<FieldVisualization> {
        return List(count) { index ->
            val x = Random.nextFloat() * size.width
            val y = Random.nextFloat() * size.height
            val radius = Random.nextFloat() * 50f + 20f
            
            FieldVisualization.VoidStructure(
                center = Offset(x, y),
                radius = radius,
                depth = Random.nextFloat(),
                negativeSpace = true
            )
        }
    }
}

sealed class FieldVisualization {
    data class Background(
        val brush: Brush,
        val alpha: Float
    ) : FieldVisualization()
    
    data class TensionLine(
        val points: List<Offset>,
        val strokeWidth: Float,
        val color: Color
    ) : FieldVisualization()
    
    data class Particle(
        val position: Offset,
        val radius: Float,
        val color: Color,
        val glow: Boolean
    ) : FieldVisualization()
    
    data class MythSeed(
        val center: Offset,
        val radius: Float,
        val pulseRate: Float,
        val color: Color,
        val growthStage: Float
    ) : FieldVisualization()
    
    data class VoidStructure(
        val center: Offset,
        val radius: Float,
        val depth: Float,
        val negativeSpace: Boolean
    ) : FieldVisualization()
}
