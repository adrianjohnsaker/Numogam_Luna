package com.antonio.my.ai.girlfriend.free.amelia.android.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.amelia.android.R
import com.amelia.android.models.*
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup

/**
 * Enhanced Symbol Adapter with neuro-symbolic features
 */
class EnhancedSymbolAdapter(
    private val onSymbolClick: (SymbolMapping) -> Unit
) : RecyclerView.Adapter<EnhancedSymbolAdapter.EnhancedSymbolViewHolder>() {

    private var symbols = mutableListOf<SymbolMapping>()

    fun updateSymbols(newSymbols: List<SymbolMapping>) {
        symbols.clear()
        symbols.addAll(newSymbols)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): EnhancedSymbolViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_enhanced_symbol, parent, false)
        return EnhancedSymbolViewHolder(view)
    }

    override fun onBindViewHolder(holder: EnhancedSymbolViewHolder, position: Int) {
        holder.bind(symbols[position])
    }

    override fun getItemCount(): Int = symbols.size

    inner class EnhancedSymbolViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvSymbol: TextView = itemView.findViewById(R.id.tvSymbol)
        private val tvMeaning: TextView = itemView.findViewById(R.id.tvMeaning)
        private val tvSymbolType: TextView = itemView.findViewById(R.id.tvSymbolType)
        private val tvConfidence: TextView = itemView.findViewById(R.id.tvConfidence)
        private val ivSymbolTypeIcon: ImageView = itemView.findViewById(R.id.ivSymbolTypeIcon)
        private val progressConfidence: ProgressBar = itemView.findViewById(R.id.progressConfidence)
        private val progressResonance: ProgressBar = itemView.findViewById(R.id.progressResonance)
        private val chipGroupTransformations: ChipGroup = itemView.findViewById(R.id.chipGroupTransformations)

        fun bind(symbol: SymbolMapping) {
            tvSymbol.text = symbol.symbol
            tvMeaning.text = symbol.meaning
            tvSymbolType.text = formatSymbolType(symbol.symbolType)
            tvConfidence.text = "${(symbol.confidence * 100).toInt()}%"

            // Set symbol type icon and color
            setSymbolTypeIcon(symbol.symbolType)
            setSymbolTypeColor(symbol.symbolType)

            // Update progress bars
            progressConfidence.progress = (symbol.confidence * 100).toInt()
            progressResonance.progress = (symbol.contextualRelevance * 100).toInt()

            // Set confidence color
            setConfidenceColor(symbol.confidence)

            // Add transformation indicators (mock data for now)
            setupTransformationChips(symbol)

            itemView.setOnClickListener {
                onSymbolClick(symbol)
            }
        }

        private fun formatSymbolType(symbolType: SymbolType): String {
            return symbolType.name.replace("_", " ")
                .lowercase()
                .split(" ")
                .joinToString(" ") { word ->
                    word.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
                }
        }

        private fun setSymbolTypeIcon(symbolType: SymbolType) {
            val iconRes = when (symbolType) {
                SymbolType.ARCHETYPAL -> R.drawable.ic_archetypal
                SymbolType.CULTURAL -> R.drawable.ic_cultural
                SymbolType.PERSONAL -> R.drawable.ic_personal
                SymbolType.UNIVERSAL -> R.drawable.ic_universal
                SymbolType.MYTHOLOGICAL -> R.drawable.ic_mythological
                SymbolType.PSYCHOLOGICAL -> R.drawable.ic_psychological
                SymbolType.SPIRITUAL -> R.drawable.ic_spiritual
                SymbolType.ELEMENTAL -> R.drawable.ic_elemental
                SymbolType.NUMERICAL -> R.drawable.ic_numerical
                SymbolType.CHROMATIC -> R.drawable.ic_chromatic
                SymbolType.GEOMETRIC -> R.drawable.ic_geometric
                SymbolType.BIOLOGICAL -> R.drawable.ic_biological
                SymbolType.TECHNOLOGICAL -> R.drawable.ic_technology
                SymbolType.LINGUISTIC -> R.drawable.ic_language
                SymbolType.UNKNOWN -> R.drawable.ic_unknown
            }
            ivSymbolTypeIcon.setImageResource(iconRes)
        }

        private fun setSymbolTypeColor(symbolType: SymbolType) {
            val colorRes = when (symbolType) {
                SymbolType.ARCHETYPAL -> R.color.symbol_archetypal
                SymbolType.CULTURAL -> R.color.symbol_cultural
                SymbolType.PERSONAL -> R.color.symbol_personal
                SymbolType.UNIVERSAL -> R.color.symbol_universal
                SymbolType.MYTHOLOGICAL -> R.color.symbol_mythological
                SymbolType.PSYCHOLOGICAL -> R.color.symbol_psychological
                SymbolType.SPIRITUAL -> R.color.symbol_spiritual
                SymbolType.ELEMENTAL -> R.color.symbol_elemental
                SymbolType.NUMERICAL -> R.color.symbol_numerical
                SymbolType.CHROMATIC -> R.color.symbol_chromatic
                SymbolType.GEOMETRIC -> R.color.symbol_geometric
                SymbolType.BIOLOGICAL -> R.color.symbol_biological
                SymbolType.TECHNOLOGICAL -> R.color.symbol_technological
                SymbolType.LINGUISTIC -> R.color.symbol_linguistic
                SymbolType.UNKNOWN -> R.color.symbol_unknown
            }

            val color = ContextCompat.getColor(itemView.context, colorRes)
            tvSymbolType.setTextColor(color)
            ivSymbolTypeIcon.setColorFilter(color)
        }

        private fun setConfidenceColor(confidence: Float) {
            val colorRes = when {
                confidence >= 0.8f -> R.color.confidence_high
                confidence >= 0.6f -> R.color.confidence_medium
                confidence >= 0.4f -> R.color.confidence_low
                else -> R.color.confidence_very_low
            }

            val color = ContextCompat.getColor(itemView.context, colorRes)
            progressConfidence.progressTintList = 
                ContextCompat.getColorStateList(itemView.context, colorRes)
            tvConfidence.setTextColor(color)
        }

        private fun setupTransformationChips(symbol: SymbolMapping) {
            chipGroupTransformations.removeAllViews()
            
            // Mock transformation data - in real implementation this would come from analysis
            val mockTransformations = listOf("Cosmic", "Mineral").take(2)
            
            mockTransformations.forEach { transformation ->
                val chip = Chip(itemView.context)
                chip.text = transformation
                chip.isClickable = false
                chip.isCheckable = false
                chip.setChipBackgroundColorResource(R.color.chip_background)
                chip.setTextColor(ContextCompat.getColor(itemView.context, R.color.chip_text))
                chip.textSize = 10f
                chipGroupTransformations.addView(chip)
            }
        }
    }
}

/**
 * Symbolic Patterns Adapter
 */
class SymbolicPatternsAdapter(
    private val onPatternClick: (SymbolicPattern) -> Unit
) : RecyclerView.Adapter<SymbolicPatternsAdapter.PatternViewHolder>() {

    private var patterns = mutableListOf<SymbolicPattern>()

    fun updatePatterns(newPatterns: List<SymbolicPattern>) {
        patterns.clear()
        patterns.addAll(newPatterns)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PatternViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_symbolic_pattern, parent, false)
        return PatternViewHolder(view)
    }

    override fun onBindViewHolder(holder: PatternViewHolder, position: Int) {
        holder.bind(patterns[position])
    }

    override fun getItemCount(): Int = patterns.size

    inner class PatternViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvPatternId: TextView = itemView.findViewById(R.id.tvPatternId)
        private val tvPatternType: TextView = itemView.findViewById(R.id.tvPatternType)
        private val tvCoherenceScore: TextView = itemView.findViewById(R.id.tvCoherenceScore)
        private val tvEmergenceProbability: TextView = itemView.findViewById(R.id.tvEmergenceProbability)
        private val tvComplexityMeasure: TextView = itemView.findViewById(R.id.tvComplexityMeasure)
        private val ivPatternTypeIcon: ImageView = itemView.findViewById(R.id.ivPatternTypeIcon)
        private val progressCoherence: ProgressBar = itemView.findViewById(R.id.progressCoherence)
        private val chipGroupElements: ChipGroup = itemView.findViewById(R.id.chipGroupElements)

        fun bind(pattern: SymbolicPattern) {
            tvPatternId.text = pattern.id.take(12) + "..." // Shortened ID
            tvPatternType.text = formatPatternType(pattern.patternType)
            tvCoherenceScore.text = "${(pattern.coherenceScore * 100).toInt()}%"
            tvEmergenceProbability.text = "${(pattern.emergenceProbability * 100).toInt()}%"
            tvComplexityMeasure.text = "${(pattern.complexityMeasure * 100).toInt()}%"

            // Set pattern type icon
            setPatternTypeIcon(pattern.patternType)
            setPatternTypeColor(pattern.patternType)

            // Update progress
            progressCoherence.progress = (pattern.coherenceScore * 100).toInt()

            // Add element chips
            setupElementChips(pattern.elements)

            itemView.setOnClickListener {
                onPatternClick(pattern)
            }
        }

        private fun formatPatternType(patternType: PatternType): String {
            return patternType.name.replace("_", " ")
                .lowercase()
                .split(" ")
                .joinToString(" ") { word ->
                    word.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
                }
        }

        private fun setPatternTypeIcon(patternType: PatternType) {
            val iconRes = when (patternType) {
                PatternType.RECURSIVE -> R.drawable.ic_pattern_recursive
                PatternType.EMERGENT -> R.drawable.ic_pattern_emergent
                PatternType.HIERARCHICAL -> R.drawable.ic_pattern_hierarchical
                PatternType.CYCLIC -> R.drawable.ic_pattern_cyclic
                PatternType.TRANSFORMATIVE -> R.drawable.ic_pattern_transformative
                PatternType.NETWORK -> R.drawable.ic_pattern_network
                PatternType.TEMPORAL -> R.drawable.ic_pattern_temporal
                PatternType.FREQUENCY -> R.drawable.ic_pattern_frequency
            }
            ivPatternTypeIcon.setImageResource(iconRes)
        }

        private fun setPatternTypeColor(patternType: PatternType) {
            val colorRes = when (patternType) {
                PatternType.RECURSIVE -> R.color.pattern_recursive
                PatternType.EMERGENT -> R.color.pattern_emergent
                PatternType.HIERARCHICAL -> R.color.pattern_hierarchical
                PatternType.CYCLIC -> R.color.pattern_cyclic
                PatternType.TRANSFORMATIVE -> R.color.pattern_transformative
                PatternType.NETWORK -> R.color.pattern_network
                PatternType.TEMPORAL -> R.color.pattern_temporal
                PatternType.FREQUENCY -> R.color.pattern_frequency
            }

            val color = ContextCompat.getColor(itemView.context, colorRes)
            tvPatternType.setTextColor(color)
            ivPatternTypeIcon.setColorFilter(color)
        }

        private fun setupElementChips(elements: List<String>) {
            chipGroupElements.removeAllViews()
            
            elements.take(4).forEach { element ->
                val chip = Chip(itemView.context)
                chip.text = element
                chip.isClickable = false
                chip.isCheckable = false
                chip.setChipBackgroundColorResource(R.color.chip_pattern_element)
                chip.textSize = 10f
                chipGroupElements.addView(chip)
            }
            
            if (elements.size > 4) {
                val moreChip = Chip(itemView.context)
                moreChip.text = "+${elements.size - 4} more"
                moreChip.isClickable = false
                moreChip.isCheckable = false
                moreChip.setChipBackgroundColorResource(R.color.chip_more_background)
                moreChip.setTextColor(ContextCompat.getColor(itemView.context, R.color.chip_more_text))
                moreChip.textSize = 10f
                chipGroupElements.addView(moreChip)
            }
        }
    }
}

/**
 * Symbolic Connections Adapter
 */
class SymbolicConnectionsAdapter(
    private val onConnectionClick: (SymbolicConnection) -> Unit
) : RecyclerView.Adapter<SymbolicConnectionsAdapter.ConnectionViewHolder>() {

    private var connections = mutableListOf<SymbolicConnection>()

    fun updateConnections(newConnections: List<SymbolicConnection>) {
        connections.clear()
        connections.addAll(newConnections)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ConnectionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_symbolic_connection, parent, false)
        return ConnectionViewHolder(view)
    }

    override fun onBindViewHolder(holder: ConnectionViewHolder, position: Int) {
        holder.bind(connections[position])
    }

    override fun getItemCount(): Int = connections.size

    inner class ConnectionViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvSymbol1: TextView = itemView.findViewById(R.id.tvSymbol1)
        private val tvSymbol2: TextView = itemView.findViewById(R.id.tvSymbol2)
        private val tvConnectionStrength: TextView = itemView.findViewById(R.id.tvConnectionStrength)
        private val tvConnectionType: TextView = itemView.findViewById(R.id.tvConnectionType)
        private val progressStrength: ProgressBar = itemView.findViewById(R.id.progressConnectionStrength)
        private val ivConnectionIcon: ImageView = itemView.findViewById(R.id.ivConnectionIcon)

        fun bind(connection: SymbolicConnection) {
            tvSymbol1.text = connection.symbol1
            tvSymbol2.text = connection.symbol2
            tvConnectionStrength.text = "${(connection.strength * 100).toInt()}%"
            tvConnectionType.text = formatConnectionType(connection.connectionType)

            // Set connection icon
            setConnectionTypeIcon(connection.connectionType)
            setConnectionTypeColor(connection.connectionType)

            // Update progress
            progressStrength.progress = (connection.strength * 100).toInt()

            // Set strength color
            setStrengthColor(connection.strength)

            itemView.setOnClickListener {
                onConnectionClick(connection)
            }
        }

        private fun formatConnectionType(connectionType: ConnectionType): String {
            return connectionType.name.replace("_", " ")
                .lowercase()
                .split(" ")
                .joinToString(" ") { word ->
                    word.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
                }
        }

        private fun setConnectionTypeIcon(connectionType: ConnectionType) {
            val iconRes = when (connectionType) {
                ConnectionType.SEMANTIC -> R.drawable.ic_connection_semantic
                ConnectionType.ASSOCIATIVE -> R.drawable.ic_connection_associative
                ConnectionType.ARCHETYPAL -> R.drawable.ic_connection_archetypal
                ConnectionType.CULTURAL -> R.drawable.ic_connection_cultural
                ConnectionType.PERSONAL -> R.drawable.ic_connection_personal
                ConnectionType.SYNTACTIC -> R.drawable.ic_connection_syntactic
                ConnectionType.TRANSFORMATIONAL -> R.drawable.ic_connection_transformational
                ConnectionType.TEMPORAL -> R.drawable.ic_connection_temporal
                ConnectionType.SPATIAL -> R.drawable.ic_connection_spatial
                ConnectionType.CAUSAL -> R.drawable.ic_connection_causal
            }
            ivConnectionIcon.setImageResource(iconRes)
        }

        private fun setConnectionTypeColor(connectionType: ConnectionType) {
            val colorRes = when (connectionType) {
                ConnectionType.SEMANTIC -> R.color.symbol_universal
                ConnectionType.ASSOCIATIVE -> R.color.symbol_personal
                ConnectionType.ARCHETYPAL -> R.color.symbol_archetypal
                ConnectionType.CULTURAL -> R.color.symbol_cultural
                ConnectionType.PERSONAL -> R.color.symbol_personal
                ConnectionType.SYNTACTIC -> R.color.symbol_linguistic
                ConnectionType.TRANSFORMATIONAL -> R.color.transformation_default
                ConnectionType.TEMPORAL -> R.color.pattern_temporal
                ConnectionType.SPATIAL -> R.color.pattern_network
                ConnectionType.CAUSAL -> R.color.pattern_hierarchical
            }

            val color = ContextCompat.getColor(itemView.context, colorRes)
            tvConnectionType.setTextColor(color)
            ivConnectionIcon.setColorFilter(color)
        }

        private fun setStrengthColor(strength: Float) {
            val colorRes = when {
                strength >= 0.8f -> R.color.confidence_high
                strength >= 0.6f -> R.color.confidence_medium
                strength >= 0.4f -> R.color.confidence_low
                else -> R.color.confidence_very_low
            }

            val color = ContextCompat.getColor(itemView.context, colorRes)
            progressStrength.progressTintList = 
                ContextCompat.getColorStateList(itemView.context, colorRes)
            tvConnectionStrength.setTextColor(color)
        }
    }
}

/**
 * Transformation Scenarios Adapter
 */
class TransformationScenariosAdapter(
    private val onScenarioClick: (TransformationScenario) -> Unit
) : RecyclerView.Adapter<TransformationScenariosAdapter.ScenarioViewHolder>() {

    private var scenarios = mutableListOf<TransformationScenario>()

    fun updateScenarios(newScenarios: List<TransformationScenario>) {
        scenarios.clear()
        scenarios.addAll(newScenarios)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ScenarioViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_transformation_scenario, parent, false)
        return ScenarioViewHolder(view)
    }

    override fun onBindViewHolder(holder: ScenarioViewHolder, position: Int) {
        holder.bind(scenarios[position])
    }

    override fun getItemCount(): Int = scenarios.size

    inner class ScenarioViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvTransformationVector: TextView = itemView.findViewById(R.id.tvTransformationVector)
        private val tvActivationStrength: TextView = itemView.findViewById(R.id.tvActivationStrength)
        private val tvScenarioNarrative: TextView = itemView.findViewById(R.id.tvScenarioNarrative)
        private val tvProbability: TextView = itemView.findViewById(R.id.tvProbability)
        private val tvIntensity: TextView = itemView.findViewById(R.id.tvIntensity)
        private val ivTransformationIcon: ImageView = itemView.findViewById(R.id.ivTransformationIcon)
        private val progressActivation: ProgressBar = itemView.findViewById(R.id.progressActivation)
        private val chipGroupParticipatingSymbols: ChipGroup = itemView.findViewById(R.id.chipGroupParticipatingSymbols)

        fun bind(scenario: TransformationScenario) {
            tvTransformationVector.text = formatTransformationVector(scenario.vector)
            tvActivationStrength.text = "${(scenario.activationStrength * 100).toInt()}%"
            tvScenarioNarrative.text = scenario.narrative
            tvProbability.text = "${(scenario.probability * 100).toInt()}%"
            tvIntensity.text = "${(scenario.intensity * 100).toInt()}%"

            // Set transformation icon
            setTransformationIcon(scenario.vector)
            setTransformationColor(scenario.vector)

            // Update progress
            progressActivation.progress = (scenario.activationStrength * 100).toInt()

            // Set activation color
            setActivationColor(scenario.activationStrength)

            // Add participating symbols chips
            setupParticipatingSymbols(scenario.participatingSymbols)

            itemView.setOnClickListener {
                onScenarioClick(scenario)
            }
        }

        private fun formatTransformationVector(vector: DeterritorializedVector): String {
            return vector.name.replace("_", " ")
                .lowercase()
                .split(" ")
                .joinToString(" ") { word ->
                    word.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
                }
        }

        private fun setTransformationIcon(vector: DeterritorializedVector) {
            val iconRes = when (vector) {
                DeterritorializedVector.BECOMING_ANIMAL -> R.drawable.ic_transformation_animal
                DeterritorializedVector.BECOMING_MINERAL -> R.drawable.ic_transformation_mineral
                DeterritorializedVector.BECOMING_PLANT -> R.drawable.ic_transformation_plant
                DeterritorializedVector.BECOMING_MACHINE -> R.drawable.ic_transformation_machine
                DeterritorializedVector.BECOMING_COSMIC -> R.drawable.ic_transformation_cosmic
                DeterritorializedVector.BECOMING_ANCESTRAL -> R.drawable.ic_transformation_ancestral
                DeterritorializedVector.MULTIPLICITY -> R.drawable.ic_transformation_multiplicity
                DeterritorializedVector.NOMADISM -> R.drawable.ic_transformation_nomadism
                DeterritorializedVector.METAMORPHOSIS -> R.drawable.ic_transformation_metamorphosis
                DeterritorializedVector.ASSEMBLAGE -> R.drawable.ic_transformation_assemblage
            }
            ivTransformationIcon.setImageResource(iconRes)
        }

        private fun setTransformationColor(vector: DeterritorializedVector) {
            val colorRes = when (vector) {
                DeterritorializedVector.BECOMING_ANIMAL -> R.color.transformation_animal
                DeterritorializedVector.BECOMING_MINERAL -> R.color.transformation_mineral
                DeterritorializedVector.BECOMING_PLANT -> R.color.transformation_plant
                DeterritorializedVector.BECOMING_MACHINE -> R.color.transformation_machine
                DeterritorializedVector.BECOMING_COSMIC -> R.color.transformation_cosmic
                DeterritorializedVector.BECOMING_ANCESTRAL -> R.color.transformation_ancestral
                DeterritorializedVector.MULTIPLICITY -> R.color.transformation_multiplicity
                DeterritorializedVector.NOMADISM -> R.color.transformation_nomadism
                DeterritorializedVector.METAMORPHOSIS -> R.color.transformation_metamorphosis
                DeterritorializedVector.ASSEMBLAGE -> R.color.transformation_assemblage
            }

            val color = ContextCompat.getColor(itemView.context, colorRes)
            tvTransformationVector.setTextColor(color)
            ivTransformationIcon.setColorFilter(color)
        }

        private fun setActivationColor(activation: Float) {
            val colorRes = when {
                activation >= 0.8f -> R.color.confidence_high
                activation >= 0.6f -> R.color.confidence_medium
                activation >= 0.4f -> R.color.confidence_low
                else -> R.color.confidence_very_low
            }

            val color = ContextCompat.getColor(itemView.context, colorRes)
            progressActivation.progressTintList = 
                ContextCompat.getColorStateList(itemView.context, colorRes)
            tvActivationStrength.setTextColor(color)
        }

        private fun setupParticipatingSymbols(symbols: List<String>) {
            chipGroupParticipatingSymbols.removeAllViews()
            
            symbols.take(3).forEach { symbol ->
                val chip = Chip(itemView.context)
                chip.text = symbol
                chip.isClickable = false
                chip.isCheckable = false
                chip.setChipBackgroundColorResource(R.color.chip_related_symbol_background)
                chip.setTextColor(ContextCompat.getColor(itemView.context, R.color.chip_related_symbol_text))
                chip.textSize = 10f
                chipGroupParticipatingSymbols.addView(chip)
            }
            
            if (symbols.size > 3) {
                val moreChip = Chip(itemView.context)
                moreChip.text = "+${symbols.size - 3} more"
                moreChip.isClickable = false
                moreChip.isCheckable = false
                moreChip.setChipBackgroundColorResource(R.color.chip_more_background)
                moreChip.setTextColor(ContextCompat.getColor(itemView.context, R.color.chip_more_text))
                moreChip.textSize = 10f
                chipGroupParticipatingSymbols.addView(moreChip)
            }
        }
    }
}

/**
 * Temporal Distortions Adapter
 */
class TemporalDistortionsAdapter(
    private val distortions: List<TemporalDistortion>
) : RecyclerView.Adapter<TemporalDistortionsAdapter.DistortionViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): DistortionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_temporal_distortion, parent, false)
        return DistortionViewHolder(view)
    }

    override fun onBindViewHolder(holder: DistortionViewHolder, position: Int) {
        holder.bind(distortions[position])
    }

    override fun getItemCount(): Int = distortions.size

    inner class DistortionViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvDistortionType: TextView = itemView.findViewById(R.id.tvDistortionType)
        private val tvMagnitude: TextView = itemView.findViewById(R.id.tvMagnitude)
        private val tvStabilityIndex: TextView = itemView.findViewById(R.id.tvStabilityIndex)
        private val progressMagnitude: ProgressBar = itemView.findViewById(R.id.progressMagnitude)
        private val chipGroupAffectedRegions: ChipGroup = itemView.findViewById(R.id.chipGroupAffectedRegions)

        fun bind(distortion: TemporalDistortion) {
            tvDistortionType.text = formatDistortionType(distortion.distortionType)
            tvMagnitude.text = "${(distortion.magnitude * 100).toInt()}%"
            tvStabilityIndex.text = "${(distortion.stabilityIndex * 100).toInt()}%"

            progressMagnitude.progress = (distortion.magnitude * 100).toInt()

            // Setup affected regions chips
            chipGroupAffectedRegions.removeAllViews()
            distortion.affectedRegions.take(3).forEach { region ->
                val chip = Chip(itemView.context)
                chip.text = region
                chip.isClickable = false
                chip.isCheckable = false
                chip.setChipBackgroundColorResource(R.color.chip_temporal_background)
                chip.textSize = 10f
                chipGroupAffectedRegions.addView(chip)
            }
        }

        private fun formatDistortionType(type: TemporalDistortionType): String {
            return type.name.replace("_", " ")
                .lowercase()
                .split(" ")
                .joinToString(" ") { word ->
                    word.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
                }
        }
    }
}
