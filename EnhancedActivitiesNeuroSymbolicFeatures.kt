package com.antonio.my.ai.girlfriend.free.amelia.android.activities

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.StaggeredGridLayoutManager
import com.amelia.android.R
import com.amelia.android.adapters.*
import com.amelia.android.databinding.*
import com.amelia.android.models.*
import com.amelia.android.viewmodels.*
import com.github.mikephil.charting.charts.*
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.components.Description
import kotlinx.coroutines.launch

/**
 * Enhanced Dream Analysis Results Activity
 * Displays comprehensive neuro-symbolic analysis results
 */
class EnhancedAnalysisResultActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityEnhancedAnalysisResultBinding
    private val viewModel: AnalysisResultViewModel by viewModels()
    
    private lateinit var symbolsAdapter: EnhancedSymbolAdapter
    private lateinit var patternsAdapter: SymbolicPatternsAdapter
    private lateinit var connectionsAdapter: SymbolicConnectionsAdapter
    private lateinit var transformationAdapter: TransformationScenariosAdapter
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityEnhancedAnalysisResultBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setupToolbar()
        setupAdapters()
        setupObservers()
        loadAnalysisResult()
    }
    
    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.apply {
            setDisplayHomeAsUpEnabled(true)
            title = "Enhanced Analysis Results"
        }
    }
    
    private fun setupAdapters() {
        // Enhanced symbols adapter
        symbolsAdapter = EnhancedSymbolAdapter { symbol ->
            openSymbolDetail(symbol)
        }
        
        binding.rvSymbols.apply {
            layoutManager = StaggeredGridLayoutManager(2, StaggeredGridLayoutManager.VERTICAL)
            adapter = symbolsAdapter
        }
        
        // Patterns adapter
        patternsAdapter = SymbolicPatternsAdapter { pattern ->
            openPatternDetail(pattern)
        }
        
        binding.rvPatterns.apply {
            layoutManager = LinearLayoutManager(this@EnhancedAnalysisResultActivity)
            adapter = patternsAdapter
        }
        
        // Connections adapter
        connectionsAdapter = SymbolicConnectionsAdapter { connection ->
            visualizeConnection(connection)
        }
        
        binding.rvConnections.apply {
            layoutManager = LinearLayoutManager(this@EnhancedAnalysisResultActivity)
            adapter = connectionsAdapter
        }
        
        // Transformation scenarios adapter
        transformationAdapter = TransformationScenariosAdapter { scenario ->
            openTransformationDetail(scenario)
        }
        
        binding.rvTransformations.apply {
            layoutManager = LinearLayoutManager(this@EnhancedAnalysisResultActivity)
            adapter = transformationAdapter
        }
    }
    
    private fun setupObservers() {
        viewModel.analysisResult.observe(this) { result ->
            displayAnalysisResult(result)
        }
        
        viewModel.fieldDynamics.observe(this) { dynamics ->
            displayFieldDynamics(dynamics)
        }
        
        viewModel.isLoading.observe(this) { isLoading ->
            binding.progressBar.visibility = if (isLoading) 
                android.view.View.VISIBLE else android.view.View.GONE
        }
        
        viewModel.error.observe(this) { error ->
            error?.let {
                Toast.makeText(this, it, Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun loadAnalysisResult() {
        val analysisId = intent.getStringExtra("analysis_id")
        if (analysisId != null) {
            viewModel.loadAnalysisResult(analysisId)
        } else {
            Toast.makeText(this, "No analysis ID provided", Toast.LENGTH_SHORT).show()
            finish()
        }
    }
    
    private fun displayAnalysisResult(result: DreamAnalysisResult) {
        // Update symbols
        symbolsAdapter.updateSymbols(result.symbols)
        
        // Update patterns
        patternsAdapter.updatePatterns(result.patterns)
        
        // Update connections
        connectionsAdapter.updateConnections(result.connections)
        
        // Update transformation scenarios
        transformationAdapter.updateScenarios(result.transformationScenarios)
        
        // Update field coherence chart
        setupFieldCoherenceChart(result.fieldCoherence)
        
        // Update neural insights
        displayNeuralInsights(result.neuralInsights)
        
        // Update complexity metrics
        binding.tvComplexityScore.text = "${(result.complexityMeasure * 100).toInt()}%"
        binding.progressComplexity.progress = (result.complexityMeasure * 100).toInt()
        
        // Update analysis quality
        binding.tvAnalysisQuality.text = result.getAnalysisQuality().name
        binding.chipAnalysisQuality.text = result.getAnalysisQuality().name
    }
    
    private fun displayFieldDynamics(dynamics: DreamFieldDynamics) {
        // Field intensity radar chart
        setupFieldIntensityChart(dynamics.fieldIntensity)
        
        // Transformation field visualization
        setupTransformationFieldChart(dynamics.transformationField)
        
        // Temporal distortions
        if (dynamics.temporalDistortions.isNotEmpty()) {
            binding.chipTemporalDistortions.visibility = android.view.View.VISIBLE
            binding.chipTemporalDistortions.text = "${dynamics.temporalDistortions.size} distortions"
        }
    }
    
    private fun setupFieldCoherenceChart(coherence: FieldCoherenceMetrics) {
        val chart = binding.chartFieldCoherence
        
        val entries = listOf(
            BarEntry(0f, coherence.overallCoherence),
            BarEntry(1f, coherence.symbolCoherence),
            BarEntry(2f, coherence.patternCoherence),
            BarEntry(3f, coherence.networkCoherence),
            BarEntry(4f, coherence.archetypalCoherence)
        )
        
        val dataSet = BarDataSet(entries, "Field Coherence")
        dataSet.colors = listOf(
            getColor(R.color.coherence_overall),
            getColor(R.color.coherence_symbol),
            getColor(R.color.coherence_pattern),
            getColor(R.color.coherence_network),
            getColor(R.color.coherence_archetypal)
        )
        
        val data = BarData(dataSet)
        chart.data = data
        
        chart.xAxis.apply {
            valueFormatter = FieldCoherenceAxisValueFormatter()
            position = com.github.mikephil.charting.components.XAxis.XAxisPosition.BOTTOM
            setDrawGridLines(false)
        }
        
        chart.axisLeft.apply {
            axisMinimum = 0f
            axisMaximum = 1f
        }
        
        chart.axisRight.isEnabled = false
        chart.legend.isEnabled = false
        
        val description = Description()
        description.text = "Field Coherence Metrics"
        chart.description = description
        
        chart.animateY(1000)
        chart.invalidate()
    }
    
    private fun setupFieldIntensityChart(intensity: FieldIntensity) {
        val chart = binding.chartFieldIntensity
        
        val entries = listOf(
            RadarEntry(intensity.coherence),
            RadarEntry(intensity.entropy),
            RadarEntry(intensity.luminosity),
            RadarEntry(intensity.temporalFlux),
            RadarEntry(intensity.dimensionalDepth),
            RadarEntry(intensity.archetypalResonance)
        )
        
        val dataSet = RadarDataSet(entries, "Field Intensity")
        dataSet.color = getColor(R.color.field_intensity_primary)
        dataSet.fillColor = getColor(R.color.field_intensity_fill)
        dataSet.setDrawFilled(true)
        dataSet.fillAlpha = 80
        dataSet.lineWidth = 2f
        dataSet.isDrawHighlightCircleEnabled = true
        
        val data = RadarData(dataSet)
        chart.data = data
        
        chart.xAxis.valueFormatter = FieldIntensityAxisValueFormatter()
        chart.yAxis.apply {
            axisMinimum = 0f
            axisMaximum = 1f
            setLabelCount(5, false)
        }
        
        chart.animateXY(1000, 1000)
        chart.invalidate()
    }
    
    private fun setupTransformationFieldChart(transformationField: TransformationField) {
        val chart = binding.chartTransformationField
        
        val entries = transformationField.activeVectors.entries.mapIndexed { index, (vector, strength) ->
            BarEntry(index.toFloat(), strength)
        }
        
        val dataSet = BarDataSet(entries, "Transformation Vectors")
        dataSet.colors = transformationField.activeVectors.keys.map { vector ->
            getTransformationVectorColor(vector)
        }
        
        val data = BarData(dataSet)
        chart.data = data
        
        chart.xAxis.apply {
            valueFormatter = TransformationVectorAxisValueFormatter(transformationField.activeVectors.keys.toList())
            position = com.github.mikephil.charting.components.XAxis.XAxisPosition.BOTTOM
            labelRotationAngle = 45f
        }
        
        chart.animateY(1000)
        chart.invalidate()
    }
    
    private fun displayNeuralInsights(insights: NeuralSymbolicInsights) {
        // Dominant archetypes
        binding.chipGroupArchetypes.removeAllViews()
        insights.dominantArchetypes.forEach { archetype ->
            val chip = com.google.android.material.chip.Chip(this)
            chip.text = archetype.replaceFirstChar { it.uppercase() }
            chip.setChipBackgroundColorResource(R.color.chip_archetype_background)
            binding.chipGroupArchetypes.addView(chip)
        }
        
        // Symbolic density
        binding.tvSymbolicDensity.text = insights.symbolicDensity.toString()
        
        // Pattern emergence
        binding.tvPatternEmergence.text = insights.patternEmergence.toString()
        
        // Personal resonance
        binding.progressPersonalResonance.progress = (insights.personalResonanceStrength * 100).toInt()
        binding.tvPersonalResonance.text = "${(insights.personalResonanceStrength * 100).toInt()}%"
        
        // Transformation readiness
        setupTransformationReadinessChart(insights.transformationReadiness)
    }
    
    private fun setupTransformationReadinessChart(readiness: Map<DeterritorializedVector, Float>) {
        val chart = binding.chartTransformationReadiness
        
        val entries = readiness.entries.mapIndexed { index, (_, value) ->
            PieEntry(value, "")
        }
        
        val dataSet = PieDataSet(entries, "Transformation Readiness")
        dataSet.colors = readiness.keys.map { getTransformationVectorColor(it) }
        dataSet.sliceSpace = 3f
        dataSet.selectionShift = 5f
        
        val data = PieData(dataSet)
        data.setValueTextSize(12f)
        data.setValueTextColor(getColor(R.color.chart_text))
        
        chart.data = data
        chart.description.isEnabled = false
        chart.legend.isEnabled = false
        chart.setUsePercentValues(true)
        chart.animateY(1000)
        chart.invalidate()
    }
    
    private fun openSymbolDetail(symbol: SymbolMapping) {
        val intent = Intent(this, SymbolDetailActivity::class.java)
        intent.putExtra("symbol_mapping", symbol)
        startActivity(intent)
    }
    
    private fun openPatternDetail(pattern: SymbolicPattern) {
        val intent = Intent(this, PatternDetailActivity::class.java)
        intent.putExtra("pattern", pattern)
        startActivity(intent)
    }
    
    private fun visualizeConnection(connection: SymbolicConnection) {
        val intent = Intent(this, ConnectionVisualizationActivity::class.java)
        intent.putExtra("connection", connection)
        startActivity(intent)
    }
    
    private fun openTransformationDetail(scenario: TransformationScenario) {
        val intent = Intent(this, TransformationScenarioActivity::class.java)
        intent.putExtra("scenario", scenario)
        startActivity(intent)
    }
    
    private fun getTransformationVectorColor(vector: DeterritorializedVector): Int {
        return when (vector) {
            DeterritorializedVector.BECOMING_ANIMAL -> getColor(R.color.transformation_animal)
            DeterritorializedVector.BECOMING_MINERAL -> getColor(R.color.transformation_mineral)
            DeterritorializedVector.BECOMING_PLANT -> getColor(R.color.transformation_plant)
            DeterritorializedVector.BECOMING_MACHINE -> getColor(R.color.transformation_machine)
            DeterritorializedVector.BECOMING_COSMIC -> getColor(R.color.transformation_cosmic)
            DeterritorializedVector.MULTIPLICITY -> getColor(R.color.transformation_multiplicity)
            DeterritorializedVector.NOMADISM -> getColor(R.color.transformation_nomadism)
            else -> getColor(R.color.transformation_default)
        }
    }
    
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_enhanced_analysis_result, menu)
        return true
    }
    
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                onBackPressed()
                true
            }
            R.id.action_generate_narrative -> {
                generateNarrative()
                true
            }
            R.id.action_compare_analysis -> {
                compareAnalysis()
                true
            }
            R.id.action_export_analysis -> {
                exportAnalysis()
                true
            }
            R.id.action_field_dynamics -> {
                openFieldDynamics()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
    
    private fun generateNarrative() {
        val intent = Intent(this, NarrativeGenerationActivity::class.java)
        intent.putExtra("analysis_id", viewModel.analysisResult.value?.id)
        startActivity(intent)
    }
    
    private fun compareAnalysis() {
        val intent = Intent(this, AnalysisComparisonActivity::class.java)
        intent.putExtra("analysis_id", viewModel.analysisResult.value?.id)
        startActivity(intent)
    }
    
    private fun exportAnalysis() {
        viewModel.analysisResult.value?.let { result ->
            lifecycleScope.launch {
                viewModel.exportAnalysis(result, ExportFormat.JSON)
            }
        }
    }
    
    private fun openFieldDynamics() {
        val intent = Intent(this, FieldDynamicsActivity::class.java)
        intent.putExtra("analysis_id", viewModel.analysisResult.value?.id)
        startActivity(intent)
    }
}

/**
 * Pattern Detail Activity - shows detailed pattern analysis
 */
class PatternDetailActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityPatternDetailBinding
    private val viewModel: PatternDetailViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPatternDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        val pattern = intent.getParcelableExtra<SymbolicPattern>("pattern")
        if (pattern != null) {
            displayPatternDetails(pattern)
            setupPatternVisualization(pattern)
        } else {
            finish()
        }
    }
    
    private fun displayPatternDetails(pattern: SymbolicPattern) {
        binding.tvPatternId.text = pattern.id
        binding.tvPatternType.text = pattern.patternType.name.replace("_", " ")
        binding.tvCoherenceScore.text = "${(pattern.coherenceScore * 100).toInt()}%"
        binding.tvEmergenceProbability.text = "${(pattern.emergenceProbability * 100).toInt()}%"
        binding.tvComplexityMeasure.text = "${(pattern.complexityMeasure * 100).toInt()}%"
        
        // Display pattern elements
        binding.chipGroupElements.removeAllViews()
        pattern.elements.forEach { element ->
            val chip = com.google.android.material.chip.Chip(this)
            chip.text = element
            chip.setChipBackgroundColorResource(R.color.chip_pattern_element)
            binding.chipGroupElements.addView(chip)
        }
        
        // Display archetypal basis
        binding.chipGroupArchetypes.removeAllViews()
        pattern.archetypalBasis.forEach { archetype ->
            val chip = com.google.android.material.chip.Chip(this)
            chip.text = archetype
            chip.setChipBackgroundColorResource(R.color.chip_archetype_background)
            binding.chipGroupArchetypes.addView(chip)
        }
    }
    
    private fun setupPatternVisualization(pattern: SymbolicPattern) {
        if (pattern.temporalSignature.isNotEmpty()) {
            val chart = binding.chartTemporalSignature
            
            val entries = pattern.temporalSignature.mapIndexed { index, value ->
                Entry(index.toFloat(), value)
            }
            
            val dataSet = LineDataSet(entries, "Temporal Signature")
            dataSet.color = getColor(R.color.pattern_temporal_signature)
            dataSet.setCircleColor(getColor(R.color.pattern_temporal_signature))
            dataSet.lineWidth = 3f
            dataSet.circleRadius = 4f
            dataSet.setDrawFilled(true)
            dataSet.fillColor = getColor(R.color.pattern_temporal_signature_fill)
            
            val data = LineData(dataSet)
            chart.data = data
            chart.animateX(1000)
            chart.invalidate()
        }
    }
}

/**
 * Narrative Generation Activity
 */
class NarrativeGenerationActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityNarrativeGenerationBinding
    private val viewModel: NarrativeGenerationViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityNarrativeGenerationBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setupSpinners()
        setupObservers()
        
        val analysisId = intent.getStringExtra("analysis_id")
        if (analysisId != null) {
            viewModel.loadAnalysisForNarrative(analysisId)
        }
    }
    
    private fun setupSpinners() {
        // Narrative style spinner
        val styleAdapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            NarrativeStyle.values().map { it.name.lowercase().replaceFirstChar { c -> c.uppercase() } }
        )
        styleAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerNarrativeStyle.adapter = styleAdapter
        
        // Narrative length spinner
        val lengthAdapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            NarrativeLength.values().map { it.name.lowercase().replaceFirstChar { c -> c.uppercase() } }
        )
        lengthAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerNarrativeLength.adapter = lengthAdapter
    }
    
    private fun setupObservers() {
        viewModel.narrative.observe(this) { narrative ->
            displayNarrative(narrative)
        }
        
        viewModel.isGenerating.observe(this) { isGenerating ->
            binding.btnGenerate.isEnabled = !isGenerating
            binding.progressBar.visibility = if (isGenerating) 
                android.view.View.VISIBLE else android.view.View.GONE
        }
        
        binding.btnGenerate.setOnClickListener {
            generateNarrative()
        }
    }
    
    private fun generateNarrative() {
        val style = NarrativeStyle.values()[binding.spinnerNarrativeStyle.selectedItemPosition]
        val length = NarrativeLength.values()[binding.spinnerNarrativeLength.selectedItemPosition]
        val intensity = binding.sliderTransformationIntensity.value
        
        viewModel.generateNarrative(style, length, intensity)
    }
    
    private fun displayNarrative(narrative: DreamNarrative) {
        binding.tvNarrativeText.text = narrative.text
        binding.tvStyle.text = narrative.style.name
        binding.tvComplexity.text = narrative.getComplexityLevel()
        
        // Display archetypal themes
        binding.chipGroupThemes.removeAllViews()
        narrative.archetypalThemes.forEach { theme ->
            val chip = com.google.android.material.chip.Chip(this)
            chip.text = theme
            binding.chipGroupThemes.addView(chip)
        }
        
        // Display field intensity
        setupFieldIntensityVisualization(narrative.fieldIntensity)
    }
    
    private fun setupFieldIntensityVisualization(intensity: FieldIntensity) {
        binding.progressCoherence.progress = (intensity.coherence * 100).toInt()
        binding.progressEntropy.progress = (intensity.entropy * 100).toInt()
        binding.progressLuminosity.progress = (intensity.luminosity * 100).toInt()
        binding.progressTemporalFlux.progress = (intensity.temporalFlux * 100).toInt()
        binding.progressDimensionalDepth.progress = (intensity.dimensionalDepth * 100).toInt()
        binding.progressArchetypalResonance.progress = (intensity.archetypalResonance * 100).toInt()
    }
}

/**
 * Field Dynamics Visualization Activity
 */
class FieldDynamicsActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityFieldDynamicsBinding
    private val viewModel: FieldDynamicsViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFieldDynamicsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        val analysisId = intent.getStringExtra("analysis_id")
        if (analysisId != null) {
            viewModel.loadFieldDynamics(analysisId)
        }
        
        setupObservers()
    }
    
    private fun setupObservers() {
        viewModel.fieldDynamics.observe(this) { dynamics ->
            displayFieldDynamics(dynamics)
        }
    }
    
    private fun displayFieldDynamics(dynamics: DreamFieldDynamics) {
        // Field intensity visualization
        setupFieldIntensityRadar(dynamics.fieldIntensity)
        
        // Transformation field heatmap
        setupTransformationFieldHeatmap(dynamics.transformationField)
        
        // Consciousness mapping
        setupConsciousnessMapping(dynamics.consciousnessMapping)
        
        // Temporal distortions
        setupTemporalDistortions(dynamics.temporalDistortions)
        
        // Dimensional resonance
        setupDimensionalResonance(dynamics.dimensionalResonance)
    }
    
    private fun setupFieldIntensityRadar(intensity: FieldIntensity) {
        val chart = binding.chartFieldIntensityRadar
        
        val entries = listOf(
            RadarEntry(intensity.coherence),
            RadarEntry(intensity.entropy),
            RadarEntry(intensity.luminosity),
            RadarEntry(intensity.temporalFlux),
            RadarEntry(intensity.dimensionalDepth),
            RadarEntry(intensity.archetypalResonance)
        )
        
        val dataSet = RadarDataSet(entries, "Field Intensity")
        dataSet.color = getColor(R.color.field_intensity_primary)
        dataSet.fillColor = getColor(R.color.field_intensity_fill)
        dataSet.setDrawFilled(true)
        dataSet.fillAlpha = 120
        dataSet.lineWidth = 3f
        
        val data = RadarData(dataSet)
        chart.data = data
        chart.animateXY(1500, 1500)
        chart.invalidate()
    }
    
    private fun setupTransformationFieldHeatmap(field: TransformationField) {
        // Create a custom view for transformation field visualization
        // This would be a complex 3D or heatmap visualization
        binding.tvTransformationFieldStrength.text = "${(field.transformationPotential * 100).toInt()}%"
        binding.tvFieldCoherence.text = "${(field.fieldCoherence * 100).toInt()}%"
        binding.tvStabilityIndex.text = "${(field.stabilityIndex * 100).toInt()}%"
    }
    
    private fun setupConsciousnessMapping(mapping: ConsciousnessMapping) {
        binding.tvCognitiveComplexity.text = "${(mapping.cognitiveComplexity * 100).toInt()}%"
        binding.tvIntrospectionDepth.text = "${(mapping.introspectionDepth * 100).toInt()}%"
        
        // Consciousness levels pie chart
        val chart = binding.chartConsciousnessLevels
        
        val entries = mapping.consciousnessLevels.map { (level, strength) ->
            PieEntry(strength, level.replaceFirstChar { it.uppercase() })
        }
        
        val dataSet = PieDataSet(entries, "Consciousness Levels")
        dataSet.colors = listOf(
            getColor(R.color.consciousness_surface),
            getColor(R.color.consciousness_personal),
            getColor(R.color.consciousness_collective)
        )
        
        val data = PieData(dataSet)
        chart.data = data
        chart.animateY(1000)
        chart.invalidate()
    }
    
    private fun setupTemporalDistortions(distortions: List<TemporalDistortion>) {
        binding.recyclerTemporalDistortions.apply {
            layoutManager = LinearLayoutManager(this@FieldDynamicsActivity)
            adapter = TemporalDistortionsAdapter(distortions)
        }
    }
    
    private fun setupDimensionalResonance(resonance: DimensionalResonance) {
        binding.tvPrimaryDimension.text = resonance.primaryDimension
        binding.tvResonanceFrequency.text = "${(resonance.resonanceFrequency * 100).toInt()}%"
        binding.tvStabilityIndex.text = "${(resonance.stabilityIndex * 100).toInt()}%"
        binding.tvPhaseCoherence.text = "${(resonance.phaseCoherence * 100).toInt()}%"
        binding.tvDimensionalDepth.text = resonance.dimensionalDepth.toString()
        
        // Harmonics visualization
        if (resonance.harmonics.isNotEmpty()) {
            setupHarmonicsChart(resonance.harmonics)
        }
    }
    
    private fun setupHarmonicsChart(harmonics: List<Float>) {
        val chart = binding.chartHarmonics
        
        val entries = harmonics.mapIndexed { index, value ->
            BarEntry(index.toFloat(), value)
        }
        
        val dataSet = BarDataSet(entries, "Harmonic Frequencies")
        dataSet.color = getColor(R.color.harmonics_primary)
        
        val data = BarData(dataSet)
        chart.data = data
        chart.animateY(1000)
        chart.invalidate()
    }
}

// Custom axis value formatters for charts

class FieldCoherenceAxisValueFormatter : com.github.mikephil.charting.formatter.ValueFormatter() {
    private val labels = arrayOf("Overall", "Symbol", "Pattern", "Network", "Archetypal")
    
    override fun getAxisLabel(value: Float, axis: com.github.mikephil.charting.components.AxisBase?): String {
        return labels.getOrNull(value.toInt()) ?: ""
    }
}

class FieldIntensityAxisValueFormatter : com.github.mikephil.charting.formatter.ValueFormatter() {
    private val labels = arrayOf("Coherence", "Entropy", "Luminosity", "Temporal", "Dimensional", "Archetypal")
    
    override fun getAxisLabel(value: Float, axis: com.github.mikephil.charting.components.AxisBase?): String {
        return labels.getOrNull(value.toInt()) ?: ""
    }
}

class TransformationVectorAxisValueFormatter(
    private val vectors: List<DeterritorializedVector>
) : com.github.mikephil.charting.formatter.ValueFormatter() {
    
    override fun getAxisLabel(value: Float, axis: com.github.mikephil.charting.components.AxisBase?): String {
        return vectors.getOrNull(value.toInt())?.name?.replace("_", " ") ?: ""
    }
}
