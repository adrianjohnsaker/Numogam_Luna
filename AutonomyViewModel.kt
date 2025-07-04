package com.antonio.my.ai.girlfriend.free.amelia.autonomy.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.amelia.autonomy.bridge.AutonomousProjectEngine
import com.amelia.autonomy.model.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class AutonomyViewModel : ViewModel() {

    // Active Projects
    private val _activeProjects = MutableStateFlow<List<AutonomousProject>>(emptyList())
    val activeProjects: StateFlow<List<AutonomousProject>> = _activeProjects

    // Project Metrics Map
    private val _projectMetrics = MutableStateFlow<Map<String, ProjectMetrics>>(emptyMap())
    val projectMetrics: StateFlow<Map<String, ProjectMetrics>> = _projectMetrics

    // Attractor Landscape
    private val _attractorLandscape = MutableStateFlow<List<Attractor>>(emptyList())
    val attractorLandscape: StateFlow<List<Attractor>> = _attractorLandscape

    // Connections between attractors (graph edges)
    private val _attractorConnections = MutableStateFlow<List<Pair<String, String>>>(emptyList())
    val attractorConnections: StateFlow<List<Pair<String, String>>> = _attractorConnections

    // Seeds (emergent initiatives)
    private val _initiativeSeeds = MutableStateFlow<List<InitiativeSeed>>(emptyList())
    val initiativeSeeds: StateFlow<List<InitiativeSeed>> = _initiativeSeeds

    init {
        refreshAll()
    }

    fun refreshAll() {
        viewModelScope.launch {
            loadProjects()
            loadLandscape()
            loadSeeds()
        }
    }

    private suspend fun loadProjects() {
        val projects = AutonomousProjectEngine.getActiveProjects()
        val metricsMap = AutonomousProjectEngine.getProjectMetricsMap(projects.map { it.id })
        _activeProjects.value = projects
        _projectMetrics.value = metricsMap
    }

    private suspend fun loadLandscape() {
        _attractorLandscape.value = AutonomousProjectEngine.getAttractors()
        _attractorConnections.value = AutonomousProjectEngine.getAttractorConnections()
    }

    private suspend fun loadSeeds() {
        _initiativeSeeds.value = AutonomousProjectEngine.getInitiativeSeeds()
    }

    suspend fun evaluateAndLaunchSeed(seed: InitiativeSeed): Boolean {
        return AutonomousProjectEngine.evaluateAndLaunch(seed)
    }
}
