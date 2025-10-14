package com.example.neuromodulatedmemory

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.util.*
import kotlin.concurrent.fixedRateTimer

/**
 * MainActivity demonstrating integration of NeuromodulatedMemory in an Android app
 */
class MainActivity : ComponentActivity() {

    // Initialize the ViewModel with application context and lifecycle scope
    private lateinit var viewModel: NeuroMemoryViewModel
    
    // Timer for periodic memory maintenance
    private var maintenanceTimer: Timer? = null
    
    // Track query results
    private val memoryResults = mutableStateListOf<MemoryItem>()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize ViewModel
        viewModel = NeuroMemoryViewModel(applicationContext, lifecycleScope)
        
        // Start periodic memory maintenance
        startMemoryMaintenance()
        
        setContent {
            NeuroMemoryUI(
                onStoreMemory = { key, data, salience, uncertainty ->
                    storeMemory(key, data, salience, uncertainty)
                },
                onQueryMemories = { query ->
                    queryMemories(query)
                },
                onUpdateBelief = { prior, evidence, reliability ->
                    updateBelief(prior, evidence, reliability)
                },
                memoryResults = memoryResults
            )
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        maintenanceTimer?.cancel()
    }
    
    /**
     * Sets up periodic memory maintenance to run every 30 minutes
     */
    private fun startMemoryMaintenance() {
        maintenanceTimer = fixedRateTimer("MemoryMaintenance", true, 
            0L, 30 * 60 * 1000) {
            viewModel.performMemoryMaintenance { removed, remaining ->
                Log.d("MainActivity", "Memory maintenance: removed $removed, remaining $remaining")
                
                // Only show user notification if memories were removed
                if (removed > 0) {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, 
                            "Memory maintenance complete: $removed memories faded, $remaining remain", 
                            Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }
    
    /**
     * Stores a new memory using the ViewModel
     */
    private fun storeMemory(key: String, data: String, salience: Float, uncertainty: Float) {
        viewModel.storeUserExperience(
            experienceId = key,
            description = data,
            importance = salience,
            confidenceLevel = 1.0f - uncertainty
        ) { success, message ->
            runOnUiThread {
                val statusMessage = if (success) "Memory stored successfully" else message
                Toast.makeText(this, statusMessage, Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    /**
     * Queries memories by text similarity
     */
    private fun queryMemories(query: String) {
        lifecycleScope.launch {
            try {
                val client = NeuroMemoryClient(applicationContext, lifecycleScope)
                
                client.queryMemories(
                    query = query,
                    threshold = 0.5f,
                    onSuccess = { result ->
                        Log.d("MainActivity", "Query results: $result")
                        
                        // Clear previous results
                        memoryResults.clear()
                        
                        // Parse result and update UI
                        val resultsObj = result.getJSONObject("results")
                        val keys = resultsObj.keys()
                        
                        while (keys.hasNext()) {
                            val key = keys.next()
                            val memory = resultsObj.getJSONObject(key)
                            
                            memoryResults.add(
                                MemoryItem(
                                    key = key,
                                    data = memory.getString("data"),
                                    weight = memory.getDouble("weight").toFloat(),
                                    timestamp = memory.getLong("timestamp")
                                )
                            )
                        }
                        
                        // Show message if no results
                        if (memoryResults.isEmpty()) {
                            Toast.makeText(this@MainActivity, 
                                "No memories found matching your query", 
                                Toast.LENGTH_SHORT).show()
                        }
                    },
                    onError = { error ->
                        Log.e("MainActivity", "Query failed: $error")
                        Toast.makeText(this@MainActivity, 
                            "Query failed: $error", 
                            Toast.LENGTH_SHORT).show()
                    }
                )
            } catch (e: Exception) {
                Log.e("MainActivity", "Error querying memories", e)
                Toast.makeText(this@MainActivity, 
                    "Error: ${e.message}", 
                    Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    /**
     * Updates a belief using Bayesian updating with neuromodulation
     */
    private fun updateBelief(prior: Float, evidence: Float, reliability: Float) {
        viewModel.updateUserBelief(
            currentBelief = prior,
            newEvidenceStrength = evidence,
            evidenceReliability = reliability
        ) { updatedBelief ->
            runOnUiThread {
                Toast.makeText(this, 
                    "Belief updated: $prior → $updatedBelief", 
                    Toast.LENGTH_LONG).show()
            }
        }
    }
}

/**
 * Data class for memory items in the UI
 */
data class MemoryItem(
    val key: String,
    val data: String,
    val weight: Float,
    val timestamp: Long
)

/**
 * Compose UI for the Neuromodulated Memory app
 */
@Composable
fun NeuroMemoryUI(
    onStoreMemory: (String, String, Float, Float) -> Unit,
    onQueryMemories: (String) -> Unit,
    onUpdateBelief: (Float, Float, Float) -> Unit,
    memoryResults: List<MemoryItem>
) {
    var tabIndex by remember { mutableStateOf(0) }
    val tabs = listOf("Store Memory", "Query Memory", "Update Belief")
    
    MaterialTheme {
        Column(modifier = Modifier.fillMaxSize()) {
            TabRow(selectedTabIndex = tabIndex) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = tabIndex == index,
                        onClick = { tabIndex = index },
                        text = { Text(title) }
                    )
                }
            }
            
            when (tabIndex) {
                0 -> StoreMemoryScreen(onStoreMemory)
                1 -> QueryMemoryScreen(onQueryMemories, memoryResults)
                2 -> UpdateBeliefScreen(onUpdateBelief)
            }
        }
    }
}

/**
 * Screen for storing memories
 */
@Composable
fun StoreMemoryScreen(onStoreMemory: (String, String, Float, Float) -> Unit) {
    var key by remember { mutableStateOf("") }
    var data by remember { mutableStateOf("") }
    var salience by remember { mutableStateOf(0.5f) }
    var uncertainty by remember { mutableStateOf(0.3f) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Store New Memory",
            style = MaterialTheme.typography.headlineSmall
        )
        
        OutlinedTextField(
            value = key,
            onValueChange = { key = it },
            label = { Text("Memory Key") },
            modifier = Modifier.fillMaxWidth()
        )
        
        OutlinedTextField(
            value = data,
            onValueChange = { data = it },
            label = { Text("Memory Content") },
            modifier = Modifier.fillMaxWidth().height(120.dp)
        )
        
        Text("Salience: ${String.format("%.2f", salience)}")
        Slider(
            value = salience,
            onValueChange = { salience = it },
            valueRange = 0f..1f,
            steps = 20,
            modifier = Modifier.fillMaxWidth()
        )
        
        Text("Uncertainty: ${String.format("%.2f", uncertainty)}")
        Slider(
            value = uncertainty,
            onValueChange = { uncertainty = it },
            valueRange = 0f..1f,
            steps = 20,
            modifier = Modifier.fillMaxWidth()
        )
        
        Button(
            onClick = { 
                if (key.isNotBlank() && data.isNotBlank()) {
                    onStoreMemory(key, data, salience, uncertainty)
                    key = ""
                    data = ""
                }
            },
            modifier = Modifier.align(Alignment.End)
        ) {
            Text("Store Memory")
        }
    }
}

/**
 * Screen for querying memories
 */
@Composable
fun QueryMemoryScreen(
    onQueryMemories: (String) -> Unit,
    memoryResults: List<MemoryItem>
) {
    var query by remember { mutableStateOf("") }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Query Memories",
            style = MaterialTheme.typography.headlineSmall
        )
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                label = { Text("Search Query") },
                modifier = Modifier.weight(1f)
            )
            
            Spacer(modifier = Modifier.width(8.dp))
            
            Button(
                onClick = { 
                    if (query.isNotBlank()) {
                        onQueryMemories(query)
                    }
                }
            ) {
                Text("Search")
            }
        }
        
        if (memoryResults.isNotEmpty()) {
            Text(
                text = "Results (${memoryResults.size})",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            LazyColumn(
                modifier = Modifier.fillMaxWidth()
            ) {
                items(memoryResults) { item ->
                    MemoryResultItem(item)
                    Divider()
                }
            }
        }
    }
}

/**
 * Individual memory result item
 */
@Composable
fun MemoryResultItem(item: MemoryItem) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Text(
            text = item.key,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        
        Text(
            text = item.data,
            style = MaterialTheme.typography.bodyMedium
        )
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "Weight: ${String.format("%.2f", item.weight)}",
                style = MaterialTheme.typography.bodySmall,
                color = when {
                    item.weight > 0.7f -> Color(0xFF388E3C) // Green
                    item.weight > 0.4f -> Color(0xFFFFA000) // Amber
                    else -> Color(0xFFD32F2F) // Red
                }
            )
            
            Text(
                text = "Stored: ${formatTimestamp(item.timestamp)}",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )
        }
    }
}

/**
 * Screen for updating beliefs
 */
@Composable
fun UpdateBeliefScreen(onUpdateBelief: (Float, Float, Float) -> Unit) {
    var prior by remember { mutableStateOf(0.5f) }
    var evidence by remember { mutableStateOf(1.0f) }
    var reliability by remember { mutableStateOf(0.7f) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Update Belief",
            style = MaterialTheme.typography.headlineSmall
        )
        
        Text("Prior Belief: ${String.format("%.2f", prior)}")
        Slider(
            value = prior,
            onValueChange = { prior = it },
            valueRange = 0.01f..0.99f,
            steps = 98,
            modifier = Modifier.fillMaxWidth()
        )
        
        Text("Evidence Strength: ${String.format("%.2f", evidence)}")
        Slider(
            value = evidence,
            onValueChange = { evidence = it },
            valueRange = 0.1f..10f,
            steps = 99,
            modifier = Modifier.fillMaxWidth()
        )
        
        Text("Evidence Reliability: ${String.format("%.2f", reliability)}")
        Slider(
            value = reliability,
            onValueChange = { reliability = it },
            valueRange = 0f..1f,
            steps = 20,
            modifier = Modifier.fillMaxWidth()
        )
        
        Button(
            onClick = { onUpdateBelief(prior, evidence, reliability) },
            modifier = Modifier.align(Alignment.End)
        ) {
            Text("Update Belief")
        }
    }
}

/**
 * Formats a timestamp to a readable date string
 */
fun formatTimestamp(timestamp: Long): String {
    val date = Date(timestamp * 1000) // Convert to milliseconds
    return android.text.format.DateFormat.format("MMM dd, yyyy HH:mm", date).toString()
}

/**
 * Example Application class
 */
class NeuroMemoryApplication : android.app.Application() {
    override fun onCreate() {
        super.onCreate()
        
        // Initialize any application-wide components here
        Log.d("NeuroMemoryApp", "Application initialized")
    }
}
