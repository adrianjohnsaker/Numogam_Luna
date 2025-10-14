package com.antonio.my.ai.girlfriend.free.amelia.ui.visual

import android.Manifest
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.amelia.bridge.PythonModuleController
import org.json.JSONObject
import java.io.File

class MainActivitySystemVisualizer : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize Chaquopy runtime if not yet initialized
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        // Request storage permission (required for saving images to /sdcard/Download)
        val requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean -> }

        requestPermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)

        setContent {
            SystemVisualizerApp()
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun SystemVisualizerApp() {
        val scrollState = rememberScrollState()
        var isLoading by remember { mutableStateOf(false) }
        var graphBitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
        var metricsBitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
        var statusMessage by remember { mutableStateOf("Press Generate to visualize system data.") }

        val sampleJson = """
            {
              "metrics": {
                "time": [0,1,2,3,4,5,6,7,8,9],
                "cluster_count": [2,3,4,5,4,5,6,7,7,8],
                "observer_influence": [1.2,1.8,2.0,2.6,2.7,3.1,3.4,3.7,4.0,4.3],
                "loop_strength": [0.5,0.8,1.2,1.6,1.5,1.8,2.0,2.3,2.7,3.0]
              },
              "feedback_loops": [
                {"source": "A", "target": "B", "strength": 2.3, "type": "system_observer"},
                {"source": "B", "target": "C", "strength": 1.5, "type": "regular"},
                {"source": "C", "target": "A", "strength": 0.9, "type": "regular"}
              ]
            }
        """.trimIndent()

        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("System Visualizer", fontWeight = FontWeight.Bold, fontSize = 20.sp) }
                )
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .padding(padding)
                    .padding(16.dp)
                    .fillMaxSize()
                    .verticalScroll(scrollState),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                Text(
                    text = "Complex System Visualization Toolkit",
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp
                )

                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = {
                        isLoading = true
                        statusMessage = "Running visualization..."
                        try {
                            val result = PythonModuleController.runSystemVisualizer(sampleJson)
                            val status = result.optString("status", "error")
                            if (status == "success") {
                                val paths = result.getJSONObject("paths")
                                val graphPath = paths.getString("graph")
                                val metricsPath = paths.getString("metrics")

                                val graphFile = File(graphPath)
                                val metricsFile = File(metricsPath)

                                if (graphFile.exists()) {
                                    graphBitmap = BitmapFactory.decodeFile(graphFile.absolutePath)
                                }
                                if (metricsFile.exists()) {
                                    metricsBitmap = BitmapFactory.decodeFile(metricsFile.absolutePath)
                                }

                                statusMessage = "Visualization complete."
                            } else {
                                statusMessage = "Error: ${result.optString("message")}"
                            }
                        } catch (e: Exception) {
                            statusMessage = "Exception: ${e.message}"
                        }
                        isLoading = false
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isLoading
                ) {
                    Text(text = if (isLoading) "Generating..." else "Generate Visualization")
                }

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = statusMessage,
                    color = MaterialTheme.colorScheme.secondary,
                    fontSize = 14.sp
                )

                Spacer(modifier = Modifier.height(24.dp))

                if (graphBitmap != null) {
                    Text("Feedback Graph", fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Image(
                        bitmap = graphBitmap!!.asImageBitmap(),
                        contentDescription = "Feedback Graph",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(300.dp)
                            .padding(8.dp)
                    )
                }

                if (metricsBitmap != null) {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("System Metrics", fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Image(
                        bitmap = metricsBitmap!!.asImageBitmap(),
                        contentDescription = "System Metrics",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(300.dp)
                            .padding(8.dp)
                    )
                }
            }
        }
    }
}MainActivitySystemVisualizer.kt
