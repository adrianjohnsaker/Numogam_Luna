import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Composable
fun CombinedVisualizer() {
    var currentState by remember { mutableStateOf(ConsciousnessState.Calm) }
    var foldPoints by remember { mutableStateOf(0) }
    var flashColor by remember { mutableStateOf(Color.Gray) }
    val scope = rememberCoroutineScope()

    // Transition animations for consciousness states
    val transition = updateTransition(targetState = currentState, label = "State Transition")

    val color by transition.animateColor(label = "Color Animation") { state ->
        when (state) {
            ConsciousnessState.Calm -> Color(0xFFB3E5FC)
            ConsciousnessState.Alert -> Color(0xFFFFC107)
            ConsciousnessState.Dreaming -> Color(0xFF7E57C2)
        }
    }

    val size by transition.animateDp(label = "Size Animation") { state ->
        when (state) {
            ConsciousnessState.Calm -> 120.dp
            ConsciousnessState.Alert -> 180.dp
            ConsciousnessState.Dreaming -> 140.dp
        }
    }

    // Main layout
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Consciousness State Visualizer
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Canvas(modifier = Modifier.size(size)) {
                drawCircle(color = color)
            }

            Spacer(modifier = Modifier.height(16.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                StateButton("Calm") { currentState = ConsciousnessState.Calm }
                StateButton("Alert") { currentState = ConsciousnessState.Alert }
                StateButton("Dreaming") { currentState = ConsciousnessState.Dreaming }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Fold Point Visualizer
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(150.dp)
                    .clip(CircleShape)
                    .background(flashColor),
                contentAlignment = Alignment.Center
            ) {
                Text(text = "Fold Points: $foldPoints", color = Color.White)
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = {
                scope.launch {
                    foldPoints++
                    flashColor = Color.Magenta
                    delay(300)
                    flashColor = Color.Gray
                }
            }) {
                Text("Trigger Fold Point")
            }
        }
    }
}

@Composable
fun StateButton(label: String, onClick: () -> Unit) {
    Button(onClick = onClick) {
        Text(text = label)
    }
}

enum class ConsciousnessState {
    Calm, Alert, Dreaming
}
