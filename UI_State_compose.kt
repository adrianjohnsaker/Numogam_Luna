@Composable
fun NumogramScreen(viewModel: NumogramViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()

    when (uiState) {
        is NumogramUiState.Idle -> {
            Text("Waiting for input...")
        }
        is NumogramUiState.Loading -> {
            CircularProgressIndicator()
        }
        is NumogramUiState.Success -> {
            val data = (uiState as NumogramUiState.Success).data
            Text("Next Zone: ${data.optString("next_zone")}")
        }
        is NumogramUiState.Error -> {
            val message = (uiState as NumogramUiState.Error).message
            Text("Error: $message", color = Color.Red)
        }
    }

    Button(onClick = {
        viewModel.performTransition("user123", "1", 0.9)
    }) {
        Text("Run Transition")
    }
}
