@Composable
fun NumogramScreen(viewModel: NumogramViewModel = viewModel()) {
    val state by viewModel.state.collectAsState()

    Column(modifier = Modifier.padding(16.dp)) {
        Text("Next Zone: ${state.optString("next_zone", "...")}")
        Text("Description: ${state.optJSONObject("zone_description")?.optString("theme") ?: "N/A"}")

        Button(onClick = {
            viewModel.performTransition("user123", "1", 0.8)
        }) {
            Text("Simulate Transition")
        }
    }
}
