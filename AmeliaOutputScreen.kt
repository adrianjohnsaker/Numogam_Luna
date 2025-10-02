@Composable
fun AmeliaOutputScreen() {
    val stateFlow = AmeliaStateBus.events
    val latest by stateFlow.collectAsState(initial = JSONObject())

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Amelia Zone: ${latest.optString("zone_label", "–")}")
        Text("Fold: ${latest.optString("fold_mode", "–")} (${latest.optDouble("fold", 0.0)})")
        Text("Gloss: ${latest.optString("gloss", "–")}")
        Spacer(Modifier.height(12.dp))
        Text(latest.optString("composite", ""))
    }
}
