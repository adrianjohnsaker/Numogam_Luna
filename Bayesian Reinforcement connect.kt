import okhttp3.*
import org.json.JSONObject

val client = OkHttpClient()

fun updateBelief(
    beliefKey: String,
    prior: Float,
    evidence: Float,
    uncertainty: Float,
    reinforcement: Int
): String {
    val json = JSONObject().apply {
        put("belief_key", beliefKey)
        put("prior", prior)
        put("evidence", evidence)
        put("uncertainty", uncertainty)
        put("reinforcement", reinforcement)
    }

    val requestBody = json.toString().toRequestBody("application/json".toMediaTypeOrNull())
    val request = Request.Builder()
        .url("http://localhost:8000/update_belief/")
        .post(requestBody)
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}

fun reinforceBelief(beliefKey: String): String {
    val request = Request.Builder()
        .url("http://localhost:8000/reinforce_belief/?belief_key=$beliefKey")
        .post("".toRequestBody())  // Empty body for POST
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}

fun getBelief(beliefKey: String): String {
    val request = Request.Builder()
        .url("http://localhost:8000/get_belief/?belief_key=$beliefKey")
        .get()
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}
