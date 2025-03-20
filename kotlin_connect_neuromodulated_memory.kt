import okhttp3.*
import org.json.JSONObject

val client = OkHttpClient()

fun encodeMemory(
    key: String,
    value: Float,
    salience: Float,
    uncertainty: Float
): String {
    val json = JSONObject().apply {
        put("key", key)
        put("value", value)
        put("salience", salience)
        put("uncertainty", uncertainty)
    }

    val requestBody = json.toString().toRequestBody("application/json".toMediaTypeOrNull())
    val request = Request.Builder()
        .url("http://localhost:8000/encode_memory/")
        .post(requestBody)
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}

fun reinforceMemory(key: String): String {
    val json = JSONObject().apply {
        put("key", key)
    }

    val requestBody = json.toString().toRequestBody("application/json".toMediaTypeOrNull())
    val request = Request.Builder()
        .url("http://localhost:8000/reinforce_memory/")
        .post(requestBody)
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}

fun decayMemories(): String {
    val request = Request.Builder()
        .url("http://localhost:8000/decay_memories/")
        .post("".toRequestBody()) 
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}

fun bayesianUpdate(key: String, likelihood: Float, prior: Float): String {
    val json = JSONObject().apply {
        put("key", key)
        put("likelihood", likelihood)
        put("prior", prior)
    }

    val requestBody = json.toString().toRequestBody("application/json".toMediaTypeOrNull())
    val request = Request.Builder()
        .url("http://localhost:8000/bayesian_update/")
        .post(requestBody)
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}

fun retrieveMemory(key: String): String {
    val request = Request.Builder()
        .url("http://localhost:8000/retrieve_memory/?key=$key")
        .get()
        .build()

    client.newCall(request).execute().use { response ->
        return response.body?.string() ?: "Error: No Response"
    }
}
