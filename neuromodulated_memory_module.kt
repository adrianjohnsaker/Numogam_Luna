package com.example.neuromodulatedmemory

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.TimeUnit

class NeuroMemoryClient(private val context: Context) {
    private val baseUrl: String = "http://10.0.2.2:8000"

    suspend fun storeMemory(key: String, data: String, salience: Float, uncertainty: Float): String {
        val requestBody = JSONObject().apply {
            put("key", key)
            put("data", data)
            put("salience", salience)
            put("uncertainty", uncertainty)
        }
        return makeRequest("/store_memory/", "POST", requestBody.toString())
    }

    suspend fun exportMemory(compress: Boolean = false): String {
        return makeRequest("/export_memory/?compress=$compress", "GET")
    }

    private suspend fun makeRequest(endpoint: String, method: String, body: String? = null): String {
        return withContext(Dispatchers.IO) {
            val urlConnection = URL("$baseUrl$endpoint").openConnection() as HttpURLConnection
            try {
                urlConnection.requestMethod = method
                urlConnection.readTimeout = TimeUnit.SECONDS.toMillis(15).toInt()
                urlConnection.connectTimeout = TimeUnit.SECONDS.toMillis(15).toInt()

                if (body != null) {
                    urlConnection.doOutput = true
                    urlConnection.setRequestProperty("Content-Type", "application/json")
                    urlConnection.outputStream.use { it.write(body.toByteArray()) }
                }

                val responseCode = urlConnection.responseCode
                return@withContext if (responseCode in 200..299) {
                    urlConnection.inputStream.bufferedReader().use { it.readText() }
                } else {
                    val errorMsg = urlConnection.errorStream?.bufferedReader()?.readText()
                    throw Exception("HTTP error $responseCode: $errorMsg")
                }
            } finally {
                urlConnection.disconnect()
            }
        }
    }
}
