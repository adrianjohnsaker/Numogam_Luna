package com.example.neuromodulatedmemory

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

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

    suspend fun exportMemory(): String {
        return makeRequest("/export_memory/", "GET")
    }

    private suspend fun makeRequest(endpoint: String, method: String, body: String? = null): String {
        return withContext(Dispatchers.IO) {
            val urlConnection = URL("$baseUrl$endpoint").openConnection() as HttpURLConnection
            
            try {
                urlConnection.requestMethod = method
                
                if (body != null) {
                    urlConnection.doOutput = true
                    urlConnection.setRequestProperty("Content-Type", "application/json")
                    urlConnection.outputStream.use { it.write(body.toByteArray()) }
                }

                val responseCode = urlConnection.responseCode
                if (responseCode in 200..299) {
                    urlConnection.inputStream.bufferedReader().use { it.readText() }
                } else {
                    throw Exception("HTTP error code $responseCode")
                }
            } finally {
                urlConnection.disconnect()
            }
        }
    }
}
