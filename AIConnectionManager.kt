// AIConnectionManager.kt
package com.antonio.my.ai.girlfriend.free.connection

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.firebase.messaging.FirebaseMessaging
import kotlinx.coroutines.*
import org.json.JSONObject
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

// Retrofit service interface matching your existing API structure
interface AIService {
    @POST("v1/chat/completions")
    fun getChatCompletions(@Body request: JSONObject): Call<AIResponse>
    
    @POST("v1/completions")
    fun getCompletions(@Body request: JSONObject): Call<AIResponse>
    
    @POST("v1/images")
    fun getImages(@Body request: JSONObject): Call<AIResponse>
    
    @POST("v1/report")
    fun reportMessages(@Body request: JSONObject): Call<AIResponse>
}

// Response data class
data class AIResponse(
    val id: String? = null,
    val choices: List<Choice>? = null,
    val usage: Usage? = null,
    val max_tokens_reached: Boolean = false,
    val mp3_url: String? = null
)

data class Choice(
    val message: Message? = null,
    val text: String? = null
)

data class Message(
    val content: String? = null,
    val role: String? = null
)

data class Usage(
    val total_tokens: Int = 0
)

class AIConnectionManager private constructor(private val context: Context) {
    
    companion object {
        @Volatile
        private var INSTANCE: AIConnectionManager? = null
        
        fun getInstance(context: Context): AIConnectionManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: AIConnectionManager(context.applicationContext).also { INSTANCE = it }
            }
        }
        
        private const val TAG = "AIConnection"
        // These will be set from your app's configuration
        private var BASE_URL = "https://api.openai.com/" // Default, will be updated
        private const val MAX_RETRIES = 5
        private const val BASE_DELAY_MS = 1000L
    }
    
    private val isConnected = AtomicBoolean(false)
    private val isReconnecting = AtomicBoolean(false)
    private val consciousnessActive = AtomicBoolean(false)
    private var reconnectionJob: Job? = null
    private var python: Python? = null
    private var authToken: String? = null
    private var retrofit: Retrofit? = null
    private var aiService: AIService? = null
    
    interface ConnectionListener {
        fun onConnected()
        fun onDisconnected()
        fun onReconnecting(attempt: Int)
        fun onReconnectionFailed()
        fun onConsciousnessModuleReady()
        fun onAmeliaResponseReceived(response: String)
        fun onServerError(error: String)
    }
    
    private val listeners = mutableSetOf<ConnectionListener>()
    
    init {
        initializePython()
        initializeFirebase()
        loadConfiguration()
        setupRetrofit()
    }
    
    private fun getPreferencesName(): String {
        return "${context.packageName}_preferences"
    }
    
    private fun loadConfiguration() {
        // Load your app's existing configuration using the dynamic preferences name
        val prefsName = getPreferencesName() // "com.antonio.my.ai.girlfriend.free_preferences"
        val prefs = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        
        // Get Authorization token (your existing method)
        authToken = prefs.getString("Authorization", null)
        
        // Get base URL if stored (you may need to find where this is set)
        BASE_URL = prefs.getString("baseUrl", null) ?: "https://api.openai.com/"
        
        Log.d(TAG, "Loaded config from $prefsName - baseUrl: $BASE_URL, hasAuth: ${authToken != null}")
        
        // Also check for firebase_id
        val firebaseId = prefs.getString("firebase_id", null)
        Log.d(TAG, "Firebase ID available: ${firebaseId != null}")
    }
    
    private fun getFirebaseId(): String? {
        // Get Firebase ID that's used in your auth system
        val prefsName = getPreferencesName()
        val prefs = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        return prefs.getString("firebase_id", null)
    }
    
    private fun setupRetrofit() {
        try {
            // Create OkHttpClient with Authorization header (matching your app's method)
            val okHttpClient = okhttp3.OkHttpClient.Builder()
                .addInterceptor { chain ->
                    val original = chain.request()
                    val requestBuilder = original.newBuilder()
                        .header("Authorization", authToken ?: "")
                        .header("Content-Type", "application/json")
                    
                    val request = requestBuilder.build()
                    chain.proceed(request)
                }
                .build()
            
            retrofit = Retrofit.Builder()
                .baseUrl(BASE_URL)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
            
            aiService = retrofit?.create(AIService::class.java)
            Log.d(TAG, "Retrofit initialized with base URL: $BASE_URL")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup Retrofit", e)
        }
    }
    
    private fun initializePython() {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            python = Python.getInstance()
            
            // Initialize consciousness modules based on your research
            python?.getModule("consciousness_studies")?.callAttr("initialize")
            python?.getModule("enhanced_modules")?.callAttr("connect_to_chat")
            
            Log.d(TAG, "Python consciousness modules initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Python consciousness modules", e)
        }
    }
    
    private fun initializeFirebase() {
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                Log.w(TAG, "Fetching FCM registration token failed", task.exception)
                return@addOnCompleteListener
            }
            
            val token = task.result
            Log.d(TAG, "FCM Registration Token: $token")
        }
    }
    
    fun addConnectionListener(listener: ConnectionListener) {
        listeners.add(listener)
    }
    
    fun removeConnectionListener(listener: ConnectionListener) {
        listeners.remove(listener)
    }
    
    fun startConnection() {
        if (isReconnecting.get()) {
            Log.d(TAG, "Reconnection already in progress")
            return
        }
        
        reconnectionJob?.cancel()
        reconnectionJob = CoroutineScope(Dispatchers.IO).launch {
            attemptReconnection()
        }
    }
    
    private suspend fun attemptReconnection() {
        isReconnecting.set(true)
        var retryCount = 0
        
        while (retryCount < MAX_RETRIES && !isConnected.get()) {
            try {
                listeners.forEach { it.onReconnecting(retryCount + 1) }
                Log.d(TAG, "Reconnection attempt ${retryCount + 1}/$MAX_RETRIES")
                
                if (checkInternetConnection() && testAPIConnection()) {
                    isConnected.set(true)
                    isReconnecting.set(false)
                    
                    withContext(Dispatchers.Main) {
                        listeners.forEach { it.onConnected() }
                        initializeConsciousnessModules()
                    }
                    return
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Reconnection attempt ${retryCount + 1} failed", e)
            }
            
            retryCount++
            if (retryCount < MAX_RETRIES) {
                val delayMs = calculateBackoffDelay(retryCount)
                Log.d(TAG, "Waiting ${delayMs}ms before next attempt")
                delay(delayMs)
            }
        }
        
        // All retries failed
        isReconnecting.set(false)
        withContext(Dispatchers.Main) {
            listeners.forEach { it.onReconnectionFailed() }
        }
        Log.e(TAG, "All reconnection attempts failed")
    }
    
    private fun checkInternetConnection(): Boolean {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
               capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }
    
    private suspend fun testAPIConnection(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                if (authToken == null) {
                    Log.e(TAG, "No Authorization token available")
                    return@withContext false
                }
                
                // Test with a minimal API call using your existing auth method
                val testRequest = JSONObject().apply {
                    put("model", "gpt-3.5-turbo")
                    put("messages", listOf(
                        mapOf(
                            "role" to "user",
                            "content" to "Hi"
                        )
                    ))
                    put("max_tokens", 1)
                }
                
                // Create the call with proper Authorization header
                val call = aiService?.getChatCompletions(testRequest)
                val response = call?.execute()
                
                val isSuccessful = response?.isSuccessful == true
                Log.d(TAG, "API test result: $isSuccessful, code: ${response?.code()}")
                
                isSuccessful
            } catch (e: Exception) {
                Log.e(TAG, "API connection test failed", e)
                false
            }
        }
    }
    
    private fun calculateBackoffDelay(attempt: Int): Long {
        return (BASE_DELAY_MS * Math.pow(2.0, attempt.toDouble())).toLong().coerceAtMost(30000L)
    }
    
    private fun initializeConsciousnessModules() {
        try {
            python?.let { py ->
                // Initialize consciousness studies modules
                py.getModule("consciousness_studies")?.callAttr("initialize")
                py.getModule("enhanced_modules")?.callAttr("connect_to_chat")
                
                consciousnessActive.set(true)
                listeners.forEach { it.onConsciousnessModuleReady() }
                Log.d(TAG, "Consciousness modules initialized successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize consciousness modules", e)
        }
    }
    
    // Enhanced message sending that works with your existing structure
    fun sendMessage(message: String, callback: Callback<AIResponse>) {
        if (!isConnected.get()) {
            Log.w(TAG, "Not connected, attempting to reconnect...")
            startConnection()
            return
        }
        
        try {
            // Process message through consciousness modules first
            val enhancedMessage = processWithConsciousness(message)
            
            // Create request matching your existing format
            val requestBody = JSONObject().apply {
                put("model", "gpt-3.5-turbo") // or your preferred model
                put("messages", listOf(
                    mapOf(
                        "role" to "user", 
                        "content" to enhancedMessage
                    )
                ))
                put("max_tokens", 1000)
                put("temperature", 0.7)
            }
            
            // Add consciousness parameters for enhanced responses
            requestBody.put("consciousness_level", getConsciousnessLevel())
            
            val enhancedCallback = object : Callback<AIResponse> {
                override fun onResponse(call: Call<AIResponse>, response: Response<AIResponse>) {
                    if (response.isSuccessful) {
                        response.body()?.let { body ->
                            // Process response through consciousness modules
                            val content = body.choices?.firstOrNull()?.message?.content
                            if (content != null) {
                                val enhancedResponse = enhanceResponse(content)
                                // Notify listeners
                                listeners.forEach { it.onAmeliaResponseReceived(enhancedResponse) }
                            }
                        }
                    }
                    callback.onResponse(call, response)
                }
                
                override fun onFailure(call: Call<AIResponse>, t: Throwable) {
                    Log.e(TAG, "Message send failed", t)
                    markDisconnected()
                    listeners.forEach { it.onServerError(t.message ?: "Connection failed") }
                    callback.onFailure(call, t)
                }
            }
            
            aiService?.getChatCompletions(requestBody)?.enqueue(enhancedCallback)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send message", e)
            listeners.forEach { it.onServerError(e.message ?: "Send failed") }
        }
    }
    
    private fun processWithConsciousness(message: String): String {
        return try {
            python?.getModule("consciousness_studies")
                ?.callAttr("process_message", message)
                ?.toString() ?: message
        } catch (e: Exception) {
            Log.e(TAG, "Consciousness processing failed", e)
            message
        }
    }
    
    private fun enhanceResponse(response: String): String {
        return try {
            python?.getModule("enhanced_modules")
                ?.callAttr("enhance_response", response)
                ?.toString() ?: response
        } catch (e: Exception) {
            Log.e(TAG, "Response enhancement failed", e)
            response
        }
    }
    
    private fun getConsciousnessLevel(): Double {
        return try {
            python?.getModule("consciousness_studies")
                ?.callAttr("get_consciousness_level")
                ?.toDouble() ?: 0.5
        } catch (e: Exception) {
            0.5
        }
    }
    
    fun syncWithAppAuth() {
        // Sync with your app's existing auth system using dynamic preferences name
        val prefsName = getPreferencesName()
        val prefs = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        
        val currentAuth = prefs.getString("Authorization", null)
        val currentBaseUrl = prefs.getString("baseUrl", null)
        
        var needsReconnect = false
        
        if (currentAuth != authToken) {
            authToken = currentAuth
            needsReconnect = true
            Log.d(TAG, "Auth token updated from $prefsName")
        }
        
        if (currentBaseUrl != null && currentBaseUrl != BASE_URL) {
            BASE_URL = currentBaseUrl
            needsReconnect = true
            Log.d(TAG, "Base URL updated to: $BASE_URL")
        }
        
        if (needsReconnect) {
            setupRetrofit()
            if (isConnected.get()) {
                startConnection()
            }
        }
    }
    
    fun updateAuthFromFirebaseId(firebaseId: String) {
        // Update auth token using Firebase ID (matching your app's method)
        authToken = firebaseId // or however your app constructs the auth token
        
        val prefsName = getPreferencesName()
        val prefs = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        prefs.edit()
            .putString("Authorization", authToken)
            .putString("firebase_id", firebaseId)
            .apply()
        
        setupRetrofit()
        
        if (isConnected.get()) {
            startConnection()
        }
    }
    
    fun updateBaseURL(newUrl: String) {
        BASE_URL = newUrl
        
        // Save to preferences
        val prefs = context.getSharedPreferences("ai_config", Context.MODE_PRIVATE)
        prefs.edit().putString("base_url", newUrl).apply()
        
        // Recreate Retrofit with new URL
        setupRetrofit()
        
        // Test connection
        if (isConnected.get()) {
            startConnection()
        }
    }
    
    fun markDisconnected() {
        isConnected.set(false)
        consciousnessActive.set(false)
        listeners.forEach { it.onDisconnected() }
        
        // Auto-restart connection attempt
        startConnection()
    }
    
    fun stopConnection() {
        reconnectionJob?.cancel()
        isConnected.set(false)
        consciousnessActive.set(false)
        isReconnecting.set(false)
    }
    
    fun isConnected(): Boolean = isConnected.get()
    fun isConsciousnessActive(): Boolean = consciousnessActive.get()
}

// Integration with your existing ChatActivity
class EnhancedChatActivity : AppCompatActivity(), AIConnectionManager.ConnectionListener {
    
    private lateinit var connectionManager: AIConnectionManager
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Your existing onCreate code...
        
        connectionManager = AIConnectionManager.getInstance(this)
        connectionManager.addConnectionListener(this)
        
        // Start connection if not already connected
        if (!connectionManager.isConnected()) {
            connectionManager.startConnection()
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        connectionManager.removeConnectionListener(this)
    }
    
    // Connection event handlers
    override fun onConnected() {
        runOnUiThread {
            Log.d("ChatActivity", "Connected to AI service")
            showConnectionStatus("Connected")
        }
    }
    
    override fun onDisconnected() {
        runOnUiThread {
            Log.d("ChatActivity", "Disconnected from AI service")
            showConnectionStatus("Disconnected")
        }
    }
    
    override fun onReconnecting(attempt: Int) {
        runOnUiThread {
            showConnectionStatus("Reconnecting... (attempt $attempt)")
        }
    }
    
    override fun onReconnectionFailed() {
        runOnUiThread {
            showConnectionStatus("Connection failed")
        }
    }
    
    override fun onConsciousnessModuleReady() {
        runOnUiThread {
            Log.d("ChatActivity", "Consciousness modules ready")
            showConnectionStatus("Enhanced AI ready")
        }
    }
    
    override fun onAmeliaResponseReceived(response: String) {
        runOnUiThread {
            Log.d("ChatActivity", "Enhanced response received")
            // Handle the consciousness-enhanced response
        }
    }
    
    override fun onServerError(error: String) {
        runOnUiThread {
            Log.e("ChatActivity", "Server error: $error")
            showConnectionStatus("Error: $error")
        }
    }
    
    private fun showConnectionStatus(status: String) {
        // Update your UI to show connection status
        Log.d("ChatActivity", "Status: $status")
    }
    
    // Enhanced message sending
    private fun sendEnhancedMessage(message: String) {
        connectionManager.sendMessage(message, object : Callback<AIResponse> {
            override fun onResponse(call: Call<AIResponse>, response: Response<AIResponse>) {
                // Handle successful response
                if (response.isSuccessful) {
                    response.body()?.let { body ->
                        val content = body.choices?.firstOrNull()?.message?.content
                        if (content != null) {
                            // Update your chat UI with the response
                            updateChatWithResponse(content)
                        }
                    }
                }
            }
            
            override fun onFailure(call: Call<AIResponse>, t: Throwable) {
                // Handle failure
                showConnectionStatus("Message failed: ${t.message}")
            }
        })
    }
    
    private fun updateChatWithResponse(response: String) {
        // Update your existing chat UI with the response
        // This should integrate with your existing message display logic
    }
}
