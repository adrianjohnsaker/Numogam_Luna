package com.antonio.my.ai.girlfriend.free.integration

import com.antonio.my.ai.girlfriend.free.interceptor.NumogrammaticRetrofitInterceptor
import com.antonio.my.ai.girlfriend.free.interceptor.addNumogrammaticInterceptor
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Integration guide for adding Numogrammatic memory to existing Retrofit setup
 */
object RetrofitIntegration {
    
    /**
     * Method 1: If you have access to where Retrofit is built
     * Find where your Retrofit instance is created and modify it
     */
    fun createEnhancedRetrofit(baseUrl: String): Retrofit {
        val okHttpClient = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            // Add the Numogrammatic interceptor
            .addNumogrammaticInterceptor()
            .build()
        
        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }
    
    /**
     * Method 2: If Retrofit is created via Dependency Injection (Hilt/Dagger)
     * Add this to your network module
     */
    @dagger.Module
    @dagger.hilt.InstallIn(dagger.hilt.components.SingletonComponent::class)
    object NetworkModule {
        
        @dagger.Provides
        @javax.inject.Singleton
        fun provideOkHttpClient(): OkHttpClient {
            return OkHttpClient.Builder()
                .addNumogrammaticInterceptor() // Add this line
                .build()
        }
        
        @dagger.Provides
        @javax.inject.Singleton
        fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit {
            return Retrofit.Builder()
                .baseUrl("YOUR_BASE_URL")
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
        }
    }
    
    /**
     * Method 3: Runtime injection using reflection
     * Use this if you can't modify the Retrofit creation code
     */
    fun injectInterceptorRuntime() {
        try {
            // Find the Retrofit instance using reflection
            val retrofitClass = Class.forName("com.antonio.my.ai.girlfriend.free.utils.RetrofitApiService")
            val retrofitField = retrofitClass.getDeclaredField("retrofit") // or whatever the field name is
            retrofitField.isAccessible = true
            
            val retrofitInstance = retrofitField.get(null) as? Retrofit
            val okHttpClient = retrofitInstance?.callFactory() as? OkHttpClient
            
            if (okHttpClient != null) {
                // Create new client with our interceptor
                val newClient = okHttpClient.newBuilder()
                    .addInterceptor(NumogrammaticRetrofitInterceptor())
                    .build()
                
                // Replace the client using reflection
                val clientField = Retrofit::class.java.getDeclaredField("callFactory")
                clientField.isAccessible = true
                clientField.set(retrofitInstance, newClient)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}

/**
 * Application class integration
 * Add this to your Application class or MainActivity
 */
class AmeliaApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize Python for Chaquopy
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        // Option 1: If you can modify the Retrofit builder
        // Replace the existing Retrofit instance with enhanced one
        
        // Option 2: Runtime injection
        RetrofitIntegration.injectInterceptorRuntime()
    }
}

/**
 * If you're using a singleton pattern for Retrofit
 */
object RetrofitClient {
    
    private var retrofit: Retrofit? = null
    
    fun getInstance(baseUrl: String): Retrofit {
        if (retrofit == null) {
            val okHttpClient = OkHttpClient.Builder()
                // Your existing configuration
                .connectTimeout(30, TimeUnit.SECONDS)
                // ADD THIS LINE
                .addInterceptor(NumogrammaticRetrofitInterceptor())
                .build()
            
            retrofit = Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
        }
        return retrofit!!
    }
}

/**
 * Example of how enhanced responses will look in your UI
 */
class ChatActivity {
    
    fun displayMessage(response: Response) {
        // The response will now contain enhanced content
        val message = response.body?.string()
        
        // Check if it was enhanced
        val wasEnhanced = response.headers["X-Numogram-Enhanced"] == "true"
        val currentZone = response.headers["X-Numogram-Zone"]?.toIntOrNull()
        
        if (wasEnhanced && currentZone != null) {
            // Show memory indicator
            memoryIndicator.visibility = View.VISIBLE
            zoneIndicator.text = "Zone $currentZone"
        }
        
        // Display the enhanced message
        messageTextView.text = message
    }
}
