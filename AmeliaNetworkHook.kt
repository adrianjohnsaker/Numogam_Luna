package com.amelia.bridge

import android.content.Context
import android.util.Log
import dalvik.system.DexFile
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import java.io.File
import java.lang.reflect.Field
import java.lang.reflect.Modifier

/**
 * Installs AmeliaInterceptor into any static OkHttpClient fields and/or Retrofit instances
 * discovered in the app’s classes at runtime. Called once from BaseApplication.onCreate().
 */
object AmeliaNetworkHook {
    private const val TAG = "AmeliaHook"

    fun install(context: Context) {
        try {
            var patched = 0
            val dex = DexFile(File(context.packageCodePath))
            val entries = dex.entries()
            val cl = context.classLoader

            while (entries.hasMoreElements()) {
                val name = entries.nextElement()

                // Keep it scoped to app code to stay fast + safe
                if (!name.startsWith("com.antonio.my.ai.girlfriend.free")
                    && !name.startsWith("com.amelia")
                    && !name.startsWith("com.antonio")
                ) continue

                runCatching {
                    val cls = Class.forName(name, false, cl)
                    for (field in cls.declaredFields) {
                        field.isAccessible = true

                        // 1) Swap static OkHttpClient fields
                        if (Modifier.isStatic(field.modifiers) &&
                            OkHttpClient::class.java.isAssignableFrom(field.type)
                        ) {
                            val oldClient = field.get(null) as? OkHttpClient ?: continue
                            if (!hasAmeliaInterceptor(oldClient)) {
                                val newClient = oldClient.newBuilder()
                                    .addInterceptor(AmeliaInterceptor(context))
                                    .build()
                                tryUnsetFinal(field)
                                field.set(null, newClient)
                                patched++
                            }
                        }

                        // 2) Patch Retrofit.callFactory (immutable) to client with our interceptor
                        if (Modifier.isStatic(field.modifiers) &&
                            Retrofit::class.java.isAssignableFrom(field.type)
                        ) {
                            val retrofit = field.get(null) as? Retrofit ?: continue
                            val cf = retrofit.callFactory()
                            val client = (cf as? OkHttpClient) ?: continue
                            if (!hasAmeliaInterceptor(client)) {
                                val newClient = client.newBuilder()
                                    .addInterceptor(AmeliaInterceptor(context))
                                    .build()
                                // Swap Retrofit.private field callFactory via reflection
                                runCatching {
                                    val callFactoryField =
                                        Retrofit::class.java.getDeclaredField("callFactory")
                                    callFactoryField.isAccessible = true
                                    tryUnsetFinal(callFactoryField)
                                    callFactoryField.set(retrofit, newClient)
                                    patched++
                                }.onFailure {
                                    Log.w(TAG, "Failed to patch Retrofit.callFactory", it)
                                }
                            }
                        }
                    }
                }
            }
            Log.i(TAG, "Amelia interceptor installed. Patched targets: $patched")
        } catch (t: Throwable) {
            Log.e(TAG, "AmeliaNetworkHook.install failed", t)
        }
    }

    private fun hasAmeliaInterceptor(client: OkHttpClient): Boolean =
        client.interceptors().any { it is AmeliaInterceptor }

    private fun tryUnsetFinal(f: Field) {
        runCatching {
            val modifiersField = Field::class.java.getDeclaredField("modifiers")
            modifiersField.isAccessible = true
            modifiersField.setInt(f, f.modifiers and Modifier.FINAL.inv())
        }
    }
}
