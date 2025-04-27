package com.antonio.my.ai.girlfriend.free.util

import android.content.Context

object AppContextHolder {
    private lateinit var appContext: Context

    fun init(context: Context) {
        if (!::appContext.isInitialized) {
            appContext = context.applicationContext
        }
    }

    fun get(): Context {
        if (!::appContext.isInitialized) {
            throw IllegalStateException("AppContextHolder is not initialized. Call init(context) in your main activity.")
        }
        return appContext
    }
}
