package com.antonio.my.ai.girlfriend.free

import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class AmeliaModuleInterceptor private constructor() {

    companion object {
        private var instance: AmeliaModuleInterceptor? = null

        fun getInstance(): AmeliaModuleInterceptor {
            if (instance == null) {
                instance = AmeliaModuleInterceptor()
            }
            return instance!!
        }
    }

    fun intercept(message: String): String {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(MyApplication.context))
        }
        val py = Python.getInstance()
        val module = py.getModule("module_manager")
        val result = module.callAttr("process_message", message)
        return result.toString()
    }
}
