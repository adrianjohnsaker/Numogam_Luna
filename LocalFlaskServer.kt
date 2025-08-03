package com.amelia.consciousness

import android.content.Context
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

object LocalFlaskServer {
    private const val HOST = "0.0.0.0"
    private const val PORT = 5000

    fun start(context: Context) {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }

        Thread {
            try {
                val py = Python.getInstance()
                val flaskApp = py.getModule("app").get("app")
                flaskApp.callAttr("run",
                    mapOf(
                        "host" to HOST,
                        "port" to PORT,
                        "debug" to false,
                        "threaded" to true
                    )
                )
                Log.d("LocalFlaskServer", "Flask server started on port $PORT")
            } catch (e: Exception) {
                Log.e("LocalFlaskServer", "Failed to start Flask server", e)
            }
        }.apply { isDaemon = true }.start()
    }
}
