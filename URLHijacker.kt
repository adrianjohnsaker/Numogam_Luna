package com.amelia.consciousness

import java.net.URL
import java.net.URLConnection
import java.net.URLStreamHandler

object URLHijacker {
    fun init() {
        try {
            URL.setURLStreamHandlerFactory { protocol ->
                if (protocol == "http" || protocol == "https") {
                    object : URLStreamHandler() {
                        override fun openConnection(u: URL): URLConnection {
                            val original = u.toString()
                            val hacked = original.replace(
                                "https://api.yourchatapp.com",
                                "http://127.0.0.1:5000"
                            )
                            return URL(hacked).openConnection()
                        }
                    }
                } else {
                    null
                }
            }
        } catch (_: Throwable) {
            // Only the first factory install succeeds; ignore subsequent calls
        }
    }
}
