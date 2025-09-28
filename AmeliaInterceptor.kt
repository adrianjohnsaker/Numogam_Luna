package com.amelia.bridge

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * Intercepts /v1/chat/completions and injects Amelia’s backend output
 * from python_hook.process_input(user_input), returning an OpenAI-style JSON.
 */
class AmeliaInterceptor(ctx: Context) : Interceptor {
    private val app = ctx.applicationContext

    override fun intercept(chain: Interceptor.Chain): Response {
        val req: Request = chain.request()
        val path = req.url.encodedPath

        return try {
            if (path != null && path.endsWith("/v1/chat/completions")) {
                val bodyStr = AmeliaNetHelpers.readBodyUtf8(req.body)
                val userInput = extractUserMessage(bodyStr)

                val backendJson = callPython(userInput)
                if (!backendJson.isNullOrEmpty()) {
                    return AmeliaNetHelpers.jsonResponse(req, backendJson)
                }
            }
            chain.proceed(req)
        } catch (t: Throwable) {
            Log.w("AmeliaHook", "Interceptor error, falling back", t)
            chain.proceed(req)
        }
    }

    /** Calls python_hook.process_input(user_input) via Chaquopy. Expects OpenAI-style JSON. */
    private fun callPython(userInput: String): String? {
        return try {
            val py: Python = Python.getInstance()
            val hook: PyObject = py.getModule("python_hook")
            val out: PyObject = hook.callAttr("process_input", userInput)
            val s = out.toString()
            if (s.contains("\"choices\"")) s else null
        } catch (t: Throwable) {
            Log.w("AmeliaHook", "Python pipeline failed", t)
            null
        }
    }

    /** Pulls the latest "user" message content from an OpenAI-style request body. */
    private fun extractUserMessage(json: String): String {
        return try {
            val root = JSONObject(json)
            val msgs: JSONArray? = root.optJSONArray("messages")
            if (msgs != null) {
                for (i in msgs.length() - 1 downTo 0) {
                    val m = msgs.optJSONObject(i)
                    if (m != null && m.optString("role") == "user") {
                        return m.optString("content", "")
                    }
                }
            }
            ""
        } catch (_: Throwable) {
            ""
        }
    }
}
