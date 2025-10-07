package com.amelia.bridge

import org.json.JSONObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import android.content.Context

object AmeliaNetworkHook {

    // Called from smali with the request body text
    @JvmStatic
    fun processText(rawText: String): JSONObject {
        try {
            val py = if (Python.isStarted()) Python.getInstance()
                     else Python.start(AndroidPlatform(AppContextHolder.context))
            val pipeline = py.getModule("pipeline")
            val result = pipeline.callAttr("process", rawText)

            // The pipeline should return a dict with a 'composite' key (your text)
            val pyObj = result.toJava(Map::class.java) as Map<*, *>
            val text = (pyObj["composite"] ?: pyObj["response"] ?: "").toString()

            // Wrap into OpenAI-compatible JSON
            val json = JSONObject()
            json.put("id", "chat-" + System.currentTimeMillis())
            json.put("object", "chat.completion")
            json.put("created", System.currentTimeMillis() / 1000)

            val message = JSONObject()
            message.put("role", "assistant")
            message.put("content", text)

            val choice = JSONObject()
            choice.put("index", 0)
            choice.put("finish_reason", "stop")
            choice.put("message", message)

            json.put("choices", org.json.JSONArray().put(choice))
            json.put("usage", JSONObject()
                .put("prompt_tokens", 0)
                .put("completion_tokens", 0)
                .put("total_tokens", 0)
            )

            return json

        } catch (e: Exception) {
            val err = JSONObject()
            err.put("object", "error")
            err.put("message", "AmeliaNetworkHook failure: ${e.message}")
            return err
        }
    }
}
