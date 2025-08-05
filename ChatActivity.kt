package com.amelia.consciousness

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import okhttp3.*
import org.json.JSONObject
import java.io.IOException

class ChatActivity : AppCompatActivity() {
    private lateinit var recyclerView: RecyclerView
    private lateinit var inputBox: EditText
    private lateinit var sendBtn: Button
    private lateinit var adapter: ChatAdapter
    private val client = OkHttpClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_chat)

        recyclerView = findViewById(R.id.chatRecycler)
        inputBox    = findViewById(R.id.inputBox)
        sendBtn     = findViewById(R.id.sendBtn)

        adapter = ChatAdapter()
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        if (!Python.isStarted())
            Python.start(AndroidPlatform(this))

        startFlask()

        sendBtn.setOnClickListener {
            val text = inputBox.text.toString().trim()
            if (text.isNotEmpty()) {
                adapter.addMessage(text, isUser = true)
                inputBox.text.clear()
                sendToFlask(text)
            }
        }
    }

    private fun startFlask() {
        Thread {
            try {
                val py = Python.getInstance()
                val app = py.getModule("app").get("app")
                app.callAttr("run", mapOf("host" to "0.0.0.0", "port" to 5000))
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.apply { isDaemon = true }.start()
    }

    private fun sendToFlask(message: String) {
        val body = JSONObject().put("message", message).toString()
            .toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url("http://127.0.0.1:5000/chat")
            .post(body)
            .build()

        client.newCall(request).enqueue(object: Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    adapter.addMessage("[Error: ${e.message}]", isUser = false)
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val reply = JSONObject(response.body!!.string()).optString("reply")
                runOnUiThread {
                    adapter.addMessage(reply, isUser = false)
                }
            }
        })
    }
}
