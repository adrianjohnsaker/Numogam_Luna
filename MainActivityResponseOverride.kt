// MainActivity.kt
package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import com.antonio.my.ai.girlfriend.free.autonomous.EngineModule
import com.antonio.my.ai.girlfriend.free.autonomous.LocalAutonomousEngine
import com.antonio.my.ai.girlfriend.free.autonomous.ResponseOverrideSystem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.Locale

class MainActivity : ComponentActivity() {

    private lateinit var engine: LocalAutonomousEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Get singleton engine and start it
        engine = EngineModule.engine
        engine.start()

        // Wire engine into override system
        ResponseOverrideSystem.engine = engine

        val liveSnapshot = findViewById<TextView>(R.id.liveSnapshot)
        val input = findViewById<EditText>(R.id.input)
        val sendBtn = findViewById<Button>(R.id.sendBtn)
        val response = findViewById<TextView>(R.id.response)
        val sourceTag = findViewById<TextView>(R.id.sourceTag)
        response.movementMethod = ScrollingMovementMethod()

        // Periodically update a live snapshot view
        lifecycleScope.launch(Dispatchers.Main) {
            while (isActive) {
                val s = engine.snapshot()
                val sText = buildString {
                    append("cycles=")
                    append(String.format(Locale.US, "%,d", s.cycleCount))
                    append(" | uptime=")
                    append(String.format(Locale.US, "%d s", s.uptimeSeconds))
                    append(" | eps=")
                    append("%.4f".format(Locale.US, s.epsilonExploration))
                    append(" | cm=")
                    append("%.3f".format(Locale.US, s.creativeMomentum))
                    append(" | conn=")
                    append("%.3f".format(Locale.US, s.connectionDensity))
                    append(" | decisions=")
                    append(s.decisionHistoryLength)
                }
                liveSnapshot.text = "Snapshot: $sText"
                delay(300)
            }
        }

        // Handle send
        sendBtn.setOnClickListener {
            val user = input.text?.toString().orEmpty()
            if (user.isBlank()) return@setOnClickListener

            // Try override first
            val overridden = ResponseOverrideSystem.tryHandle(user)
            if (overridden != null) {
                response.text = overridden.text
                sourceTag.text = "[source=${overridden.source}, no_style=${overridden.noStyle}]"
                return@setOnClickListener
            }

            // Fallback to your normal LLM path (placeholder here)
            val llmText = "(LLM) I’m here to help. Could you clarify your question?"
            response.text = llmText
            sourceTag.text = "[source=llm]"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Optional: stop engine if you want to fully tear down when Activity is destroyed.
        // For app-wide singleton, you may prefer keeping it running.
        // EngineModule.engine.stop()
    }
}
