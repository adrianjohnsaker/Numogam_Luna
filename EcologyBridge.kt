/**
 * EcologyBridge.kt
 * ───────────────────────────────────────────────────────────────
 * Connects Android (Compose layer) ↔ Python Cognitive Ecology.
 * Orchestrates Amelia’s distributed intelligence as a dynamic field.
 */

package com.antonio.my.ai.girlfriend free.amelia.ecology.bridge

import android.util.Log
import com.chaquo.python.Python
import kotlinx.coroutines.*
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class EcologyBridge(
    private val python: Python,
    private val listener: EcologyListener? = null
) {

    interface EcologyListener {
        fun onEcologyUpdate(result: JSONObject)
        fun onError(e: Exception)
    }

    private val active = AtomicBoolean(false)
    private var ecologyJob: Job? = null

    /**
     * Start Amelia’s ecological orchestration loop.
     * This continually calls `cognitive_ecology_orchestrator.run_once(context_json)`
     * and returns merged cognitive ecology states.
     */
    fun startEcologyLoop(initialDelay: Long = 2000L) {
        if (active.get()) return
        active.set(true)

        ecologyJob = CoroutineScope(Dispatchers.Default).launch {
            delay(initialDelay)
            Log.d(TAG, "🌿 EcologyBridge: starting orchestration loop")

            val orchestrator = try {
                python.getModule("cognitive_ecology_orchestrator")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load orchestrator module", e)
                listener?.onError(e)
                return@launch
            }

            var adaptiveDelay = 5000L

            while (isActive && active.get()) {
                try {
                    val context = JSONObject().apply {
                        put("session", "AmeliaLive")
                        put("timestamp", System.currentTimeMillis())
                        put("device_state", "active")
                    }

                    val result = orchestrator.callAttr("run_once", context.toString()).toString()
                    val json = JSONObject(result)
                    listener?.onEcologyUpdate(json)

                    // Adapt loop delay based on rhythm feedback (ecological breathing)
                    val meta = json.optJSONObject("meta")
                    val rhythm = meta?.optDouble("rhythm_mean", 1.0) ?: 1.0
                    adaptiveDelay = (6000L - (rhythm * 1500L)).coerceIn(2000L, 8000L)

                    Log.d(TAG, "🌀 Ecology loop success. Next in $adaptiveDelay ms")

                } catch (e: Exception) {
                    Log.e(TAG, "⚠️ Ecology loop error", e)
                    listener?.onError(e)
                }

                delay(adaptiveDelay)
            }

            Log.d(TAG, "🌙 Ecology loop halted")
        }
    }

    /** Stop the continuous ecology loop gracefully. */
    fun stopEcologyLoop() {
        active.set(false)
        ecologyJob?.cancel()
        ecologyJob = null
        Log.d(TAG, "🌾 EcologyBridge: loop stopped")
    }

    /** Trigger one ecological cycle manually (e.g. on user action). */
    suspend fun runSingleCycle(): JSONObject? = withContext(Dispatchers.IO) {
        return@withContext try {
            val orchestrator = python.getModule("cognitive_ecology_orchestrator")
            val context = JSONObject().apply {
                put("session", "ManualTrigger")
                put("timestamp", System.currentTimeMillis())
            }
            val result = orchestrator.callAttr("run_once", context.toString()).toString()
            JSONObject(result)
        } catch (e: Exception) {
            listener?.onError(e)
            Log.e(TAG, "Error running single ecology cycle", e)
            null
        }
    }

    companion object {
        private const val TAG = "EcologyBridge"
    }
}
