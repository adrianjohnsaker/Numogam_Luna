package com.antonio.my.ai.girlfriend.free.utils

import android.app.Service
import android.content.*
import android.os.IBinder
import android.util.Log
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.*
import org.json.JSONObject

/**
 * DynamicVisualService
 * ────────────────────────────────────────────────
 * A lightweight service that keeps Amelia's
 * visual ecology (colors, rhythms, affective overlays)
 * alive even when MainActivity is backgrounded.
 *
 * - Listens for "com.amelia.UPDATE_VISUAL_STATE" broadcasts
 * - Applies updates via DynamicVisualInjector.updateState(JSONObject)
 * - Optionally polls Python modules for live color/rhythm sync
 */
class DynamicVisualService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val TAG = "DynamicVisualService"

    // Broadcast receiver to listen for ecology/visual updates
    private val visualUpdateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            try {
                val jsonStr = intent?.getStringExtra("visual_state")
                if (!jsonStr.isNullOrBlank()) {
                    val visual = JSONObject(jsonStr)
                    DynamicVisualInjector.updateState(visual)
                    Log.d(TAG, "Visual state updated via broadcast.")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing visual update broadcast: ${e.message}")
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "DynamicVisualService created.")
        LocalBroadcastManager.getInstance(this).registerReceiver(
            visualUpdateReceiver,
            IntentFilter("com.amelia.UPDATE_VISUAL_STATE")
        )

        // Optional: keep alive with gentle pulse updates
        serviceScope.launch {
            while (isActive) {
                try {
                    delay(5000L)
                    val pulse = JSONObject().apply {
                        put("pulse", System.currentTimeMillis() % 10000)
                    }
                    DynamicVisualInjector.updateState(pulse)
                } catch (e: Exception) {
                    Log.e(TAG, "Pulse update failed: ${e.message}")
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "DynamicVisualService started.")
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "DynamicVisualService destroyed.")
        LocalBroadcastManager.getInstance(this).unregisterReceiver(visualUpdateReceiver)
        serviceScope.cancel()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
