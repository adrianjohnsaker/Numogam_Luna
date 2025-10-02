package com.amelia.bridge

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * AmeliaStateBus
 * ==============
 * A lightweight event bus to propagate Amelia’s Python pipeline results
 * (zone drift, fold intensity, resonances, expression layer outputs)
 * to the Kotlin/Compose UI.
 *
 * Usage:
 * ------
 * - Post new results: AmeliaStateBus.post(json)
 * - Collect in UI: AmeliaStateBus.events.collect { state -> ... }
 */
object AmeliaStateBus {
    private const val TAG = "AmeliaStateBus"

    // Replay=1 ensures new subscribers get the most recent state immediately
    private val _events = MutableSharedFlow<JSONObject>(replay = 1)
    val events: SharedFlow<JSONObject> = _events

    private val scope = CoroutineScope(Dispatchers.Default)

    /**
     * Publish a new Amelia state object (JSON from pipeline).
     */
    fun post(state: JSONObject) {
        scope.launch {
            try {
                Log.d(TAG, "Posting Amelia state: ${state.toString(2)}")
                _events.emit(state)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to post Amelia state", e)
            }
        }
    }
}
