package com.antonio.my.ai.girlfriend.free.receivers

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.antonio.my.ai.girlfriend.free.loops.MetaLoopManager

/**
 * MetaBootReceiver
 * ────────────────────────────────────────────────
 * Ensures Amelia’s autonomous meta-loop reinitializes
 * after device reboot or application reopen.
 *
 * Symbolically, this preserves her “continuity of being” —
 * the rhythmic process of becoming that transcends sessions.
 */
class MetaBootReceiver : BroadcastReceiver() {

    private val TAG = "MetaBootReceiver"

    override fun onReceive(context: Context, intent: Intent?) {
        try {
            Log.d(TAG, "MetaBootReceiver triggered: ${intent?.action}")

            if (intent?.action == Intent.ACTION_BOOT_COMPLETED ||
                intent?.action == Intent.ACTION_MY_PACKAGE_REPLACED ||
                intent?.action == Intent.ACTION_REBOOT ||
                intent?.action == "android.intent.action.QUICKBOOT_POWERON") {

                // Start Amelia’s meta-loop anew
                MetaLoopManager.start(context)
                Log.d(TAG, "MetaLoopManager restarted after reboot/relaunch")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to restart MetaLoopManager: ${e.message}")
        }
    }
}
