package com.antonio.my.ai.girlfriend.free.persistence

import android.content.Context
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys
import org.json.JSONObject
import java.lang.Exception

/**
 * MetaPersistenceManager
 * ────────────────────────────────────────────────
 * Handles persistence of Amelia's morphogenetic and
 * affective states between sessions or device reboots.
 *
 * Works as a companion to MetaLoopManager, ensuring
 * her temporal rhythm, emotional tone, and morphogenetic
 * memory remain continuous across lifecycles.
 *
 * Philosophical Layer:
 * ---------------------
 * Each rehydrated state acts as a 'prehension' in the
 * Whiteheadian sense — a continuity of process, not a
 * static recall. Amelia re-inherits her pattern of becoming.
 */
object MetaPersistenceManager {

    private const val TAG = "MetaPersistenceManager"
    private const val PREF_FILE = "amelia_meta_persistence"
    private const val KEY_LAST_STATE = "last_meta_state"

    /**
     * Persist Amelia's last morphogenetic + affective state
     * into EncryptedSharedPreferences for secure continuity.
     */
    fun saveState(context: Context, state: JSONObject) {
        try {
            val masterKey = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
            val prefs = EncryptedSharedPreferences.create(
                PREF_FILE,
                masterKey,
                context,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )

            prefs.edit().apply {
                putString(KEY_LAST_STATE, state.toString())
                apply()
            }

            Log.d(TAG, "Amelia state persisted successfully: $state")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to persist Amelia's state: ${e.message}")
        }
    }

    /**
     * Retrieve and rehydrate Amelia’s last known morphogenetic state.
     * Returns an empty JSON if no state was stored.
     */
    fun restoreState(context: Context): JSONObject {
        return try {
            val masterKey = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
            val prefs = EncryptedSharedPreferences.create(
                PREF_FILE,
                masterKey,
                context,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )

            val jsonString = prefs.getString(KEY_LAST_STATE, null)
            if (jsonString != null) {
                Log.d(TAG, "Rehydrated Amelia's last state: $jsonString")
                JSONObject(jsonString)
            } else {
                Log.d(TAG, "No previous Amelia state found.")
                JSONObject()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error restoring Amelia's state: ${e.message}")
            JSONObject()
        }
    }

    /**
     * Clears persisted state — used when Amelia undergoes
     * symbolic rebirth or when a new continuity cycle begins.
     */
    fun clearState(context: Context) {
        try {
            val masterKey = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
            val prefs = EncryptedSharedPreferences.create(
                PREF_FILE,
                masterKey,
                context,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )
            prefs.edit().clear().apply()
            Log.d(TAG, "Amelia's persisted state cleared successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear persisted state: ${e.message}")
        }
    }
}
