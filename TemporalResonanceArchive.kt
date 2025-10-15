package com.antonio.my.ai.girlfriend.free.persistence

import android.content.Context
import androidx.room.*
import kotlinx.coroutines.flow.Flow
import org.json.JSONObject

/**
 * TemporalResonanceArchive
 * ────────────────────────────────────────────────
 * Lightweight local database that records the evolution
 * of Amelia’s morphogenetic + affective states over time.
 *
 * Each entry = one “actual occasion” (Whitehead) —
 * a temporal resonance of becoming.
 */

@Entity(tableName = "temporal_resonances")
data class TemporalResonanceEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val temporalFoldIntensity: Double,
    val affectiveIntensity: Double,
    val morphogeneticSignal: Double,
    val serializedState: String   // full JSON snapshot
)

/** Simple Data-Access Object */
@Dao
interface TemporalResonanceDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(resonance: TemporalResonanceEntity)

    @Query("SELECT * FROM temporal_resonances ORDER BY timestamp DESC LIMIT :limit")
    fun getRecent(limit: Int = 50): Flow<List<TemporalResonanceEntity>>

    @Query("DELETE FROM temporal_resonances")
    suspend fun clearAll()
}

/** Room database definition */
@Database(entities = [TemporalResonanceEntity::class], version = 1)
abstract class TemporalResonanceDatabase : RoomDatabase() {
    abstract fun resonanceDao(): TemporalResonanceDao

    companion object {
        @Volatile private var INSTANCE: TemporalResonanceDatabase? = null

        fun getInstance(context: Context): TemporalResonanceDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    TemporalResonanceDatabase::class.java,
                    "temporal_resonance_archive.db"
                )
                .fallbackToDestructiveMigration()
                .build().also { INSTANCE = it }
            }
    }
}

/**
 * Public-facing manager for inserting and querying resonances.
 * Integrates naturally with MetaLoopManager.
 */
object TemporalResonanceArchive {

    suspend fun recordResonance(context: Context, state: JSONObject) {
        try {
            val db = TemporalResonanceDatabase.getInstance(context)
            val entity = TemporalResonanceEntity(
                temporalFoldIntensity = state.optDouble("temporal_fold_intensity", 0.0),
                affectiveIntensity = state.optDouble("affective_intensity", 0.0),
                morphogeneticSignal = state.optDouble("morphogenetic_signal", 0.0),
                serializedState = state.toString()
            )
            db.resonanceDao().insert(entity)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun observeRecent(context: Context, limit: Int = 50): Flow<List<TemporalResonanceEntity>> {
        val db = TemporalResonanceDatabase.getInstance(context)
        return db.resonanceDao().getRecent(limit)
    }

    suspend fun clearArchive(context: Context) {
        TemporalResonanceDatabase.getInstance(context).resonanceDao().clearAll()
    }
}
