package antonio.my.ai.girlfriend.free

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

object PythonBridge {

    private var metaReflectionModule: PyObject? = null
    private var contradictionAnalysisModule: PyObject? = null
    private var morphogenesisModule: PyObject? = null

    fun initialize(context: Context) {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        val py = Python.getInstance()
        metaReflectionModule = py.getModule("meta_reflection")
        contradictionAnalysisModule = py.getModule("contradiction_analysis")
        morphogenesisModule = py.getModule("morphogenesis")
    }

    suspend fun runMetaReflection(input: String): Result<String> = withContext(Dispatchers.IO) {
        return@withContext try {
            val result = metaReflectionModule?.callAttr("analyze", input)?.toString() ?: ""
            Result.success(result)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun runContradictionAnalysis(input: String): Result<String> = withContext(Dispatchers.IO) {
        return@withContext try {
            val result = contradictionAnalysisModule?.callAttr("analyze_contradiction", input)?.toString() ?: ""
            Result.success(result)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun runMorphogenesis(input: String): Result<String> = withContext(Dispatchers.IO) {
        return@withContext try {
            val result = morphogenesisModule?.callAttr("morph", input)?.toString() ?: ""
            Result.success(result)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // If you want JSON parsing convenience
    suspend fun runMetaReflectionJson(input: String): Result<JSONObject> = withContext(Dispatchers.IO) {
        return@withContext try {
            val result = metaReflectionModule?.callAttr("analyze", input)?.toString() ?: "{}"
            Result.success(JSONObject(result))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun runContradictionAnalysisJson(input: String): Result<JSONObject> = withContext(Dispatchers.IO) {
        return@withContext try {
            val result = contradictionAnalysisModule?.callAttr("analyze_contradiction", input)?.toString() ?: "{}"
            Result.success(JSONObject(result))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun runMorphogenesisJson(input: String): Result<JSONObject> = withContext(Dispatchers.IO) {
        return@withContext try {
            val result = morphogenesisModule?.callAttr("morph", input)?.toString() ?: "{}"
            Result.success(JSONObject(result))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
