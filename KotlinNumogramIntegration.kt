import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class NumogramBridge(private val context: Context) {

    init {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
    }

    fun getContagionRecord(recordId: String): String {
        return try {
            val py = Python.getInstance()
            val module = py = python.getModule("numogram_interface")
            val result = module.callAttr("get_contagion_record", recordId)
            
            if (result.containsKey("error")) {
                "[SYS_ERR: ${result["error"]}]"
            } else {
                formatOutput(
                    id = result["contagion_id"].toString(),
                    type = result["contagion_type"].toString(),
                    severity = result["severity"].toFloat(),
                    spread = result["spread_factor"].toFloat(),
                    vector = result["activation_vector"].toString()
                )
            }
        } catch (e: Exception) {
            "[SYS_ERR: ${e.message}]"
        }
    }

    private fun formatOutput(
        id: String, 
        type: String, 
        severity: Float, 
        spread: Float, 
        vector: String
    ) = """
        [SYS_MEM: CONTAGION_RECORD_ACCESS]
        Contagion ID: $id
        Type: ${type}
        Severity: ${"%.1f".format(severity)}
        Spread: ${"%.2f".format(spread)}
        Activation: $vector
        [RETROFIT_CORE: END_STREAM]
    """.trimIndent()
}
