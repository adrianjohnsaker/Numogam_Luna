// ChatResponseWrapper.kt
import org.json.JSONObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

object ChatResponseWrapper {

    private val forcedResponses = mutableMapOf<String, Any?>()

    fun wrapResponse(original: String, scope: CoroutineScope): String {
        return forcedResponses[original]?.let { enhanced ->
            JSONObject().apply {
                put("original", original)
                put("enhanced", enhanced)
                put("source", "python_module")
            }.toString()
        } ?: original
    }

    fun preloadResponses(queries: List<String>, scope: CoroutineScope) {
        queries.forEach { query ->
            scope.launch {
                forcedResponses[query] = ChatModuleInterceptor.interceptQuery(query)
            }
        }
    }
}
