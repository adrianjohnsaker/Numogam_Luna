// ChatModuleInterceptor.kt
import android.content.Context
import java.util.regex.Pattern
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

object ChatModuleInterceptor {

    private val modulePatterns = mutableMapOf<Pattern, ModuleRoute>()
    private lateinit var context: Context

    data class ModuleRoute(
        val module: String,
        val function: String,
        val argumentExtractor: (String) -> Array<Any?> = { arrayOf(it) }
    )

    fun init(ctx: Context) {
        context = ctx
        // Predefined patterns (can be loaded from JSON/DB)
        addPattern(
            pattern = Pattern.compile("(?i)calculate|math|equation"),
            route = ModuleRoute("math_utils", "calculate") { match ->
                arrayOf(match.split(" ").last().toIntOrNull() ?: 0)
            }
        )
        addPattern(
            pattern = Pattern.compile("(?i)translate|language"),
            route = ModuleRoute("language_tools", "translate_text")
        )
        // Add more patterns dynamically from Python modules
        loadDynamicPatterns()
    }

    private fun loadDynamicPatterns() {
        PhantomBridge.moduleProxies.forEach { (moduleName, proxy) ->
            proxy.functions().forEach { funcName ->
                if (funcName.startsWith("handle_")) {
                    val pattern = PhantomBridge.call<String>(
                        moduleName, 
                        "get_pattern", 
                        funcName.removePrefix("handle_")
                    ) ?: return@forEach
                    
                    addPattern(
                        pattern = Pattern.compile(pattern, Pattern.CASE_INSENSITIVE),
                        route = ModuleRoute(moduleName, funcName)
                    )
                }
            }
        }
    }

    fun addPattern(pattern: Pattern, route: ModuleRoute) {
        modulePatterns[pattern] = route
    }

    suspend fun interceptQuery(query: String): Any? = withContext(Dispatchers.IO) {
        modulePatterns.entries.firstOrNull { entry ->
            entry.key.matcher(query).find()
        }?.let { (pattern, route) ->
            val args = route.argumentExtractor(query)
            PhantomBridge.call<Any?>(route.module, route.function, *args)
        }
    }
}
