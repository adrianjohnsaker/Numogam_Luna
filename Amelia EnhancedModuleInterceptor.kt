// AmeliaModuleInterceptor.kt
import android.content.Context
import com.chaquo.python.PyObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.regex.Pattern

object ChatModuleInterceptor {

    private lateinit var context: Context
    private val staticPatterns = mutableMapOf<Pattern, ModuleRoute>()
    private val dynamicPatterns = mutableMapOf<Pattern, ModuleRoute>()
    private var patternsLoaded = false

    data class ModuleRoute(
        val module: String,
        val function: String,
        val argumentExtractor: (String) -> Array<Any?> = { arrayOf(it) }
    )

    suspend fun init(ctx: Context) = withContext(Dispatchers.IO) {
        context = ctx
        loadStaticPatterns()
        loadDynamicPatterns()
        patternsLoaded = true
    }

    private fun loadStaticPatterns() {
        // Predefined critical patterns
        addStaticPattern(
            pattern = Pattern.compile("(?i)calc|compute|solve"),
            route = ModuleRoute("math_utils", "calculate") { query ->
                extractNumbers(query)
            }
        )
        
        addStaticPattern(
            pattern = Pattern.compile("(?i)translate|convert language"),
            route = ModuleRoute("language", "translate_text")
        )
    }

    private suspend fun loadDynamicPatterns() = withContext(Dispatchers.IO) {
        try {
            val pyPatterns: List<Map<String, String>>? = PhantomBridge.call(
                "pybridge.scanner", 
                "get_all_patterns"
            )
            
            pyPatterns?.forEach { patternInfo ->
                addDynamicPattern(
                    pattern = Pattern.compile(patternInfo["pattern"]!!, Pattern.CASE_INSENSITIVE),
                    route = ModuleRoute(
                        module = patternInfo["module"]!!,
                        function = patternInfo["function"]!!,
                        argumentExtractor = { arrayOf(it) } // Default extractor
                    )
                )
            }
        } catch (e: Exception) {
            // Silently continue if Python scanner fails
        }
    }

    fun addStaticPattern(pattern: Pattern, route: ModuleRoute) {
        staticPatterns[pattern] = route
    }

    fun addDynamicPattern(pattern: Pattern, route: ModuleRoute) {
        dynamicPatterns[pattern] = route
    }

    suspend fun interceptQuery(query: String): Any? = withContext(Dispatchers.IO) {
        if (!patternsLoaded) return@withContext null
        
        // Check static patterns first (higher priority)
        findMatchingRoute(query, staticPatterns)?.let { route ->
            return@withContext executeRoute(route, query)
        }
        
        // Fall back to dynamic patterns
        findMatchingRoute(query, dynamicPatterns)?.let { route ->
            return@withContext executeRoute(route, query)
        }
        
        null
    }

    private fun findMatchingRoute(
        query: String,
        patternMap: Map<Pattern, ModuleRoute>
    ): ModuleRoute? {
        return patternMap.entries.firstOrNull { entry ->
            entry.key.matcher(query).find()
        }?.value
    }

    private fun executeRoute(route: ModuleRoute, query: String): Any? {
        return try {
            val args = route.argumentExtractor(query)
            PhantomBridge.call<Any?>(route.module, route.function, *args)
        } catch (e: Exception) {
            null
        }
    }

    private fun extractNumbers(query: String): Array<Any?> {
        val numbers = query.split(" ")
            .mapNotNull { it.toIntOrNull() }
            .takeIf { it.isNotEmpty() }
            ?: listOf(0)
        return arrayOf(numbers)
    }
}
