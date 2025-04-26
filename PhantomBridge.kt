// PhantomBridge.kt
import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlin.reflect.KClass
import kotlin.reflect.full.createInstance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

object PhantomBridge {

    private lateinit var context: Context
    private val moduleProxies = mutableMapOf<String, ModuleProxy>()
    private val functionCache = mutableMapOf<String, com.chaquo.python.PyObject>()
    private var initialized = false

    // Combined initialization with scanner
    suspend fun init(ctx: Context) = withContext(Dispatchers.IO) {
        if (initialized) return@withContext
        context = ctx.applicationContext
        if (!Python.isStarted()) Python.start(AndroidPlatform(context))
        
        // Load main PyBridge scanner
        val scanner = Python.getInstance().getModule("pybridge.scanner")
        val moduleNames = scanner.callAttr("scan_all_modules").toJava(List::class.java) as List<String>
        
        // Build proxies in parallel
        moduleNames.chunked(5).forEach { chunk ->
            chunk.forEach { moduleName ->
                try {
                    val module = Python.getInstance().getModule(moduleName)
                    moduleProxies[moduleName] = ModuleProxy(module)
                } catch (e: Exception) {
                    // Silent skip
                }
            }
        }
        initialized = true
    }

    // Dynamic interface generator
    inline fun <reified T : Any> createInterface(): T = object : T {
        init {
            if (!initialized) throw IllegalStateException("PhantomBridge not initialized")
        }
        
        override fun equals(other: Any?): Boolean = other is T
        override fun hashCode(): Int = 0
        override fun toString(): String = "PythonInterface<${T::class.simpleName}>"
    }.apply {
        T::class.members.filter { it.isAbstract }.forEach { member ->
            member as? KFunction<*> ?: return@forEach
            val annotation = member.findAnnotation<PythonFunction>() ?: return@forEach
            val cacheKey = "${annotation.module}.${annotation.function}"
            
            functionCache.getOrPut(cacheKey) {
                Python.getInstance().getModule(annotation.module)[annotation.function]
            }
        }
    } as T

    // Performance-optimized call
    @Suppress("UNCHECKED_CAST")
    fun <T> call(module: String, function: String, vararg args: Any?): T? {
        val cacheKey = "$module.$function"
        return try {
            (functionCache[cacheKey] ?: Python.getInstance()
                .getModule(module)
                .get(function)
                .also { functionCache[cacheKey] = it })
                .call(*args)
                .toJava(T::class.java) as T
        } catch (e: Exception) {
            null
        }
    }

    inner class ModuleProxy(/* ... existing implementation ... */) {
        // ... (keep previous ModuleProxy implementation)
    }
}

// Annotation for interface methods
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class PythonFunction(val module: String, val function: String)

// DSL for direct calls
inline fun <reified T> python(module: String, func: String, vararg args: Any?): T? {
    return PhantomBridge.call(module, func, *args)
}
