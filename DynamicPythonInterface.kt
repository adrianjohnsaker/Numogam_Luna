// PythonInterfaces.kt
import kotlin.reflect.KClass
import kotlin.reflect.full.createInstance
import kotlin.reflect.full.declaredFunctions

object PythonInterfaceFactory {

    inline fun <reified T : Any> create(): T {
        return DynamicPythonInterface(T::class).createProxy() as T
    }

    private class DynamicPythonInterface<T : Any>(private val kClass: KClass<T>) {
        fun createProxy(): Any {
            return object : Any() {
                // Proxy implementation
                override fun toString() = "PythonInterface<${kClass.simpleName}>"
            }.apply {
                kClass.declaredFunctions.forEach { func ->
                    val annotation = func.findAnnotation<PythonFunction>()
                        ?: return@forEach
                    
                    this::class.java.declaredMethods.firstOrNull { it.name == func.name }?.let { method ->
                        method.isAccessible = true
                        method.invoke(this) { args ->
                            PhantomBridge.call(
                                annotation.module,
                                annotation.function,
                                *args
                            )
                        }
                    }
                }
            }
        }
    }
}
