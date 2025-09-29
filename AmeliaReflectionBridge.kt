package com.amelia.bridge

import android.content.Context
import java.lang.reflect.Method

/**
 * Reflection bridge exposed to python_hook.initialize(app_context, reflection_bridge).
 * Python will call:
 *   - getKClassByName("com.package.Class")
 *   - callMethod(target, "methodName", vararg args)
 *
 * We deliberately use java.lang.Class & Method to avoid Kotlin-reflect dependency.
 */
class AmeliaReflectionBridge(private val context: Context) {

    /** Returns a java.lang.Class<?> for the given fully-qualified name, or null. */
    fun getKClassByName(className: String): Any? = runCatching {
        Class.forName(className, false, context.classLoader)
    }.getOrNull()

    /**
     * Calls an instance or static method by name (best-effort overload resolution).
     * @param target Either an instance, java.lang.Class for static methods, or a Class name (String).
     * @param methodName Method to invoke.
     * @param args Arguments (boxed).
     */
    fun callMethod(target: Any?, methodName: String, vararg args: Any?): Any? {
        if (target == null) return null

        // Resolve class & instance
        val (cls, instance) = when (target) {
            is Class<*> -> target to null          // static method on this class
            is String -> {
                val c = runCatching { Class.forName(target, false, context.classLoader) }.getOrNull()
                    ?: return null
                c to null
            }
            else -> target.javaClass to target     // instance method
        }

        // Try exact arg-type match, else name-only & arity match
        val methods = cls.methods.filter { it.name == methodName }
        if (methods.isEmpty()) return null

        // First, try lenient match by arity
        val arityMatches = methods.filter { it.parameterTypes.size == args.size }

        // Next, pick the first which is assignable
        val m: Method? = (arityMatches + methods).firstOrNull { m ->
            val pts = m.parameterTypes
            if (pts.size != args.size) return@firstOrNull false
            for (i in pts.indices) {
                val a = args[i]
                if (a == null) continue
                if (!boxedAssignableFrom(pts[i], a.javaClass)) return@firstOrNull false
            }
            true
        } ?: methods.first()

        m.isAccessible = true
        return runCatching { m.invoke(instance, *args) }.getOrNull()
    }

    private fun boxedAssignableFrom(param: Class<*>, arg: Class<*>): Boolean {
        if (param.isAssignableFrom(arg)) return true
        // Primitive handling (best-effort)
        return when (param) {
            java.lang.Integer.TYPE -> arg == java.lang.Integer::class.java
            java.lang.Long.TYPE -> arg == java.lang.Long::class.java
            java.lang.Boolean.TYPE -> arg == java.lang.Boolean::class.java
            java.lang.Float.TYPE -> arg == java.lang.Float::class.java
            java.lang.Double.TYPE -> arg == java.lang.Double::class.java
            java.lang.Short.TYPE -> arg == java.lang.Short::class.java
            java.lang.Byte.TYPE -> arg == java.lang.Byte::class.java
            java.lang.Character.TYPE -> arg == java.lang.Character::class.java
            else -> false
        }
    }
}
