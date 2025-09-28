// EngineModule.kt
package com.antonio.my.ai.girlfriend.free.autonomous

object EngineModule {
    // App-wide singleton
    val engine: LocalAutonomousEngine by lazy { LocalAutonomousEngine() }
}
