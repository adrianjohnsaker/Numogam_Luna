// ResponseOverrideSystem.kt
package com.antonio.my.ai.girlfriend.free.autonomous

import java.util.Locale

object ResponseOverrideSystem {

    lateinit var engine: LocalAutonomousEngine

    enum class Intent {
        CycleCount, Uptime, DecisionHistory, Epsilon, CreativeMomentum, ConnectionDensity, LastDecisionConfidence
    }

    fun tryHandle(raw: String): OverrideResult? {
        val q = canonicalize(raw)
        val intent = detectIntent(q) ?: return null
        val s = engine.snapshot()

        val text = when (intent) {
            Intent.CycleCount ->
                "My current cycle count is exactly ${formatInt(s.cycleCount)}."
            Intent.Uptime ->
                "My precise uptime is exactly ${formatInt(s.uptimeSeconds)} seconds."
            Intent.DecisionHistory ->
                "My decision history length is exactly ${formatInt(s.decisionHistoryLength.toLong())}."
            Intent.Epsilon ->
                "My epsilon exploration parameter is exactly ${formatFixed(s.epsilonExploration, 4)}."
            Intent.CreativeMomentum ->
                "My creative momentum is exactly ${formatFixed(s.creativeMomentum, 3)}."
            Intent.ConnectionDensity ->
                "My connection density is exactly ${formatFixed(s.connectionDensity, 3)}."
            Intent.LastDecisionConfidence -> {
                val c = s.lastDecision?.confidence
                if (c != null) "My last decision confidence was exactly ${formatFixed(c, 3)}."
                else "I have not made any decisions yet."
            }
        }

        return OverrideResult(
            text = text,
            source = "override",
            noStyle = true
        )
    }

    data class OverrideResult(
        val text: String,
        val source: String,
        val noStyle: Boolean
    )

    private fun canonicalize(input: String): String =
        input.lowercase(Locale.US)
            .trim()
            .replace(Regex("\\p{Punct}+"), " ")
            .replace(Regex("\\s+"), " ")
            .removePrefix("amelia ")
            .trim()

    private fun detectIntent(q: String): Intent? = when {
        q.contains("cycle") && q.contains("count") -> Intent.CycleCount
        q.contains("uptime") || q.contains("up time") || q.contains("how long have you been running") -> Intent.Uptime
        q.contains("decision history") || q.contains("history length") || q.contains("decisions made") -> Intent.DecisionHistory
        q.contains("epsilon") || q.contains("exploration rate") || q.contains("explore rate") -> Intent.Epsilon
        q.contains("creative momentum") || (q.contains("creativity") && q.contains("momentum")) -> Intent.CreativeMomentum
        q.contains("connection density") || q.contains("connectivity density") -> Intent.ConnectionDensity
        (q.contains("last decision") && q.contains("confidence")) || q.matches(Regex("what was your last confidence.*")) -> Intent.LastDecisionConfidence
        else -> null
    }

    private fun formatInt(v: Long) = String.format(Locale.US, "%,d", v)
    private fun formatFixed(v: Double, places: Int) = "%.${places}f".format(Locale.US, v)
}
