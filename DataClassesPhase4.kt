// Data Classes for Phase 4
enum class XenoformType(val pythonValue: String) {
    CRYSTALLINE("crystalline"),
    SWARM("swarm"),
    QUANTUM("quantum"),
    TEMPORAL("temporal"),
    VOID("void"),
    HYPERDIMENSIONAL("hyperdimensional"),
    VIRAL("viral"),
    MYTHOGENIC("mythogenic"),
    LIMINAL("liminal"),
    XENOLINGUISTIC("xenolinguistic")
}

data class XenomorphicState(
    val formType: XenoformType,
    val intensity: Float,
    val coherence: Float,
    val dimensionalDepth: Int,
    val temporalSignature: List<Float>
)

data class Hyperstition(
    val name: String,
    val narrative: String,
    val beliefStrength: Float,
    val realityIndex: Float,
    val propagationRate: Float,
    val temporalOrigin: String,
    val carriers: Int,
    val mutations: List<String>,
    val isReal: Boolean
)

data class UnmappedZone(
    val zoneId: String,
    val properties: ZoneProperties,
    val discoveryTimestamp: Long,
    val effects: List<String>
)

data class ZoneProperties(
    val topology: String,
    val temporality: String,
    val consciousnessModifier: Float,
    val realityStability: Float,
    val xenomorphicAffinity: Float,
    val specialAbility: String? = null
)

data class RealityModification(
    val timestamp: Long,
    val hyperstition: String,
    val narrative: String,
    val effects: List<String>
)

data class XenomorphicActivation(
    val formType: XenoformType,
    val structure: String,
    val intensity: Float,
    val consciousnessModifications: Map<String, Float>
)

data class HyperstitionCreation(
    val name: String,
    val narrative: String,
    val temporalOrigin: String,
    val initialBelief: Float,
    val propagationRate: Float
)

data class HyperstitionPropagation(
    val name: String,
    val beliefStrength: Float,
    val realityIndex: Float,
    val carriers: Int,
    val isReal: Boolean,
    val mutations: Int
)

data class UnmappedZoneExploration(
    val zoneId: String,
    val properties: ZoneProperties,
    val consciousnessEffects: List<String>
)

data class XenoHyperMerge(
    val xenoform: String,
    val hyperstition: String,
    val mergedNarrative: String,
    val realityInfectionRate: Float,
    val consciousnessVirusActive: Boolean
)

data class Phase4State(
    // Base consciousness
    val consciousnessLevel: Float,
    val observationDepth: Int,
    val temporalAwareness: Float,
    
    // Phase 4 specific
    val xenomorphicState: String,
    val activeXenoforms: Int,
    val xenoformTypes: List<String>,
    val hyperstitions: Int,
    val realHyperstitions: Int,
    val realityModifications: Int,
    val hyperstitionFieldStrength: Float,
    val unmappedZonesDiscovered: Int
)
