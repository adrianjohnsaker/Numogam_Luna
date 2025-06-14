// UI State and Events
data class Phase4UiState(
    val xenomorphicState: String = "human",
    val activeXenoforms: List<String> = emptyList(),
    val currentXenoform: XenoformType? = null,
    val xenoIntensity: Float = 0f,
    val hyperstitionCount: Int = 0,
    val realHyperstitions: Int = 0,
    val activeHyperstitions: List<Hyperstition> = emptyList(),
    val realityModifications: Int = 0,
    val hyperstitionFieldStrength: Float = 0f,
    val unmappedZonesDiscovered: Int = 0,
    val discoveredZones: List<UnmappedZone> = emptyList(),
    val consciousnessLevel: Float = 0f,
    val temporalAwareness: Float = 0f
)

sealed class XenomorphicEvent {
    data class Activation(
        val formType: XenoformType,
        val intensity: Float,
        val modifications: Map<String, Float>
    ) : XenomorphicEvent()
    
    data class UnmappedZoneDiscovered(
        val zoneId: String,
        val properties: ZoneProperties,
        val effects: List<String>
    ) : XenomorphicEvent()
}

sealed class HyperstitionEvent {
    data class Created(
        val name: String,
        val narrative: String,
        val origin: String
    ) : HyperstitionEvent()
    
    data class BecameReal(
        val name: String,
        val effects: String
    ) : HyperstitionEvent()
    
    data class XenoMerged(
        val narrative: String,
        val infectionRate: Float
    ) : HyperstitionEvent()
}
