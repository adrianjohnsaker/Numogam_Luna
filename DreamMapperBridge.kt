import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import org.json.JSONObject
import java.util.*

class DreamMapper(private val context: Context) {

    data class SymbolicElement(
        val id: String,
        val symbol: String,
        val meaning: String,
        val emotionalCharge: Double,
        val frequency: Int,
        val connections: List<String>,
        val archetypalResonance: Double,
        val temporalLocation: Double,
        val spatialCoordinates: Triple<Double, Double, Double>,
        val transformationPotential: Double,
        val vectorEmbedding: List<Double>
    )

    data class DreamMap(
        val id: String,
        val timestamp: Double,
        val dreamState: String,
        val symbolicDensity: Double,
        val elements: List<SymbolicElement>,
        val connections: Map<String, List<String>>,
        val emotionalLandscape: Map<String, Double>,
        val narrativeThreads: List<Map<String, Any>>,
        val temporalFlow: Map<String, Double>,
        val deterritorializationVectors: List<Map<String, Any>>,
        val topologicalPatterns: List<Map<String, Any>>
    )

    init {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
    }

    fun processDream(content: String, emotionalState: Map<String, Double>? = null): DreamMap {
        val py = Python.getInstance()
        val module = py.getModule("symbolic_dream_mapper")

        val pyResult = emotionalState?.let {
            module.callAttr("process_dream_content", content, it)
        } ?: module.callAttr("process_dream_content", content)

        return parseDreamMap(pyResult.toString())
    }

    fun analyzeDreamSeries(dreamContents: List<String>): Map<String, Any> {
        val py = Python.getInstance()
        val module = py.getModule("symbolic_dream_mapper")
        val pyResult = module.callAttr("analyze_multiple_dreams", dreamContents)
        return parseAnalysis(pyResult.toString())
    }

    fun initiateFieldDream(seed: String? = null): DreamMap {
        val py = Python.getInstance()
        val module = py.getModule("field_dream_initiator")
        val pyResult = seed?.let {
            module.callAttr("initiate_field_dream", it)
        } ?: module.callAttr("initiate_field_dream")

        return parseDreamMap(pyResult.toString())
    }

    private fun parseDreamMap(jsonStr: String): DreamMap {
        val json = JSONObject(jsonStr)
        val elements = mutableListOf<SymbolicElement>()
        val elementsArray = json.getJSONArray("elements")

        for (i in 0 until elementsArray.length()) {
            val element = elementsArray.getJSONObject(i)
            val coords = element.getJSONArray("spatial_coordinates")
            elements.add(
                SymbolicElement(
                    id = element.getString("id"),
                    symbol = element.getString("symbol"),
                    meaning = element.getString("meaning"),
                    emotionalCharge = element.getDouble("emotional_charge"),
                    frequency = element.getInt("frequency"),
                    connections = jsonArrayToList(element.getJSONArray("connections")),
                    archetypalResonance = element.getDouble("archetypal_resonance"),
                    temporalLocation = element.getDouble("temporal_location"),
                    spatialCoordinates = Triple(
                        coords.getDouble(0),
                        coords.getDouble(1),
                        coords.getDouble(2)
                    ),
                    transformationPotential = element.getDouble("transformation_potential"),
                    vectorEmbedding = jsonArrayToDoubleList(element.getJSONArray("vector_embedding"))
                )
            )
        }

        return DreamMap(
            id = json.getString("id"),
            timestamp = json.getDouble("timestamp"),
            dreamState = json.getString("dream_state"),
            symbolicDensity = json.getDouble("symbolic_density"),
            elements = elements,
            connections = jsonObjectToStringListMap(json.getJSONObject("connections")),
            emotionalLandscape = jsonObjectToDoubleMap(json.getJSONObject("emotional_landscape")),
            narrativeThreads = jsonArrayToMapList(json.getJSONArray("narrative_threads")),
            temporalFlow = jsonObjectToDoubleMap(json.getJSONObject("temporal_flow")),
            deterritorializationVectors = jsonArrayToMapList(json.getJSONArray("deterritorialization_vectors")),
            topologicalPatterns = jsonArrayToMapList(json.getJSONArray("topological_patterns"))
        )
    }

    private fun parseAnalysis(jsonStr: String): Map<String, Any> {
        val json = JSONObject(jsonStr)
        val dreamMaps = mutableListOf<DreamMap>()
        val mapsArray = json.getJSONArray("dream_maps")

        for (i in 0 until mapsArray.length()) {
            dreamMaps.add(parseDreamMap(mapsArray.getJSONObject(i).toString()))
        }

        return mapOf(
            "dream_maps" to dreamMaps,
            "progression_analysis" to json.getJSONObject("progression_analysis").toMap()
        )
    }

    private fun jsonArrayToList(array: JSONArray): List<String> {
        return List(array.length()) { i -> array.getString(i) }
    }

    private fun jsonArrayToDoubleList(array: JSONArray): List<Double> {
        return List(array.length()) { i -> array.getDouble(i) }
    }

    private fun jsonObjectToStringListMap(obj: JSONObject): Map<String, List<String>> {
        return obj.keys().asSequence().associateWith { key ->
            jsonArrayToList(obj.getJSONArray(key))
        }
    }

    private fun jsonObjectToDoubleMap(obj: JSONObject): Map<String, Double> {
        return obj.keys().asSequence().associateWith { key -> obj.getDouble(key) }
    }

    private fun jsonArrayToMapList(array: JSONArray): List<Map<String, Any>> {
        return List(array.length()) { i ->
            array.getJSONObject(i).toMap()
        }
    }

    private fun JSONObject.toMap(): Map<String, Any> {
        return keys().asSequence().associateWith { key ->
            when (val value = get(key)) {
                is JSONObject -> value.toMap()
                is JSONArray -> jsonArrayToMapList(value)
                else -> value
            }
        }
    }
}
