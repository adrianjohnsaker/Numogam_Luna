import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        val py = Python.getInstance()
        val myModule = py.getModule("my_module")
        
        // Simple string function
        val simpleResult = myModule.callAttr("simple_function", "Hello from Kotlin")
        Log.d("PYTHON", simpleResult.toString())
        
        // Numeric function
        val sum = myModule.callAttr("add_numbers", 5, 7)
        Log.d("PYTHON", "Sum: ${sum.toInt()}")
        
        // Passing and receiving complex data
        val dataList = listOf(1, 5, 9, 12, 15, 20)
        val processingResult = myModule.callAttr("process_data", dataList.toTypedArray())
        
        // Convert PyObject to JSON for easier handling
        val resultJson = JSONObject(processingResult.toString())
        Log.d("PYTHON", "Mean: ${resultJson.getDouble("mean")}")
        Log.d("PYTHON", "Sum: ${resultJson.getDouble("sum")}")
        Log.d("PYTHON", "Max: ${resultJson.getDouble("max")}")
    }
}
