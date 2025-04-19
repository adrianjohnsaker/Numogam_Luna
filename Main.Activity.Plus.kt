import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        Python.start(AndroidPlatform(this))
        
        val py = Python.getInstance()
        val module = py.getModule("my_module")
        val result = module.callAttr("simple_function", "Hello from Kotlin")
        Log.d("PYTHON", result.toString())  // Output: "Python says: Hello from Kotlin"
    }
}
