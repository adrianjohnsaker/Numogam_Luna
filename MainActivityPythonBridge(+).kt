```kotlin
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import androidx.lifecycle.lifecycleScope
import com.example.app.pythonbridge.PythonBridge
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var pythonBridge: PythonBridge
    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize Python bridge
        pythonBridge = PythonBridge.getInstance(applicationContext)
        
        // Initialize Python environment
        lifecycleScope.launch {
            if (pythonBridge.initialize()) {
                Log.d(TAG, "Python environment initialized successfully")
                
                // Get router status
                val status = pythonBridge.getRouterStatus()
                Log.d(TAG, "Router status: $status")
                
                // Process a test query
                val result = pythonBridge.processQuery("drift:ocean waves")
                Log.d(TAG, "Query result: $result")
            } else {
                Log.e(TAG, "Failed to initialize Python environment")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pythonBridge.cleanup()
    }
}
```
