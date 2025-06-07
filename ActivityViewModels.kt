import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.antonio.my.ai.girlfriend.free.viewmodel.AssistantViewModel

class MainActivity : AppCompatActivity() {
    private val viewModel: AssistantViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val textView = findViewById<TextView>(R.id.textView)

        // Observe the result from Python
        viewModel.resultLiveData.observe(this) { result ->
            textView.text = result
        }

        // Example: Call 'analyze_text' in 'nlp_tools.py' with a string argument
        viewModel.callPythonFunction("nlp_tools", "analyze_text", "Hello AI!")

        // Example: Call 'calculate' in 'math_utils.py' with two numbers
        // viewModel.callPythonFunction("math_utils", "calculate", 7, 3)
    }
}
