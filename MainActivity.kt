package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var py: PythonModules

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize Chaquopy-backed PythonModules
        py = PythonModules(this)

        // Example usage:
        py.memory.updateMemory("user1", "zoneA", "some info")
        val info     = py.memory.retrieveMemory("user1", "zoneA")
        val greeting = py.api.greet("Alice")
        val sum      = py.api.add(5, 7)

        // Use the results (e.g., show in UI or log)
        println("Info: $info")      
        println("Greeting: $greeting")
        println("Sum: $sum")
    }
}
