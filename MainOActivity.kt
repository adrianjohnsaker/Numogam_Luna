package com.antonio.my.ai.girlfriend.free

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // setContentView(R.layout.activity_main) if you have one

        Thread {
            try {
                val modules = Py.modules()
                Log.i("Amelia", "Modules: $modules")

                for (mod in modules) {
                    val funcs = Py.functions(mod)
                    Log.i("Amelia", "Functions in $mod: $funcs")

                    if (funcs.isNotEmpty()) {
                        val out = Py.exec(mod, funcs[0], "testInput")
                        Log.i("Amelia", "$mod.${funcs[0]} → $out")
                    }
                }
            } catch (e: Exception) {
                Log.e("Amelia", "Error invoking Python", e)
            }
        }.start()
    }
}
