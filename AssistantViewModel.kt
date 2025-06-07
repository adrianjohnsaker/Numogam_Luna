package com.antonio.my.ai.girlfriend.free.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import com.antonio.my.ai.girlfriend.free.PythonBridge

class AssistantViewModel(app: Application) : AndroidViewModel(app) {
    private val bridge = PythonBridge.getInstance(app)
    val resultLiveData = MutableLiveData<String>()

    /**
     * Calls any function in any Python module via PythonBridge.
     * @param moduleName Python module name (without .py)
     * @param functionName Function name in that module
     * @param args Arguments to pass to the Python function
     */
    fun callPythonFunction(moduleName: String, functionName: String, vararg args: Any?) {
        viewModelScope.launch {
            try {
                val result = bridge.executeFunction(moduleName, functionName, *args)
                resultLiveData.postValue(result?.toString() ?: "No result")
            } catch (e: Exception) {
                resultLiveData.postValue("Error: ${e.message}")
            }
        }
    }
}
