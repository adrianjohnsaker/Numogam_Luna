package com.antonio.my.ai.girlfriend.free.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.yourapp.repository.NumogramRepository
import com.yourapp.ui.state.NumogramUiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject

class NumogramViewModel(
    private val repository: NumogramRepository = NumogramRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow<NumogramUiState>(NumogramUiState.Idle)
    val uiState: StateFlow<NumogramUiState> get() = _uiState

    fun performTransition(userId: String, currentZone: String, feedback: Double) {
        viewModelScope.launch {
            _uiState.value = NumogramUiState.Loading
            val result = repository.transition(userId, currentZone, feedback)
            handleResult(result)
        }
    }

    fun performAsyncTransition(userId: String, currentZone: String, feedback: Double) {
        viewModelScope.launch {
            _uiState.value = NumogramUiState.Loading
            val result = repository.asyncTransition(userId, currentZone, feedback)
            handleResult(result)
        }
    }

    fun serializeMemory() {
        viewModelScope.launch {
            _uiState.value = NumogramUiState.Loading
            val result = repository.getSerializedState()
            handleResult(result)
        }
    }

    fun clearMemory() {
        viewModelScope.launch {
            val success = repository.resetMemory()
            _uiState.value = if (success) {
                NumogramUiState.Success(JSONObject().apply {
                    put("message", "Memory cleared.")
                })
            } else {
                NumogramUiState.Error("Failed to clear memory.")
            }
        }
    }

    fun cleanup() {
        viewModelScope.launch {
            val success = repository.cleanup()
            _uiState.value = if (success) {
                NumogramUiState.Success(JSONObject().apply {
                    put("message", "Resources cleaned up.")
                })
            } else {
                NumogramUiState.Error("Failed to clean up.")
            }
        }
    }

    fun safeCall(methodName: String, args: Map<String, Any>) {
        viewModelScope.launch {
            _uiState.value = NumogramUiState.Loading
            val result = repository.safeExecute(methodName, args)
            handleResult(result)
        }
    }

    private fun handleResult(result: JSONObject) {
        _uiState.value = when (result.optString("status")) {
            "success" -> NumogramUiState.Success(result)
            "error" -> NumogramUiState.Error(result.optString("error_message", "Unknown error"))
            else -> NumogramUiState.Error("Unexpected response format.")
        }
    }
}
