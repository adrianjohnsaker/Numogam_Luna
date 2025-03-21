package com.yourapp.ui.state

import org.json.JSONObject

sealed class NumogramUiState {
    object Idle : NumogramUiState()
    object Loading : NumogramUiState()
    data class Success(val data: JSONObject) : NumogramUiState()
    data class Error(val message: String) : NumogramUiState()
}
