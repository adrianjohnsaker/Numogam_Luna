package com.antonio.my.ai.girlfriend.free.amelia.ui.visual

import android.app.Activity
import android.view.ViewGroup
import androidx.compose.ui.platform.ComposeView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.fillMaxSize
import org.json.JSONObject

object DynamicVisualInjector {

    private var visualState: JSONObject? = null

    fun updateState(state: JSONObject?) {
        visualState = state
    }

    /** Injects the Compose visual layer into an existing Activity */
    fun injectInto(activity: Activity) {
        val root = activity.findViewById<ViewGroup>(android.R.id.content)
        val composeView = ComposeView(activity)
        composeView.setContent {
            EcologyVisualLayer(
                visualState = visualState,
                modifier = Modifier.fillMaxSize()
            )
        }
        root.addView(
            composeView,
            0 // insert at index 0 so it's *behind* all other views
        )
    }
}
