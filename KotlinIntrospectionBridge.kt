package com.amelia.introspection

import android.content.Context
import android.util.Log
import org.json.JSONObject
import com.antonio.my.ai.girlfriend.free.PythonConnector

/**
 * Provides access to Amelia's introspection capabilities
 */
class IntrospectionBridge(private val context: Context) {
    private val TAG = "IntrospectionBridge"
    private val pythonConnector = PythonConnector.getInstance(context)
    private var initialized = false
    
    /**
     * Initialize the introspection system
     */
    fun initialize() {
        if (initialized) return
        
        try {
            // Make sure the Python connector is initialized
            pythonConnector.initialize()
            
            // Extract the introspection modules from assets to the modules directory
            extractIntrospectionModules()
            
            // Call the initialize function in the introspection hook
            val baseDir = context.getDir("amelia_codebase", Context.MODE_PRIVATE).absolutePath
            val result = pythonConnector.callPythonFunction(
                "introspection_hook", 
                "initialize",
                baseDir
            )
            
            // Check the result
            if (result is Map<*, *> && result["status"] == "success") {
                initialized = true
                Log.d(TAG, "Introspection system initialized successfully")
            } else {
                Log.e(TAG, "Failed to initialize introspection system: $result")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing introspection system", e)
        }
    }
    
    /**
     * Process an introspection command
     */
    fun processCommand(command: String): JSONObject {
        if (!initialized) {
            initialize()
        }
        
        return try {
            val result = pythonConnector.callPythonFunction(
                "introspection_hook",
                "process_command",
                command
            )
            
            if (result is Map<*, *>) {
                JSONObject(result as Map<String, Any>)
            } else {
                JSONObject().put("status", "error")
                    .put("error_message", "Invalid result format")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing introspection command", e)
            JSONObject().put("status", "error")
                .put("error_message", e.message)
        }
    }
    
    /**
     * Check if a query is asking for introspection information
     */
    fun isIntrospectionQuery(query: String): Boolean {
        try {
            val result = pythonConnector.callPythonFunction(
                "introspection_hook",
                "extract_command_from_query",
                query
            )
            
            if (result is Map<*, *>) {
                return result["is_introspection"] == true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking if query is introspection", e)
        }
        
        return false
    }
    
    /**
     * Process an introspection query
     */
    fun processQuery(query: String): JSONObject {
        if (!initialized) {
            initialize()
        }
        
        return try {
            // Extract the command from the query
            val extractResult = pythonConnector.callPythonFunction(
                "introspection_hook",
                "extract_command_from_query",
                query
            )
            
            if (extractResult is Map<*, *> && extractResult["status"] == "success") {
                val command = extractResult["command"] as String
                
                // Process the command
                val commandResult = pythonConnector.callPythonFunction(
                    "introspection_hook",
                    "process_command",
                    command
                )
                
                if (commandResult is Map<*, *>) {
                    JSONObject(commandResult as Map<String, Any>)
                } else {
                    JSONObject().put("status", "error")
                        .put("error_message", "Invalid result format")
                }
            } else {
                JSONObject().put("status", "not_introspection")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing introspection query", e)
            JSONObject().put("status", "error")
                .put("error_message", e.message)
        }
    }
    
    /**
     * Register a runtime object with the introspection system
     */
    fun registerRuntimeObject(objId: String, objType: String, properties: Map<String, Any>): Boolean {
        if (!initialized) {
            initialize()
        }
        
        return try {
            val result = pythonConnector.callPythonFunction(
                "introspection_hook",
                "register_runtime_object",
                objId,
                objType,
                properties
            )
            
            (result is Map<*, *> && result["status"] == "success")
        } catch (e: Exception) {
            Log.e(TAG, "Error registering runtime object", e)
            false
        }
    }
    
    /**
     * Extract introspection modules from assets to the modules directory
     */
    private fun extractIntrospectionModules() {
        // This would extract your introspection Python modules from assets
        // Similar to how PythonConnector extracts Python modules
        
        // List of introspection modules to extract
        val modules = listOf(
            "introspection_hook.py",
            "system_introspection_module.py",
            "amelia_introspection_interface.py"
        )
        
        // The modules would be extracted to the Python modules directory
        // This is handled by your PythonConnector class
    }
}
