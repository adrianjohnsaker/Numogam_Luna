class PythonBridge(private val context: Context) {
    init {
        if (!Python.isStarted()) {
            // Enable asset-based Python loading
            Python.start(AndroidPlatform(context, true)) 
        }

        injectAssetModules()
        validateEnvironment()
    }

    private fun injectAssetModules() {
        val pythonDir = File(context.filesDir, "python_modules")
        if (!pythonDir.exists()) pythonDir.mkdir()
        
        // Copy from assets/python to internal storage
        context.assets.list("python")?.forEach { assetFile ->
            File(pythonDir, assetFile).outputStream().use { out ->
                context.assets.open("python/$assetFile").copyTo(out)
            }
        }

        // Add to Python path
        val sys = Python.getInstance().getModule("sys")
        sys["path"]?.callAttr("insert", 0, pythonDir.absolutePath)
    }

    private fun validateEnvironment() {
        try {
            val sys = Python.getInstance().getModule("sys")
            Log.d(TAG, "Python path: ${sys["path"]}")
            
            // Verify critical module
            Python.getInstance().getModule("numogram")
        } catch (e: Exception) {
            Log.e(TAG, "Environment validation failed: ${e.message}")
            throw RuntimeException("Python environment broken")
        }
    }
}
