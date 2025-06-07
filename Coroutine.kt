lifecycleScope.launch(Dispatchers.IO) {
    val result = pyModule.callAttr("my_python_func", input)
    withContext(Dispatchers.Main) {
        // Update UI with result
    }
}
