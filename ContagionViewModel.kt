class ContagionViewModel(application: Application) : AndroidViewModel(application) {

    private val numogram = NumogramBridge(application)
    private val _output = MutableLiveData<String>()
    val output: LiveData<String> = _output

    fun loadContagionRecord(recordId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val result = numogram.getContagionRecord(recordId)
            withContext(Dispatchers.Main) {
                _output.value = result
            }
        }
    }
}
