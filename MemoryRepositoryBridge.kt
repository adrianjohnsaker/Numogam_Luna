import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MemoryRepositoryBridge {
    private lateinit var memoryRepository: PyObject

    init {
        // Initialize Python environment if not already initialized
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        
        val python = Python.getInstance()
        val module = python.getModule("memory_repository")
        
        // Create MemoryRepository instance
        memoryRepository = module.callAttr("MemoryRepository")
    }

    fun saveMemory(memoryName: String, content: String) {
        memoryRepository.callAttr("save_memory", memoryName, content)
    }

    fun loadMemory(memoryName: String): Map<String, String> {
        return memoryRepository.callAttr("load_memory", memoryName).convertToJavaObject() as Map<String, String>
    }

    fun listMemories(): List<String> {
        return memoryRepository.callAttr("list_memories").convertToJavaObject() as List<String>
    }

    fun deleteMemory(memoryName: String) {
        memoryRepository.callAttr("delete_memory", memoryName)
    }

    fun initializeAmeliaMemories() {
        val module = Python.getInstance().getModule("memory_repository")
        module.callAttr("initialize_amelia_memories", memoryRepository)
    }
}
