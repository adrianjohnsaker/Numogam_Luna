val python = Python.getInstance()
val module = python.getModule("chat_enhancer")
val response = module.callAttr("enhance_response", userInput).toString()
