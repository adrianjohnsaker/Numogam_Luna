#include <jni.h>
#include <stdio.h>
#include <Python.h>
#include "com_yourpackage_PythonBridge.h"

// Global Python initialization (done once)
void initialize_python() {
    Py_Initialize();
}

// Cleanup Python (only when shutting down)
void finalize_python() {
    Py_Finalize();
}

JNIEXPORT jstring JNICALL Java_com_yourpackage_PythonBridge_sendToPython(JNIEnv *env, jclass clazz, jstring input) {
    // Convert Java string to C string
    const char *inputStr = (*env)->GetStringUTFChars(env, input, NULL);

    // Ensure Python is initialized
    if (!Py_IsInitialized()) {
        initialize_python();
    }

    // Load Python module and function
    PyObject *pyModule = PyImport_ImportModule("python_script");  // Ensure python_script.py exists
    if (!pyModule) {
        (*env)->ReleaseStringUTFChars(env, input, inputStr);
        return (*env)->NewStringUTF(env, "Error: Cannot load Python module");
    }

    PyObject *pyFunc = PyObject_GetAttrString(pyModule, "process_input");
    if (!pyFunc || !PyCallable_Check(pyFunc)) {
        Py_DECREF(pyModule);
        (*env)->ReleaseStringUTFChars(env, input, inputStr);
        return (*env)->NewStringUTF(env, "Error: Cannot find process_input function");
    }

    // Prepare Python arguments
    PyObject *pyArgs = PyTuple_Pack(1, PyUnicode_FromString(inputStr));
    PyObject *pyResult = PyObject_CallObject(pyFunc, pyArgs);

    // Extract result
    const char *resultStr = PyUnicode_AsUTF8(pyResult);

    // Clean up
    Py_DECREF(pyArgs);
    Py_DECREF(pyResult);
    Py_DECREF(pyFunc);
    Py_DECREF(pyModule);
    (*env)->ReleaseStringUTFChars(env, input, inputStr);

    // Return result to Java
    return (*env)->NewStringUTF(env, resultStr);
}
