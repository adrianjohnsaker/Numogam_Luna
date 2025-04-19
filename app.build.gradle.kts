plugins {
    id("com.android.application")
    id("kotlin-android")
    id("com.chaquo.python")  // Apply Chaquopy plugin
}

android {
    // Standard Android configuration
    compileSdk = 34
    defaultConfig {
        applicationId = "com.yourcompany.yourapp"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // Chaquopy configuration
        python {
            version = "3.11"
            buildPython = "/usr/local/bin/python3"  // Path to Python executable
            pip {
                install("numpy")                   // Python dependencies
                install("pandas")
                install("your-custom-package")     // Local modules
            }
        }

        ndk {
            abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64"))
        }
    }

    sourceSets {
        getByName("main") {
            python.srcDir("src/main/python")  // Python source directory
        }
    }

    // ... rest of Android config
}

dependencies {
    // Standard Android/Kotlin dependencies
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
}
