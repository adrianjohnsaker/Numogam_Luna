plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    // Chaquopy plugin – adjust the version as needed:
    id("com.chaquo.python") version "12.1.0"
}

android {
    compileSdk = 33

    defaultConfig {
        applicationId = "com.antonio.my.ai.girlfriend.free" 
        minSdk = 21
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"

        // Chaquopy configuration inside defaultConfig:
        python {
            version = "3.8"  // Specify the Python version your enhanced modules require
            pip {
                // List the required Python packages (enhanced module dependencies)
                install("numpy")
                install("scipy")
                // Add other packages as needed, e.g., for advanced text processing, ML, etc.
                // install("pandas")
                // install("tensorflow")
            }
        }
    }

    // Global Chaquopy configuration (optional; duplicates above if you need to set any global settings)
    chaquopy {
        defaultConfig {
            version = "3.8"
            pip {
                install("numpy")
                install("scipy")
            }
        }
    }

    // Configure the directories that hold your Python modules.
    // You can place your enhanced Python modules in these directories.
    sourceSets {
        getByName("main") {
            python {
                srcDirs("src/main/python", "src/main/assets")
            }
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            // Additional release settings as required.
        }
    }
}

dependencies {
    // Standard Android and Kotlin dependencies:
    implementation("androidx.core:core-ktx:1.9.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.activity:activity:1.6.0")
    implementation("com.google.android.material:material:1.9.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Include Firebase dependencies if needed:
    // implementation("com.google.firebase:firebase-messaging:23.2.1")

    // No additional dependencies are needed for Chaquopy since its components are packaged via the plugin.
}
