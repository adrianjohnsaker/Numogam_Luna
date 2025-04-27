plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
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

        python {
            version = "3.8"
            pip {
                install("numpy") // For numerical computations
                install("scipy") // For scientific computations
                install("nltk") // For NLP tasks
                install("textblob") // For lightweight NLP
                install("spacy") // For advanced NLP
                install("sentence-transformers") // For semantic embeddings
                install("pandas") // For data manipulation
                // Note: 'transformers' is resource-heavy; uncomment if needed and test on target devices
                // install("transformers")
            }
            // Pre-extract packages to reduce startup time
            extractPackages = ["nltk", "textblob", "spacy", "sentence-transformers", "pandas"]
        }
    }

    sourceSets {
        getByName("main") {
            python {
                srcDirs("src/main/python", "src/main/assets")
            }
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        getByName("debug") {
            // Enable Chaquopy error reporting for debugging
            isDebuggable = true
        }
    }
}

chaquopy {
    defaultConfig {
        python {
            version = "3.8"
            pip {
                install("numpy")
                install("scipy")
                install("nltk")
                install("textblob")
                install("spacy")
                install("sentence-transformers")
                install("pandas")
                // install("transformers") // Uncomment if needed
            }
            // Generate static proxies for key Python modules to improve performance
            staticProxy = ["chat_enhancer", "nlp_processor"] // Adjust module names as per your Python code
            // Enable error reporting for Python exceptions
            errorReporting = true
        }
    }
    sourceSets {
        main {
            python.srcDir "src/main/python"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.9.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.activity:activity:1.6.0")
    implementation("com.google.android.material:material:1.9.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    // implementation("com.google.firebase:firebase-messaging:23.2.1") // Uncomment if needed
}
