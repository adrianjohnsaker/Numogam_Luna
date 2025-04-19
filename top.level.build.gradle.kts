// Top-level build.gradle.kts
buildscript {
    repositories {
        google()
        mavenCentral()
        maven(url = "https://chaquo.com/maven")
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.1.0")
        classpath("com.chaquo.python:gradle:14.0.2")  // Chaquopy plugin
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// Module-level configuration
android {
    compileSdk = 33
    
    defaultConfig {
        applicationId = "com.antonio.my.ai.girlfriend.fref"
        minSdk = 24
        targetSdk = 33
        versionCode = 13
        versionName = "1.2"
        
        // Chaquopy configuration
        ndk {
            abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64"))
        }
    }
    
    // Python configuration
    chaquopy {
        defaultConfig {
            version = "3.8"
            
            pip {
                // Add any pip packages your modules might need
                install("numpy")
                install("scipy")
            }
        }
    }
    
    sourceSets {
        getByName("main") {
            python {
                srcDirs("src/main/python", "src/main/assets")
            }
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.9.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.activity:activity:1.6.0")
    implementation("com.google.android.material:material:1.9.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    
    // Firebase dependencies if needed
    // implementation("com.google.firebase:firebase-messaging:23.2.1")
    
    // No additional dependencies needed for Chaquopy
    // as they're included with the plugin
}
