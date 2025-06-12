// build.gradle (Module: app) - COMPLETE FILE
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
    id 'com.chaquo.python'
}

android {
    namespace 'com.antonio.my.ai.girlfriend.free.amelia.consciousness'
    compileSdk 34

    defaultConfig {
        applicationId "com.antonio.my.ai.girlfriend.free.amelia.consciousness"
        minSdk 24
        targetSdk 34
        versionCode 2
        versionName "2.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary true
        }

        // Python configuration
        python {
            pip {
                // Core Python packages
                install "numpy==1.24.3"
                install "scipy==1.10.1"
                
                // JSON handling
                install "ujson==5.7.0"
                
                // For Phase 1 consciousness core
                // No additional packages needed - using Python stdlib
                
                // For Phase 2 HTM and temporal processing
                // HTM functionality implemented in pure Python
            }
            
            pyc {
                src false  // Don't compile to bytecode for easier debugging
            }
            
            // Python version
            version "3.8"
        }
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
        debug {
            debuggable true
            minifyEnabled false
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    
    kotlinOptions {
        jvmTarget = '1.8'
    }
    
    buildFeatures {
        compose true
    }
    
    composeOptions {
        kotlinCompilerExtensionVersion '1.5.4'
    }
    
    packagingOptions {
        resources {
            excludes += '/META-INF/{AL2.0,LGPL2.1}'
        }
    }
}

dependencies {
    // Android Core
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.6.2'
    implementation 'androidx.activity:activity-compose:1.8.0'
    
    // Compose
    implementation platform('androidx.compose:compose-bom:2023.10.01')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.ui:ui-graphics'
    implementation 'androidx.compose.ui:ui-tooling-preview'
    implementation 'androidx.compose.material3:material3'
    implementation 'androidx.compose.animation:animation'
    
    // Lifecycle and ViewModel
    implementation 'androidx.lifecycle:lifecycle-viewmodel-compose:2.6.2'
    implementation 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.6.2'
    implementation 'androidx.lifecycle:lifecycle-runtime-compose:2.6.2'
    
    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // Python Integration
    implementation 'com.chaquo.python:python:11.0.0'
    
    // JSON Processing
    implementation 'org.json:json:20230618'
    
    // Testing
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
    androidTestImplementation platform('androidx.compose:compose-bom:2023.10.01')
    androidTestImplementation 'androidx.compose.ui:ui-test-junit4'
    debugImplementation 'androidx.compose.ui:ui-tooling'
    debugImplementation 'androidx.compose.ui:ui-test-manifest'
}

// proguard-rules.pro (if you need it)
# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# Uncomment this to preserve the line number information for
# debugging stack traces.
-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
-renamesourcefileattribute SourceFile

# Chaquopy
-keep class com.chaquo.python.** { *; }
-keep class com.amelia.consciousness.** { *; }

# Compose
-keep class androidx.compose.** { *; }
-keep class kotlin.** { *; }
-keep class kotlinx.coroutines.** { *; }
