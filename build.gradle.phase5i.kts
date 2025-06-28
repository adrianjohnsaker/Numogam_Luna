plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
    id 'com.chaquo.python'
    id 'kotlin-kapt'
    id 'com.google.dagger.hilt.android'
}

android {
    namespace 'com.antonio.my.ai.girlfriend.free.consciousness.amelia'
    compileSdk 34

    defaultConfig {
        applicationId "com.consciousness.amelia"
        minSdk 24
        targetSdk 34
        versionCode 5
        versionName "5.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary true
        }

        // Python configuration for Phase 5
        python {
            pip {
                // Core packages
                install "numpy==1.24.3"
                install "scipy==1.10.1"
                install "ujson==5.7.0"
                
                // Phase 5 specific
                // All Phase 5 functionality implemented in pure Python
            }
            
            pyc {
                src false
            }
            
            version "3.8"
        }
        
        multiDexEnabled true
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
        freeCompilerArgs += [
            "-opt-in=androidx.compose.material3.ExperimentalMaterial3Api",
            "-opt-in=androidx.compose.animation.ExperimentalAnimationApi"
        ]
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
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.7.0'
    implementation 'androidx.activity:activity-compose:1.8.2'
    
    // Compose BOM
    implementation platform('androidx.compose:compose-bom:2024.01.00')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.ui:ui-graphics'
    implementation 'androidx.compose.ui:ui-tooling-preview'
    implementation 'androidx.compose.material3:material3'
    implementation 'androidx.compose.material:material-icons-extended'
    implementation 'androidx.compose.animation:animation'
    implementation 'androidx.compose.animation:animation-graphics'
    
    // Lifecycle and ViewModel
    implementation "androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0"
    implementation "androidx.lifecycle:lifecycle-runtime-compose:2.7.0"
    
    // Navigation
    implementation "androidx.navigation:navigation-compose:2.7.6"
    
    // Hilt
    implementation "com.google.dagger:hilt-android:$hilt_version"
    kapt "com.google.dagger:hilt-compiler:$hilt_version"
    implementation "androidx.hilt:hilt-navigation-compose:1.1.0"
    
    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // Python Integration
    implementation 'com.chaquo.python:python:15.0.0'
    
    // DataStore
    implementation "androidx.datastore:datastore-preferences:1.0.0"
    
    // Work Manager
    implementation "androidx.work:work-runtime-ktx:2.9.0"
    
    // JSON Processing
    implementation 'org.json:json:20231013'
    
    // Phase 5 specific - Advanced graphics
    implementation "androidx.graphics:graphics-core:1.0.0-alpha05"
    
    // Testing
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
    androidTestImplementation platform('androidx.compose:compose-bom:2024.01.00')
    androidTestImplementation 'androidx.compose.ui:ui-test-junit4'
    debugImplementation 'androidx.compose.ui:ui-tooling'
    debugImplementation 'androidx.compose.ui:ui-test-manifest'
}

kapt {
    correctErrorTypes true
}
```
