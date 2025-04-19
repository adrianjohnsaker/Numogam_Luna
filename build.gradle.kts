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
