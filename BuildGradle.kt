// build.gradle (Project level)
buildscript {
    ext {
        compose_version = '1.5.4'
        kotlin_version = '1.9.10'
    }
}

plugins {
    id 'com.android.application' version '8.1.2' apply false
    id 'com.android.library' version '8.1.2' apply false
    id 'org.jetbrains.kotlin.android' version '1.9.10' apply false
    id 'com.chaquo.python' version '14.0.2' apply false
}

task clean(type: Delete) {
    delete rootProject.buildDir
}
