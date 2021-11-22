import java.io.ByteArrayOutputStream
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
// Scala version
val scalaLib: String by project
val scala: String by project
val batch: String by project
val deepLearningVersion = "1.0.0-M1.1"
plugins {
    java
    scala
    id("com.github.johnrengelman.shadow") version "4.0.3"
    id("org.scoverage") version "7.0.0"
    id("cz.augi.gradle.wartremover") version "0.14.2"
    idea
    kotlin("jvm") version "1.3.50"

}
repositories {
    maven {
        url = uri("${projectDir}/libs")
    }
    mavenCentral()
}

dependencies {
    // Alchemist dependency
    implementation("it.unibo.alchemist:alchemist:_")
    implementation("it.unibo.alchemist:alchemist-incarnation-scafi:_")
    implementation("it.unibo.alchemist:alchemist-swingui:_")
    // Scala par
    implementation("org.scala-lang.modules:scala-parallel-collections_${scala}:_")
    // ScaFi dependency
    implementation("org.scala-lang:scala-library:$scalaLib")
    implementation("it.unibo.scafi:scafi-core_${scala}:_")
    // Cats dependency
    implementation("org.typelevel:cats-core_${scala}:_")
    // Monocle dependency
    implementation("dev.optics:monocle-core_${scala}:_")
    implementation("dev.optics:monocle-macro_${scala}:_")
    // uPickle dependency
    implementation("com.lihaoyi:upickle_${scala}:_")
    implementation("com.lihaoyi:os-lib_${scala}:_")
    // Deep learning 4 j dependecies
    implementation("org.deeplearning4j:deeplearning4j-core:${deepLearningVersion}")
    implementation("org.nd4j:nd4j-native-platform:${deepLearningVersion}")
    implementation("org.nd4j:nd4j-api:${deepLearningVersion}")
    // ScalaPy dependencies
    implementation("me.shadaj:scalapy-core_${scala}:_")
    // Tests dependency
    testImplementation("org.scalatest:scalatest_${scala}:_")
    testImplementation("junit:junit:_")
    testImplementation("org.scalatestplus:junit-4-13_2.13:_")
}
tasks.withType<ScalaCompile> {
    scalaCompileOptions.additionalParameters = mutableListOf("-Xfatal-warnings", "-feature")
    sourceCompatibility = "1.8"
    targetCompatibility = "1.8"
}

// Heap size estimation for batches
val maxHeap: Long? by project
val heap: Long = maxHeap ?:
if (System.getProperty("os.name").toLowerCase().contains("linux")) {
    ByteArrayOutputStream().use { output ->
        exec {
            executable = "bash"
            args = listOf("-c", "cat /proc/meminfo | grep MemAvailable | grep -o '[0-9]*'")
            standardOutput = output
        }
        output.toString().trim().toLong() / 1024
    }
        .also { println("Detected ${it}MB RAM available.") }  * 9 / 10
} else {
    // Guess 16GB RAM of which 2 used by the OS
    14 * 1024L
}
val taskSizeFromProject: String? by project
val threadCountFromProject: String? by project
val taskSize: Int = taskSizeFromProject?.toInt() ?: 512
val threadCount = threadCountFromProject?.toInt() ?: maxOf(1, minOf(Runtime.getRuntime().availableProcessors(), heap.toInt() / taskSize ))
val today = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"))
val alchemistGroup = "Run Alchemist"
/*
 * This task is used to run all experiments in sequence
 */
val runAllGraphic by tasks.register<DefaultTask>("runAllGraphic") {
    group = alchemistGroup
    description = "Launches all simulations with the graphic subsystem enabled"
}

val runAllBatch by tasks.register<DefaultTask>("runAllBatch") {
    group = alchemistGroup
    description = "Launches all experiments"
}
/*
 * Scan the folder with the simulation files, and create a task for each one of them.
 */

File(rootProject.rootDir.path + "/src/main/yaml").listFiles()
    ?.filter { it.extension == "yml" }
    ?.sortedBy { it.nameWithoutExtension }
    ?.forEach {
        fun basetask(name: String, additionalConfiguration: JavaExec.() -> Unit = {}) = tasks.register<JavaExec>(name) {
            group = alchemistGroup
            description = "Launches graphic simulation ${it.nameWithoutExtension}"
            main = "it.unibo.alchemist.Alchemist"
            classpath = sourceSets["main"].runtimeClasspath
            args("-y", it.absolutePath)
            if (System.getenv("CI") == "true") {
                args("-hl", "-t", "2")
            } else {
                args("-g", "src/main/effects/${it.nameWithoutExtension}.aes")
            }
            this.additionalConfiguration()
        }
        val capitalizedName = it.nameWithoutExtension.capitalize()
        val graphic by basetask("run${capitalizedName}Graphic")
        runAllGraphic.dependsOn(graphic)
        val batch by basetask("run${capitalizedName}Batch") {
            description = "Launches batch experiments for $capitalizedName"
            jvmArgs("-XX:+AggressiveHeap")
            maxHeapSize = "${minOf(heap.toInt(), Runtime.getRuntime().availableProcessors() * taskSize)}m"
            File("data").mkdirs()
            args(
                "-e", "data/${it.nameWithoutExtension}",
                //"-b",
                "-e", "data/${today}-${name}",
                "-var", "episode",
                "-p", threadCount,
                "-i", 1
            )
        }
        runAllBatch.dependsOn(batch)
    }