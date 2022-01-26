package it.unibo

import org.yaml.snakeyaml.Yaml

import java.io.FileInputStream
import java.util.concurrent.Executors
import java.{util => jutil}
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.jdk.CollectionConverters.SeqHasAsJava
@SuppressWarnings(Array("org.wartremover.warts.All")) //because we have to deal with java world
object MultiLearningRunner extends App {
  implicit class Unsafe(a: Any) {
    def as[T]: T = a.asInstanceOf[T]
    def list: jutil.List[Any] = as[jutil.List[Any]]
    def dict: jutil.Map[AnyRef, Any] = as[jutil.Map[AnyRef, Any]]
  }
  val startingFile = "src/main/yaml/swapSourceGradientRectangle.yml"
  val yaml = new Yaml()
  val dir = os.temp.dir(prefix = "simulations")
  val alphaBetaCombination = List((0.5, 0.01), (0.1, 0.01), (0.3, 0.02))
  val epsilonCombination = List((0.9, 10), (0.05, 100), (0.1, 90), (0.9, 40))
  val bucketsAndMax = List((2, 2), (2, 4), (4, 4), (4, 16), (4, 32))
  def baseYaml = {
    val loader = new FileInputStream(startingFile)
    val result = yaml.load[java.util.Map[String, Object]](loader)
    loader.close()
    result
  }

  val allSimulations = for {
    ((alpha, beta), i) <- alphaBetaCombination.zipWithIndex
    ((epsilon, decay), j) <- epsilonCombination.zipWithIndex
    ((buckets, max), k) <- bucketsAndMax.zipWithIndex
  } yield {
    def suffix = s"$alpha-$beta-$epsilon-$decay-$buckets-$max"
    def suffixNumber = s"$i$j$k"
    val base = baseYaml
    base.put("_beta", s"it.unibo.learning.TimeVariable.independent($beta)")
    base.put("_alpha", s"it.unibo.learning.TimeVariable.independent($alpha)")
    base
      .get("export")
      .list
      .get(0)
      .dict
      .put("parameters", List(s"gradientExperiments-$suffix", 1.0, s"./data/$suffixNumber").asJava)
    base.put("_epsilon", s" it.unibo.learning.TimeVariable.exponentialDecayFunction($epsilon, $decay)")
    base.put("_buckets", buckets.toString)
    base.put("_maxRadiusMultiplier", max.toString)
    val file = dir / s"sim-$suffix.yml"
    os.write(file, yaml.dump(base))
    file
  }
  implicit val executionContext =
    ExecutionContext.fromExecutor(Executors.newFixedThreadPool(Runtime.getRuntime.availableProcessors))

  val futures = allSimulations.map(file =>
    Future {
      println(s"process: $file")
      (file, os.proc("./gradlew", "startBatchUsing", s"-Pprogram=${file.wrapped.toFile.toString}").call(check = false))
    }
  )
  futures.foreach(f => f.onComplete(result => println(result)))
  futures.foreach(f => Await.result(f, Duration.Inf))
  println("End....")
  System.exit(0)
}
