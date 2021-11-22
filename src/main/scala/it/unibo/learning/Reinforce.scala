package it.unibo.learning

import cats.data.NonEmptyList
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{Module, PyQuote, SeqConverters}
import me.shadaj.scalapy.readwrite.Writer
import org.nd4j.autodiff.samediff.{SDVariable, SameDiff, TrainingConfig}
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.conditions.IsNaN
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.weightinit.impl.{DistributionInitScheme, ReluUniformInitScheme}

import java.io.File
import scala.collection.mutable
import scala.jdk.CollectionConverters.MutableMapHasAsJava
import scala.util.Random
object Reinforce extends App {
  type Target[S, A] = ReinforceNetwork

  case class Plain(actions: NonEmptyList[Int], gamma: Double)
      extends ReinforcementLearning[Seq[(List[Int], Int, Double)], ReinforceNetwork] {
    override def improve(trajectory: Seq[(List[Int], Int, Double)], target: ReinforceNetwork, clock: Clock)(implicit
        rand: Random
    ): ReinforceNetwork = {

      val returns = trajectory.scanLeft(0.0) { case (g, (_, _, r)) => gamma * g + r }
      val withCorrection = returns.reverse.zipWithIndex.map { case (g, i) => math.pow(gamma, i) * g }
      target.reset()
      trajectory.reverse.zip(withCorrection).foreach { case ((s, a, _), g) =>
        try target.fit(a, s, g)
        catch {
          case e: Exception => println(e)
        }
      }
      target.reset()
      target
    }
  }

  trait ReinforceNetwork {
    def init(): Unit
    def store(file: File): Unit
    def fit(action: Int, state: List[Int], g: Double): Unit
    def pass(state: List[Int])(implicit rn: Random): Int
    def eval(state: List[Int]): Array[Double]
    def reset(): Unit = {}
    def close(): Unit = {}
    def load(file: File): Unit
  }
  @SuppressWarnings(Array("org.wartremover.warts.All"))
  class SameDiffReinforceNetwork(inputSize: Int, layers: List[Int], seed: Int, minStateValue: Int)
      extends ReinforceNetwork {
    var sd: SameDiff = _
    private var counter = 0
    private val emptyState = List.fill(inputSize)(minStateValue)
    Nd4j.getRandom.setSeed(seed)
    val condition = new IsNaN
    private val config = new TrainingConfig.Builder()
      .updater(new Adam(0.001))
      .l2(1e-4)
      .dataSetFeatureMapping("input", "action")
      .dataSetLabelMapping("g")
      .minimize(false)
      .build()
    def init(): Unit = {
      sd = SameDiff.create()
      val input = sd.placeHolder("input", DataType.FLOAT, -1, inputSize)
      val action = sd.placeHolder("action", DataType.UINT8, -1, 1)
      val advantage = sd.placeHolder("g", DataType.FLOAT, -1, 1)
      val first = linear(input, inputSize, layers.head, sd)
      val linearLayers = layers.zip(layers.tail).scanLeft(first) { case (prev, (in, out)) => linear(prev, in, out, sd) }
      val output = sd.nn.softmax("output", linearLayers.last)
      // reinforce part
      val actionSelection = sd.gather("middle", output, action, 1)
      val logProb = sd.math.log(actionSelection.add(0.00001))
      val loss = logProb.mul("loss", advantage)
      loss.markAsLoss()
      sd.setTrainingConfig(config)
    }

    def refineState(state: List[Int]): Array[Int] = (state ++ emptyState take inputSize).toArray

    def fit(action: Int, state: List[Int], g: Double): Unit = {
      val actionTensor = Nd4j.create(Array(Array(action)))
      val stateTensor = Nd4j.create(Array(refineState(state)))
      val gTensor = Nd4j.create(Array(g))
      val dataset = new MultiDataSet(Array(stateTensor, actionTensor), Array(gTensor))
      sd.fit(dataset)
      //println(sd.output(dataset, "loss", "output").asScala)
    }

    def pass(state: List[Int])(implicit rn: Random): Int = {
      val result = eval(state)
      val ordered = result.zipWithIndex.map { case (data, i) => i -> data }.sortBy(_._2)
      Stochastic.sampleFrom(NonEmptyList.fromListUnsafe(ordered.toList))
    }

    def store(where: File) = sd.save(where, true)

    def load(where: File): Unit =
      sd = SameDiff.load(where, true)

    private def linear(prev: SDVariable, nIn: Int, nOut: Int, sd: SameDiff): SDVariable = {
      counter += 1
      val weights = sd.`var`(s"weights-$counter", new ReluUniformInitScheme('c', 1), DataType.FLOAT, nIn, nOut)
      val bias = sd.`var`(s"bias-$counter", new DistributionInitScheme('c', Nd4j.getDistributions.createUniform(0, 1)))
      sd.nn.tanh(prev.mmul(weights).add(bias))
    }

    override def eval(state: List[Int]): Array[Double] = {
      val map = new mutable.HashMap[String, INDArray]()
      map += "input" -> Nd4j.create(Array(refineState(state)))
      map += "action" -> Nd4j.create(Array(Array(0)))
      map += "g" -> Nd4j.create(Array(0.0))
      val result = sd.outputSingle(map.asJava, "output")
      result.toDoubleVector
    }
  }

  @SuppressWarnings(Array("org.wartremover.warts.All"))
  class TfReinforceNetwork(inputSize: Int, layers: List[Int], seed: Int, minStateValue: Int, memory: Boolean = false)
      extends ReinforceNetwork {
    def list[E: Writer](elements: E*): py.Any = elements.toList.toPythonCopy
    private val emptyState = List.fill(inputSize)(minStateValue)
    private val tf: Module = py.module("tensorflow")
    private val np: Module = py.module("numpy")
    private val gc: Module = py.module("gc")
    private val optAlpha = tf.keras.optimizers.SGD(learning_rate = 0.01)
    private val optBeta = tf.keras.optimizers.SGD(learning_rate = 0.0001)

    private val normValue = 5.0
    private val hiddenLayers = layers
      .dropRight(1)
      .map(units => tf.keras.layers.Dense(units = units, activation = "relu"))
    private val kerasLayers = if (memory) { List(tf.keras.layers.LSTM(units = 4, stateful = true)) }
    else
      List.empty ::: hiddenLayers ::: List(
        tf.keras.layers.Dense(units = layers.last, activation = "softmax")
      )

    private var model = tf.keras.Sequential(kerasLayers.toPythonCopy)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    override def init(): Unit = {}
    override def store(file: File): Unit = {
      println(file.getAbsoluteFile.toString)
      model.save(file.getAbsoluteFile.toString)
    }

    override def fit(action: Int, state: List[Int], g: Double): Unit = {
      py.`with`(tf.GradientTape()) { tape =>
        val prob = model(input(state))
        val loss = py"$g * ${tf.math.log(prob.bracketAccess(0, action) + 0.00001)}"
        val gradients = tape.gradient(loss, model.trainable_variables)
        val opt = if (py"($loss.numpy())".as[Double] < 0) { optAlpha }
        else { optBeta }
        opt.apply_gradients(
          py.Dynamic.global
            .zip(tf.clip_by_global_norm(gradients, normValue).bracketAccess(0), model.trainable_variables)
        )
      }
    }

    def input(state: List[Int]): py.Any = {
      val scalaInput = (state ++ emptyState).take(inputSize).map(_.toDouble)
      if (memory) {
        np.array(list(list(list(scalaInput: _*))))
      } else {
        np.array(list(list(scalaInput: _*)))
      }
    }
    override def eval(state: List[Int]): Array[Double] = {
      model(input(state))
        .numpy()
        .tolist()
        .as[Seq[Seq[Double]]]
        .headOption
        .map(_.toArray)
        .getOrElse(Array.empty)
    }

    override def reset(): Unit =
      model.reset_states()

    override def close(): Unit = {
      model.del()
      tf.keras.backend.clear_session()
      gc.collect()
    }

    override def pass(state: List[Int])(implicit rn: Random): Int = {
      val proba = model(input(state))
        .numpy()
      np.random.choice(py.Dynamic.global.len(proba.bracketAccess(0)), p = (proba.bracketAccess(0))).as[Int]
    }

    override def load(file: File): Unit = model = tf.keras.models.load_model(file.getAbsoluteFile.toString)
  }

  val net = new TfReinforceNetwork(2, 10 :: 2 :: Nil, 123, 10)

  println(net.eval(List(1, 2)).mkString(","))
  net.fit(1, List(1, 2), 0.1)
  net.fit(1, List(1, 2), 0.1)
  net.fit(1, List(1, 2), 0.1)
  net.fit(1, List(1, 2), 0.1)
  net.fit(1, List(1, 2), 0.1)
  println(net.eval(List(1, 2)).mkString(","))
  net.close()

  println(net.eval(List(1, 2)).mkString(","))
}
