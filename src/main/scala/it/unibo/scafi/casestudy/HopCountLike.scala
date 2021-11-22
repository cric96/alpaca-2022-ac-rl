package it.unibo.scafi.casestudy

import cats.data.NonEmptySet
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.alchemist.tiggers.EndHandler
import it.unibo.learning.{Clock, Q, TimeVariable}
import it.unibo.scafi.casestudy.LearningProcess.RoundData
import it.unibo.storage.LocalStorage

import scala.util.Random

/** Common variable/constants/behaviour that have hop count learning problem */
trait HopCountLike
    extends AggregateProgram
    with HopCountLearning
    with StandardSensors
    with ScafiAlchemistSupport
    with BlockT
    with BlockG
    with FieldUtils
    with CustomSpawn
    with TemporalStateManagement
    with FixedGradients {
  // Type alias
  type State = List[Int]
  type Action = Int
  // Implicit context variable
  implicit lazy val rand: Random = randomGen
  // Storage
  lazy val qTableStorage = new LocalStorage[String](node.get[java.lang.String]("qtable_folder"))
  lazy val episode: Int = node.get[java.lang.Double]("episode").toInt
  lazy val clockTableStorage = new LocalStorage[String](node.get[java.lang.String]("clock_folder"))
  def passedTime: Double = alchemistTimestamp.toDouble
  // Variable loaded by alchemist configuration.
  lazy val learnCondition: Boolean = node.get[java.lang.Boolean]("learn")
  lazy val initialValue: Double = node.get[Double]("initial_value")
  // Other constant
  lazy val windowDifferenceSize: Int = node.get[java.lang.Integer]("window")
  lazy val trajectorySize: Int = node.get[java.lang.Integer]("trajectory")
  // Learning constants
  lazy val alpha: TimeVariable[Double] = node.get("alpha")
  lazy val beta: TimeVariable[Double] = node.get("beta")
  lazy val epsilon: TimeVariable[Double] = node.get("epsilon")
  lazy val gamma: Double = node.get[java.lang.Double]("gamma")
  // Q Learning data
  lazy val actions: NonEmptySet[Action] = node.get("actions")
  @SuppressWarnings(Array("org.wartremover.warts.Any")) // because of unsafe scala binding
  def qId: String
  // Pickle loose the default, so we need to replace it each time the map is loaded
  lazy val q: Q[State, Action] =
    qTableStorage.loadOrElse(qId, Q.zeros[State, Action]()).withDefault(initialValue)
  // Aggregate Program data
  lazy val clock: Clock = clockTableStorage.loadOrElse(mid().toString, Clock.start)
  // Constants
  val maxDiff = 100
  // Store data
  def endHandler: EndHandler[_]

  def aggregateProgram(): RoundData[State, Action, Double]

  final override def main(): Any =
    (aggregateProgram(), endHandler)

  protected def learningProblem(reference: Int) = learningProcess(q)
    .stateDefinition(stateFromWindow)
    .rewardDefinition(output => rewardSignal(reference, output))
    .actionEffectDefinition((output, action) => output + action + 1)
    .initialConditionDefinition(List.empty, Double.PositiveInfinity)

  protected def stateFromWindow(output: Double): State = {
    val minOutput = minHood(nbr(output))
    val recent = recentValues(windowDifferenceSize, minOutput)
    val oldState = recent.headOption.getOrElse(minOutput)
    val diff = (minOutput - oldState) match {
      case diff if Math.abs(diff) > maxDiff => maxDiff * diff.sign
      case diff                             => diff
    }
    recentValues(trajectorySize, diff).toList.map(_.toInt)
  }

  protected def rewardSignal(groundTruth: Double, currentValue: Double): Double =
    if ((groundTruth.toInt - currentValue.toInt) == 0) { 0 }
    else { -1 }
}
