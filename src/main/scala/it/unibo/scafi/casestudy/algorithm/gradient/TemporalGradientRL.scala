package it.unibo.scafi.casestudy.algorithm.gradient

import cats.data.NonEmptySet
import it.unibo.alchemist.model.implementations.nodes.NodeManager
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.{ScafiAlchemistSupport, _}
import it.unibo.learning.Q.MutableQ
import it.unibo.learning.{Policy, Q, QLearning}
import it.unibo.scafi.casestudy.algorithm.RLLike
import it.unibo.scafi.casestudy.algorithm.RLLike.AlgorithmHyperparameter
import it.unibo.scafi.casestudy.algorithm.gradient.TemporalGradientRL._
import it.unibo.scafi.casestudy.{GradientLikeLearning, TemporalStateManagement}
import it.unibo.storage.LocalStorage
import upickle.default.{macroRW, ReadWriter => RW}

import scala.util.Random

trait TemporalGradientRL extends RLLike {
  self: AggregateProgram
    with TemporalStateManagement
    with GradientLikeLearning
    with ScafiAlchemistSupport
    with FieldUtils
    with StandardSensors =>
  // move to another part?
  implicit class DoubleWithAlmostEquals(val d: Double) {
    def default = 0.00001
    def ~=(d2: Double) = (d - d2).abs < default
  }
  class TemporalRLAlgorithm(
      parameter: AlgorithmHyperparameter,
      actionSet: NonEmptySet[Action],
      radius: Double,
      maxBound: Int,
      bucketsCount: Int,
      windowDifferenceSize: Int,
      trajectorySize: Int
  )(implicit rand: Random)
      extends AlgorithmTemplate[History, Action] {

    private val max = maxBound * bucketsCount
    private val steps = maxBound * 2
    private val iterableSteps = -max to max by steps
    private val buckets =
      iterableSteps.reverse.drop(1).reverse.zip(iterableSteps.drop(1)).map { case (min, max) =>
        Slot(min / bucketsCount.toDouble, max / bucketsCount.toDouble)
      }
    override val name: String = "temporalRL"
    override protected def learning: QLearning.Type[History, Action] =
      QLearning.Hysteretic[History, Action](actionSet, parameter.alpha, parameter.beta, parameter.gamma)

    override protected def state(output: Double, action: Action): History = {
      def evalState(out: Double): GradientDifference = {
        val diff = output - out
        if ((diff ~= 0)) {
          Same
        } else if (diff > maxBound * radius) {
          GreaterBound
        } else if (diff < maxBound * -radius) {
          SmallerBound
        } else {
          @SuppressWarnings(Array("org.wartremover.warts.All")) // because serialization with case object
          val slot =
            buckets.find { case Slot(min, max) => diff > min * radius && diff < max * radius }.getOrElse(Same)
          slot
        }
      }
      val minOutput = evalState(minHood(nbr(output)))
      val maxOutput = evalState(maxHood(nbr(output)))
      val averageOutput = evalState(excludingSelf.sumHood(nbr(output)) / excludingSelf.reifyField(0).size)
      val recent = recentValues(windowDifferenceSize, State(minOutput, maxOutput, averageOutput))
      History(recent.toList)
    }

    override protected def actionEffect(oldOutput: Double, state: History, action: Action): Double =
      mux(source)(0.0) {
        val result = minHoodPlus(nbr(oldOutput) + nbrRange())
        action match {
          case TemporalGradientRL.ConsiderNeighbourhood => result
          case TemporalGradientRL.Ignore(upVelocity) =>
            oldOutput + upVelocity * (deltaTime().toMillis.toDouble / 1000.0)
        }
      }

    override protected def initialState: History = History(Seq.empty)

    override protected def q: Q[History, Action] = TemporalGradientRL.q

    override def episodeEnd(nodes: Iterable[NodeManager]): Unit = {
      println(":::::CHECK GREEDY POLICY:::::")
      val policy = Policy.greedy[History, Action](actionSet)
      val states = TemporalGradientRL.q.initialConfig.keys.map(_._1)
      println(s"STATE VISITED: ${states.size.toString}")
      /*states
        .map(state => state -> policy(state, q))
        .filterNot { case (state, action) =>
          action == ConsiderNeighbourhood
        }
        .foreach { case (state, action) => println(s"STATE: ${state.toString} ===> ACTION ${action.toString}") }*/
      val storage = new LocalStorage[String]("gradientQ")
      storage.save("q", q)
    }

    override protected def rewardSignal(output: Double): Double = {
      if (((peekReference - output) ~= 0) || (output.isInfinite && peekReference.isInfinite)) {
        0
      } else { -1 }
    }
  }
}

object TemporalGradientRL {
  sealed trait GradientDifference
  case object GreaterBound extends GradientDifference
  case object SmallerBound extends GradientDifference
  case object Same extends GradientDifference
  case class Slot(startMultiplier: Double, endMultiplier: Double) extends GradientDifference

  case class State(
      minDifference: GradientDifference,
      maxDifference: GradientDifference,
      meanDifference: GradientDifference
  )
  case class History(states: Seq[State])

  sealed trait Action
  implicit def ordering: Ordering[Action] = (x: Action, y: Action) =>
    (x, y) match {
      case (ConsiderNeighbourhood, Ignore(_))             => 1
      case (Ignore(_), ConsiderNeighbourhood)             => -1
      case (ConsiderNeighbourhood, ConsiderNeighbourhood) => 0
      case (Ignore(l), Ignore(r))                         => Ordering.Double.IeeeOrdering.compare(l, r)
    }
  case class Ignore(upVelocity: Double) extends Action
  case object ConsiderNeighbourhood extends Action
  val storage = new LocalStorage[String]("gradientQ")
  val q: MutableQ[History, Action] = //new MutableQ[History, Action](Map.empty.withDefault(_ => 0.0))
    new MutableQ(
      storage.load[MutableQ[History, Action]]("q").initialConfig.withDefault(_ => 0.0)
    ) //
  // for the storage
  @SuppressWarnings(Array("org.wartremover.warts.All")) // because of macro expansion
  implicit def storageForGradientDifference: RW[GradientDifference] = macroRW[GradientDifference]
  @SuppressWarnings(Array("org.wartremover.warts.All")) // because of macro expansion
  implicit def storageForSlot: RW[Slot] = macroRW[Slot]
  @SuppressWarnings(Array("org.wartremover.warts.All")) // because of macro expansion
  implicit def storageForAction: RW[Action] = macroRW[Action]
  @SuppressWarnings(Array("org.wartremover.warts.All")) // because of macro expansion
  implicit def storageForIgnore: RW[Ignore] = macroRW[Ignore]
  @SuppressWarnings(Array("org.wartremover.warts.All")) // because of macro expansion
  implicit def storageForState: RW[State] = macroRW[State]
  @SuppressWarnings(Array("org.wartremover.warts.All")) // because of macro expansion
  implicit def storageForHistory: RW[History] = macroRW[History]

}
