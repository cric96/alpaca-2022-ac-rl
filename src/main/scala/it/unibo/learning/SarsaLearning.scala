package it.unibo.learning

import cats.data.NonEmptySet
import cats.syntax.all._
import it.unibo.learning.ReinforcementLearning.Ops
import monocle.syntax.all._

import scala.util.Random

object SarsaLearning {
  case class NStepState[S, A](q: Q[S, A], clock: Clock, trajectory: Vector[(S, A, Double, S)])
  case class SarsaState[S, A](q: Q[S, A], oldTrajectory: Option[(S, A, Double, S)])
  trait Type[S, A] extends Sars.Type[S, A, SarsaState[S, A]] {
    override val ops: Ops[S, A, SarsaState[S, A]] = new Ops[S, A, SarsaState[S, A]] {
      override def extractQFromTarget(target: SarsaState[S, A]): Q[S, A] = target.q
      override def initTargetFromQ(q: Q[S, A]): SarsaState[S, A] = SarsaState(q, None)
    }
  }
  // TODO check
  case class NStep[S, A](actions: NonEmptySet[A], alpha: TimeVariable[Double], gamma: Double, trajectorySize: Int)
      extends Sars.Type[S, A, NStepState[S, A]] {
    override def improve(trajectory: (S, A, Double, S), target: NStepState[S, A], clock: Clock)(implicit
        rand: Random
    ): NStepState[S, A] = {
      val trajectories = (trajectory +: target.trajectory).take(trajectorySize + 1)
      val updateQ = if (trajectories.size >= (trajectorySize + 1)) {
        val expectedReward = trajectories.map { case (_, _, reward, _) => reward }.zipWithIndex.map { case (data, i) =>
          math.pow(gamma, i) * data
        }.sum
        val (stateT, actionT, _, _) = trajectories.lastOption.getOrElse(trajectory)
        val oldValue = target.q(stateT, actionT)
        val updateQ = target.q.update(stateT, actionT, oldValue + alpha.value(clock) * (expectedReward - oldValue))
        target.focus(_.q).replace(updateQ)
      } else {
        target
      }
      updateQ
        .focus(_.trajectory)
        .replace(trajectories)
        .focus(_.clock)
        .modify(_.tick)
    }

    override val ops: Ops[S, A, NStepState[S, A]] = new Ops[S, A, NStepState[S, A]] {
      override def extractQFromTarget(target: Aux): Q[S, A] = target.q
      override def initTargetFromQ(q: Q[S, A]): Aux = NStepState(q, Clock.start, Vector.empty)
    }
  }
  // TODO check
  case class Plain[S, A](actions: NonEmptySet[A], alpha: TimeVariable[Double], gamma: Double) extends Type[S, A] {
    override def improve(trajectory: (S, A, Double, S), target: SarsaState[S, A], clock: Clock)(implicit
        rand: Random
    ): SarsaState[S, A] = {
      val (stateTPlus, actionTPlus, _, _) = trajectory
      (target match {
        case SarsaState(_, None) => target
        case SarsaState(q, Some(oldTrajectory)) =>
          val (stateT, actionT, rewardTPlus, _) = oldTrajectory
          val oldQValue = q(stateT, actionT)
          val nextValue = q(stateTPlus, actionTPlus)
          val update = oldQValue + alpha.value(clock) * rewardTPlus + gamma * nextValue - oldQValue
          target.focus(_.q).modify(_.update(stateT, actionT, update))
        case other => other
      }).focus(_.oldTrajectory)
        .replace(Some(trajectory))
    }
  }
  // TODO to check
  case class Expected[S, A](actions: NonEmptySet[A], alpha: TimeVariable[Double], gamma: Double)
      extends QLearning.Type[S, A] {
    override def improve(trajectory: (S, A, Double, S), target: Q[S, A], clock: Clock)(implicit
        rand: Random
    ): Q[S, A] = {
      val (stateT, actionT, rewardTPlus, stateTPlus) = trajectory
      val oldQValue = target(stateT, actionT)
      val prob = 1.0 / actions.length
      val expected = actions.map(action => target(stateTPlus, action)).map(_ * prob).sumAll
      val update = oldQValue + alpha.value(clock) * rewardTPlus + gamma * expected - oldQValue
      target.update(stateT, actionT, update)
    }
  }
}
