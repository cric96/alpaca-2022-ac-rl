package it.unibo.learning

import scala.language.implicitConversions

sealed trait TimeVariable[A] {
  def value(clock: Clock): A
}

object TimeVariable {
  def independent[A](v: A): TimeVariable[A] = new TimeVariable[A] {
    override def value(clock: Clock): A = v
  }
  def follow[A](logic: Clock => A): TimeVariable[A] = new TimeVariable[A] {
    override def value(clock: Clock): A = logic(clock)
  }
  def exponentialDecayFunction(startBy: Double, factor: Double): TimeVariable[Double] =
    new TimeVariable[Double] {
      override def value(clock: Clock): Double = startBy * math.exp(-(clock.ticks / factor))
    }

  def decayByDivision(constant: Double, divisionFactor: Double): TimeVariable[Double] = {
    new TimeVariable[Double] {
      override def value(clock: Clock): Double = constant / math.pow(clock.ticks.toDouble + 1, divisionFactor)
    }
  }
  implicit def varToTimeVar[A](a: A): TimeVariable[A] = TimeVariable.independent(a)
}
