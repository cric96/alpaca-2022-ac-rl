package it.unibo.scafi.casestudy

import it.unibo.learning.QLearning

trait SarsaBased {
  self: SwapSourceLike =>
  lazy val learningAlgorithm: QLearning.Type[List[Action], Action] =
    QLearning.Hysteretic[List[Int], Int](actions, alpha, beta, gamma)
}
