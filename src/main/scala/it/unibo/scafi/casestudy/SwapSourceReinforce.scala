package it.unibo.scafi.casestudy

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.tiggers.EndHandler
import it.unibo.learning.Reinforce.{ReinforceNetwork, TfReinforceNetwork}
import it.unibo.learning.{Clock, Reinforce}
import it.unibo.scafi.casestudy.LearningProcess.RoundData
import os.{Path, pwd}

import scala.jdk.CollectionConverters.IteratorHasAsScala

class SwapSourceReinforce extends SwapSourceLike with MonteCarloBased {
  override lazy val qId: String = "global"
  val reinforceSeed = 12
  val path: Path = pwd / "qtables" / "reinforce"
  val networkFile: Path = path / "network"
  lazy val network: ReinforceNetwork = {
    os.makeDir.all(path)
    val network =
      new TfReinforceNetwork(
        trajectorySize,
        32 :: 16 :: 8 :: actions.length :: Nil,
        reinforceSeed,
        maxDiff
      )
    if (os.exists(networkFile)) {
      network.load((path / "network").wrapped.toFile)
    } else {
      network.init()
    }
    network.reset()
    network
  }
  @SuppressWarnings(Array("org.wartremover.warts.Any")) // because of unsafe scala binding
  override lazy val endHandler: EndHandler[_] = {
    val storeMonitor = new EndHandler[Any](
      sharedLogic = () => {
        clockTableStorage.save(mid().toString, node.get[Clock]("clock"))
      },
      leaderLogic = () => {
        val reinforceLearning = Reinforce.Plain(actions.toNonEmptyList, gamma)
        println(s"Episodes: ${episode.toString}")
        val nodes = alchemistEnvironment.getNodes.iterator().asScala.toList
        println(s"Population size: ${nodes.size.toString}")
        val managers = nodes.map(new SimpleNodeManager(_))
        val elements = -1 to 1
        (for {
          i <- elements
          j <- elements
        } yield (network.eval(List(i, j)), i, j)).foreach { case (arr, i, j) =>
          println(s"state = (${i.toString}, ${j.toString}), action selection = ${arr.mkString(",")}")
        }
        val trajectories = rand.shuffle(managers.map(node => (node.get[Seq[(State, Action, Double)]]("trajectory"))))
        trajectories.foreach(t => reinforceLearning.improve(t, network, Clock.start))
        network.store(networkFile.wrapped.toFile)
        managers.foreach(a => a.get[ReinforceNetwork]("network").close())
        println("end fit..")
      },
      id = mid()
    )
    alchemistEnvironment.getSimulation.addOutputMonitor(storeMonitor)
    storeMonitor
  }
  lazy val zeroBasedEpsilon = epsilon.value(Clock.start)
  // Aggregate program
  override def aggregateProgram(): RoundData[State, Action, Double] = {
    val classicHopCount = hopGradient(source) // BASELINE
    val hopCountWithoutRightSource =
      hopGradient(mid() == leftSrc) // optimal gradient when RIGHT_SRC stops being a source
    val refHopCount = if (passedTime >= rightSrcStop) hopCountWithoutRightSource else classicHopCount
    // Learning definition
    val problem = learningProblem(refHopCount.toInt)
    // RL Program execution
    val (roundData, trajectory) = {
      problem.actWith(
        learningAlgorithm.ops,
        clock,
        (state, _, _) => network.pass(state)
      )
    }

    val stateOfTheArt = svdGradient()(source = source, () => 1)
    val rlBasedError = refHopCount - roundData.output
    val overEstimate =
      if (rlBasedError > 0) { 1 }
      else { 0 }
    val underEstimate =
      if (rlBasedError < 0) { 1 }
      else { 0 }
    // Store alchemist info
    node.put("overestimate", overEstimate)
    node.put("underestimate", underEstimate)
    node.put("qtable", roundData.q)
    node.put("clock", roundData.clock)
    node.put("classicHopCount", classicHopCount)
    node.put("rlbasedHopCount", roundData.output)
    node.put(s"err_classicHopCount", Math.abs(refHopCount - classicHopCount))
    node.put(s"err_rlbasedHopCount", Math.abs(refHopCount - roundData.output))
    node.put(s"passed_time", passedTime)
    node.put("src", source)
    node.put("action", roundData.action)
    node.put(s"err_flexHopCount", Math.abs(refHopCount - stateOfTheArt))
    node.put("trajectory", trajectory)
    node.put("network", network)
    roundData
  }
}
