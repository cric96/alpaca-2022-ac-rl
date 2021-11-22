
// Dependency
import $ivy.`com.lihaoyi::os-lib:0.7.8`
import $ivy.`com.github.tototoshi::scala-csv:1.3.8`
import $ivy.`me.shadaj::scalapy-core:0.5.0`
// Proc
import ammonite.ops._
import ammonite.ops.ImplicitWd._
// CSV
import com.github.tototoshi.csv._
// Python deps
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import java.io.File
import scala.collection.Factory

implicit object MyFormat extends DefaultCSVFormat { override val delimiter = ' ' }

@main
def averageByIndicies(skip: Int, experimentName: String, indices: Int*): Any = {
  val workingDir = os.pwd / "data"
  val imageOutputDir = (workingDir / "imgs")
  val imageOutputDirPy = imageOutputDir.wrapped.toAbsolutePath.toString

  if(!os.exists(imageOutputDir)){ os.makeDir(imageOutputDir) }

  val orderedExperiments =  os.list(os.pwd / "data")
    .filter(os.isFile)
    .filter(_.toString().contains(experimentName))
    .sortBy(_.wrapped.toString.split("-").last.toDouble)
  
  val experiments = orderedExperiments
    .map(file => file.wrapped.toAbsolutePath.toString)
    .map(new File(_))
    .map(CSVReader.open)
    .map(_.all())
    .map(file => file.filter(row => row.forall(!_.contains("#"))))
    .drop(skip)
  
  val selectedIndicies = experiments
    .map(
      experiment => experiment.map(row => select(row, indices:_*)).map(row => row.map(_.toDouble))
    ).map(
      experiment => 
        experiment.reduce((acc, data) => acc.zip(data).map { case (a, b) => a + b} ).map(data => data / experiment.size)
    )
    

  val plotIndicies = experiments.indices.toList
  val plots = selectedIndicies
    .map(element => element.map(List(_)))
    .reduce((acc, data) => acc.zip(data).map { case (acc, data) => acc ::: data})
    .map(_.toPythonCopy)
    .toSeq
  
  // Python part
  val plt = py.module("matplotlib.pyplot")
  py.module("matplotlib").rc("figure", figsize = (7, 2))
  val steps = plotIndicies.toPythonCopy
  plots.foreach(plot => plt.plot(plot))
  //plt.show()
  plt.ylabel("average error")
  plt.xlabel("episodes")
  plt.title("training errors")
  plt.savefig(s"$imageOutputDirPy/mean-error.pdf")
  plt.savefig(s"$imageOutputDirPy/mean-error.png")
  "Ok"
}
// Utility function
def select[A, F[a] <: Seq[a]](seq: F[A], indicies: Int*)(implicit factory: Factory[A, F[A]]): F[A] = {
  seq.zipWithIndex.filter { case (_, i) => indicies.contains(i) }.map(_._1).to(factory)
}