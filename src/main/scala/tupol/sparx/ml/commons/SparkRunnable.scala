package tupol.sparx.ml.commons

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.{Logging, SparkConf, SparkContext}


import scala.util.Try


/**
 * Trivial trait for running basic Spark apps, both as stand alone apps and as Spark JobServer jobs
 *
 */
trait SparkRunnable extends Logging {

  /**
    * This is the key for basically choosing a certain app and it should have
    * the form of 'app.....', reflected also in the configuration structure.
    *
    * @return
    */
  def appName: String

  def run(implicit sc: SparkContext, config: Config): Any

  def validate(implicit sc: SparkContext, config: Config): Try[Boolean]

  // Run as app
  def main(implicit args: Array[String]): Unit = {
    val runnableName = this.getClass.getName
    logInfo(s"Running $runnableName")
    implicit val sc = createDefaultSparkContext(runnableName)
    implicit val config = ConfigFactory.parseString(args.mkString("\n")).
      withFallback(ConfigFactory.defaultReference()).getConfig(appName)
    logInfo(s"$appName: Configuration:\n ${config.root.render()}")
    this.run
  }

  private def createDefaultSparkContext(runnerName: String) = {
    val defSparkConf = new SparkConf(true)
    val sparkConf = defSparkConf.setAppName(runnerName).
      setMaster(defSparkConf.get("spark.master", "local[*]"))

    new SparkContext(sparkConf)
  }
}
