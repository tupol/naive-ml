package tupol.sparx.ml.pipelines.clustering

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.SparkContext
import tupol.sparx.ml.commons.SparkRunnable

import scala.util.{Success, Try}

/**
  * This is mainly useful to show the flow when starting from scratch.
  *
  * This is also useful when one want to run just a small part.
  */
object Main extends SparkRunnable {

  // Temporary preserve the KDD as our main app
  val appName = NAB

  def run(implicit sc: SparkContext, config: Config) = run(appName)

  def run(appName: String)(implicit sc: SparkContext, config: Config): Any = {
    // Get the best model output path from
    val outputModelPath = TrainXKMeans(appName).run(sc, config)

    logDebug(s"$appName: Set configuration parameter: $appName.prediction.pipeline=$outputModelPath")

    //Prepare the configuration for the prediction (a bit nasty... but it will do for now)
    val predictionConf = ConfigFactory.
      parseString(s"""prediction.pipeline="$outputModelPath"""").
      withFallback(config)

    // Run the prediction
    PredictXKMeans(appName).run(sc, predictionConf)

  }

  // TODO implement a configuration validation
  override def validate(implicit sc: SparkContext, config: Config): Try[Boolean] = Success(true)

}
