package tupol.sparx.ml.pipelines.clustering

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.SparkContext
import tupol.sparx.ml.commons.SparkRunnable
import tupol.sparx.ml.commons.ml.{PredictXKMeans, TrainXKMeans, PreProcessorBuilder}

/**
 * This is mainly useful to show the flow when starting from scratch.
 *
 * This is also useful when one want to run just a small part.
 */
abstract class AbstractMain extends SparkRunnable {

  def preProcessorBuilder: PreProcessorBuilder

  val requiredParameters = Seq(
    "prefix",
    "path.wip",
    "path.output",
    "training.file",
    "prediction.file"
  )

  def run(implicit sc: SparkContext, config: Config) = run(sc, config, appName)

  def run(implicit sc: SparkContext, config: Config, appName: String): Any = {
    // Get the best model output path from
    val outputModelPath = new TrainXKMeans(appName, preProcessorBuilder).run(sc, config)

    logDebug(s"$appName: Set configuration parameter: $appName.prediction.pipeline=$outputModelPath")
    //Prepare the configuration for the prediction (a bit nasty... but it will do for now)
    val predictionConf = ConfigFactory.
      parseString(s"""prediction.pipeline="$outputModelPath"""").
      withFallback(config)

    // Run the prediction
    new PredictXKMeans(appName).run(sc, predictionConf)

  }

}
