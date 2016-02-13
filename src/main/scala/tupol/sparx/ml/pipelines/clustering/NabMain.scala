package tupol.sparx.ml.pipelines.clustering

import com.typesafe.config.Config
import org.apache.spark.SparkContext
import tupol.sparx.ml.commons.SparkRunnable

import scala.util.{Success, Try}

/**
  * This is mainly useful to show the flow when starting from scratch.
  */
object NabMain extends SparkRunnable {

  val appName = NAB

  def run(implicit sc: SparkContext, config: Config) = Main.run(appName)

  // TODO implement a configuration validation
  override def validate(implicit sc: SparkContext, config: Config): Try[Boolean] = Success(true)

}
