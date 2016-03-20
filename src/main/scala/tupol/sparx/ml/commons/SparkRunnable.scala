package tupol.sparx.ml.commons

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.{Logging, SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

/**
  * Trivial trait for running basic Spark apps.
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

  /**
    * The list of required application parameters. If defined they are checked in the `validate()` function
    *
    * @return
    */
  def requiredParameters: Seq[String]

  def run(implicit sc: SparkContext, conf: Config): Any

  /**
    * Any object extending this trait becomes a runnable application.
    *
    * @param args
    */
  def main(implicit args: Array[String]): Unit = {
    val runnableName = this.getClass.getName
    logInfo(s"Running $runnableName")
    val sc = createDefaultSparkContext(runnableName)

    val config = Try(ConfigFactory.parseString(args.mkString("\n")).
      withFallback(ConfigFactory.defaultReference()).getConfig(appName)) match {
      case Success(conf) => conf
      case Failure(ex) =>
        logWarning(s"No configuration defined for application '$appName'; using root configuration.")
        ConfigFactory.defaultReference()
    }

    logInfo(s"$appName: Application Parameters:\n${args.mkString("\n")}")
    logInfo(s"$appName: Configuration:\n${config.root.render()}")
    this.validate(sc, config) match {
      case Success(ok) => this.run(sc, config)
      case Failure(ex) => sys.error(ex.getMessage); sys.exit(-1)
    }
  }

  private def createDefaultSparkContext(runnerName: String) = {
    val defSparkConf = new SparkConf(true)
    val sparkConf = defSparkConf.setAppName(runnerName).
      setMaster(defSparkConf.get("spark.master", "local[*]"))
    new SparkContext(sparkConf)
  }

  /**
    * The default validate function that checks the requiredParameters are at least present.
    *
    * @param sc
    * @param config
    * @return
    */
  def validate(sc: SparkContext, config: Config): Try[String] = {
    requiredParameters.map(param => (param, config.hasPath(param))).filterNot(_._2).
      map(_._1) match {
      case Nil => Success("Ok")
      case missingParams => Failure(new IllegalArgumentException(s"The following required parameters are missing: ${missingParams.mkString(", ")}"))
    }
  }
}
