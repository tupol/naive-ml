package tupol.sparx.ml.streaming

import com.typesafe.config.Config
import tupol.sparx.ml.commons.io._
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel

import scala.util.{ Failure, Success }

object configuration {

  /**
    * Convenience wrapper around the Config object containing common configuration parameters for all stages.
    */
  class CommonConfiguration(config: Config) {

    // General application configuration section
    val prefix: String = config.getString("prefix")
    val rawDataColName: String = config.getString("raw.input.col.name")
    val rawTimestampColName: String = config.getString("raw.timestamp.col.name")

    // Prediction configuration section
    val batchDurationSeconds: Int = config.getInt("stream.batch.duration.seconds")

    def pipelines(sc: SparkContext) = config.getString("prediction.pipelines").split(",").map(_.trim).
      flatMap { file =>
        loadObjectFromHdfsFile[PipelineModel](sc, file) match {
          case Success(m) =>
            println(s"Successfully loaded pipeline from $file"); Some(m)
          case Failure(x) => sys.error(s"Failed to load pipeline from $file.\n${x.getLocalizedMessage}"); None
        }
      }

  }

  /**
    * Convenience wrapper around the Config object containing KafkaToEsPredictor specific configuration parameters.
    */
  class KafkaStreamConfiguration(config: Config) extends CommonConfiguration(config: Config) {
    val kafkaBrokers = config.getString("stream.kafka.brokers")
    val kafkaTopics = config.getString("stream.kafka.topics").split(",").map(_.trim)
  }

  /**
    * Convenience wrapper around the Config object containing KafkaToEsPredictor specific configuration parameters.
    */
  class FileStreamConfiguration(config: Config) extends CommonConfiguration(config: Config) {
    val inputPredictionData: String = config.getString("stream.file.directory")
  }

  /**
    * Convenience wrapper around the Config object containing FileToEsPredictor specific configuration parameters.
    */
  class ElasticSearchConfiguration(config: Config) extends CommonConfiguration(config: Config) {
    val esIndexRoot = config.getString("es.index.root")
  }

}
