package tupol.sparx.ml.streaming

import tupol.sparx.ml.commons.io._
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel

import scala.util.{Failure, Success}

/**
  * Convenience wrapper around the Config object.
  */
case class Configuration(conf: Config) {

  val prefix: String = conf.getString("prefix")

  val rawDataColName: String = conf.getString("raw.input.col.name")
  val rawTimestampColName: String = conf.getString("raw.timestamp.col.name")

  // Prediction configuration section
  val batchDurationSeconds: Int = conf.getInt("stream.batch.duration.seconds")

  val inputPredictionData: String = conf.getString("stream.file.directory")

  def pipelines(implicit sc: SparkContext) = conf.getString("prediction.pipelines").split(",").map(_.trim).
    flatMap { file =>
        loadObjectFromHdfsFile[PipelineModel](sc, file.trim) match
        {
          case Success(m) => println(s"Successfully loaded pipeline from $file"); Some(m)
          case Failure(x) => sys.error(s"Failed to load pipeline from $file.\n${x.getLocalizedMessage}"); None
        }
      }

  val esIndexRoot = conf.getString("es.index.root")

  val kafkaBrokers = conf.getString("stream.kafka.brokers")

  val kafkaTopics = conf.getString("stream.kafka.topics").split(",").map(_.trim)


}


