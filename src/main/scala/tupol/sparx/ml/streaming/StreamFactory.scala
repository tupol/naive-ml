package tupol.sparx.ml.streaming

import com.typesafe.config.Config
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.dstream.DStream
import tupol.sparx.ml.streaming.configuration.{FileStreamConfiguration, KafkaStreamConfiguration}

/**
  * Common trait for Stream factories
  */
trait StreamFactory {
  def createStream(ssc: StreamingContext, config: Config): DStream[String]
}

/**
  * File stream factory
  */
object FileStreamFactory extends StreamFactory {
  def createStream(ssc: StreamingContext, config: Config): DStream[String] = {
    // Get the corresponding configuration
    val conf = new FileStreamConfiguration(config)
    ssc.textFileStream(conf.inputPredictionData)
  }
}

/**
  * Kafka stream factory
  */
object KafkaStreamFactory extends StreamFactory {
  import kafka.serializer.StringDecoder
  import org.apache.spark.streaming.kafka._

  def createStream(ssc: StreamingContext, config: Config): DStream[String] = {
    // Get the corresponding configuration
    val conf = new KafkaStreamConfiguration(config)
    val kafkaParams = Map[String, String]("metadata.broker.list" -> conf.kafkaBrokers)
    KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](ssc, kafkaParams, conf.kafkaTopics.toSet).
      map(_._2)
  }
}
