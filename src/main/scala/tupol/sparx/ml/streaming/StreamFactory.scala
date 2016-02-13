package tupol.sparx.ml.streaming

import kafka.serializer.StringDecoder
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.kafka.KafkaUtils

/**
  * Common trait for Stream factories
  */
trait StreamFactory {
  def createStream(ssc: StreamingContext, conf: Configuration): DStream[String]
}

/**
  * File stream factory
  */
object FileStreamFactory extends StreamFactory {
  def createStream(ssc: StreamingContext, conf: Configuration): DStream[String] = {
    ssc.textFileStream(conf.inputPredictionData)
  }
}

/**
  * Kafka stream factory
  */
object KafkaStreamFactory extends StreamFactory {

  def createStream(ssc: StreamingContext, conf: Configuration): DStream[String] = {
    val kafkaParams = Map[String, String]("metadata.broker.list" -> conf.kafkaBrokers)
    KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](ssc, kafkaParams, conf.kafkaTopics.toSet).
      map(_._2)
  }
}
