package tupol.sparx.ml.streaming.apps

import tupol.sparx.ml.streaming.{ElasticSearchConsumer, KafkaStreamFactory, StreamPredictor}

/**
 * Convenience runnable that reads from Kafka and stores to ElasticSearch
 */
object KafkaToEsPredictor extends StreamPredictor(KafkaStreamFactory, ElasticSearchConsumer) {

  val requiredParameters: Seq[String] = Seq(
    "prefix",
    "prediction.pipelines",
    "stream.kafka.brokers",
    "stream.kafka.topics"
  )

}
