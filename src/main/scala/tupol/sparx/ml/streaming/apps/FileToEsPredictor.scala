package tupol.sparx.ml.streaming.apps

import tupol.sparx.ml.streaming.{FileStreamFactory, ElasticSearchConsumer, StreamPredictor}

/**
 * Convenience runnable that reads from a directory and stores to ElasticSearch
 */
object FileToEsPredictor extends StreamPredictor(FileStreamFactory, ElasticSearchConsumer) {

  val requiredParameters: Seq[String] = Seq(
    "prefix",
    "prediction.pipelines",
    "stream.file.directory"
  )

}
