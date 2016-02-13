package tupol.sparx.ml.streaming

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.elasticsearch.spark.sql._

import scala.util.Try

/**
  * Common trait for consumers of data
  */
trait Consumer {
  def consume(sqlContext: SQLContext, data : DataFrame, conf: Configuration): Unit
}

/**
  * Consume the data by storing it to ElasticSearch
  */
object ElasticSearchConsumer extends Consumer {
  def consume(sqlContext: SQLContext, data : DataFrame, conf: Configuration): Unit = {
    Try{
      data.saveToEs(conf.esIndexRoot + s"/${conf.prefix}_predictions_stream")
    }
  }
}

/**
  * Consume the data by storing it to Cassandra
  */
object CassandraConsumer extends Consumer {
  def consume(sqlContext: SQLContext, data : DataFrame, conf: Configuration): Unit = {
    throw new Exception("Cassandra Consumer not implemented yet!")
  }
}

