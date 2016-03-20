package tupol.sparx.ml.streaming

import com.typesafe.config.Config
import org.apache.spark.Logging
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.elasticsearch.spark.sql._
import tupol.sparx.ml.streaming.configuration.ElasticSearchConfiguration

import scala.util.{Failure, Try}

/**
  * Common trait for consumers of data
  */
trait Consumer {
  def consume(sqlContext: SQLContext, data: DataFrame, config: Config): Unit
}

/**
  * Consume the data by storing it to ElasticSearch
  */
object ElasticSearchConsumer extends Consumer with Logging {
  def consume(sqlContext: SQLContext, data: DataFrame, config: Config): Unit = {
    // Get the corresponding configuration
    val conf = new ElasticSearchConfiguration(config)
    Try(data.saveToEs(conf.esIndexRoot + s"/${conf.prefix}_predictions_stream")) match {
      case Failure(exception) => println(s"Encountered exception ${exception.getMessage}")
      case _ => ()
    }
  }
}

/**
  * Consume the data by storing it to Cassandra
  */
object CassandraConsumer extends Consumer {
  def consume(sqlContext: SQLContext, data: DataFrame, config: Config): Unit = {
    // Get the corresponding configuration
    throw new Exception("Cassandra Consumer not implemented yet!")
  }
}

