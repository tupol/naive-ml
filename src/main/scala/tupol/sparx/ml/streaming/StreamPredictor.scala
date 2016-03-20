package tupol.sparx.ml.streaming

import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext}
import tupol.sparx.ml.commons.SparkRunnable
import tupol.sparx.ml.commons.ml._
import tupol.sparx.ml.streaming.configuration.CommonConfiguration



/**
  * Consume a stream of strings and transform it, run the predictions and consume the prediction.
  *
  * TODOs:
  * [ ] figure out how to load and hot-swap the KMeans models (e.g. over time we might want to swap the KMeans
  * while the stream is still on;
  * [ ] figure out how to persist the predictions and what should go in there (e.g. predicted cluster,
  * distance to centroid, original input data, KMeans model version/id/timestamp);
  * [ ] figure out how to train, monitor training behavior and save the models trained dynamically
  */
abstract class StreamPredictor(streamBuilder: StreamFactory, consumer: Consumer) extends SparkRunnable {

  val appName = "app.streaming"

  def run(implicit sc: SparkContext, config: Config) = {

    val conf = new CommonConfiguration(config)

    val pipelines = conf.pipelines(sc)

    if (pipelines.isEmpty) {
      //TODO Figure out if sys.exit is ok for a spark app, maybe not
      sys.error("No valid pipelines specified.")
      sys.exit(-1)
    }

    val ssc = new StreamingContext(sc, Seconds(conf.batchDurationSeconds))

    val rawInputRdd = streamBuilder.createStream(ssc, config)

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    // Initialize the streaming kMeans predictors
    pipelines.foreach { pipeline =>
      rawInputRdd.foreachRDD { rdd =>
        val data = rdd.map(in => (java.time.Instant.now.toString, in)).
          toDF(conf.rawTimestampColName, conf.rawDataColName)
        val evaluatedData = pipeline.transform(data).
          select(conf.rawTimestampColName, pipeline.exportableColumns: _*).
          withColumn("PIPELINE_ID", lit(pipeline.uid))
        // TODO Move the "PIPELINE_ID" to a constant, configuration param...
        // TODO Maybe create a "meta-data" transformer that can add all the info regarding the pipeline used in the output records.

        // Consume the data with the specified consumer
        consumer.consume(sqlContext, evaluatedData, config)
      }
    }

    ssc.start
    ssc.awaitTermination

  }

}
