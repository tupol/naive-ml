package tupol.sparx.ml.commons.ml

import com.typesafe.config.Config
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.{Logging, SparkContext}
import tupol.sparx.ml.commons.SparkRunnable
import tupol.sparx.ml.commons.io._
import tupol.sparx.ml.pipelines.clustering.Configuration

import scala.util.{Success, Try}

/**
  * Predictor for the given app.
  */
class PredictXKMeans(val appName: String) extends SparkRunnable with Logging {

  def run(implicit sc: SparkContext, config: Config) = {

    val conf = new Configuration(config)

    val sqc = new SQLContext(sc)

    import sqc.implicits._

    logInfo(s"$appName: Running prediction on ${conf.inputPredictionData} data file using ${conf.pipeline} pipeline model.")

    val data = sc.textFile(conf.inputPredictionData).
      map(in => (java.time.Instant.now.toString, in)).
      toDF(conf.rawTimestampColName, conf.rawDataColName)

    val pipeline = loadObjectFromHdfsFile[PipelineModel](sc, conf.pipeline).get

    val predictions = pipeline.transform(data).
      withColumn("PIPELINE_ID", lit(pipeline.uid))
    // TODO Move the "PIPELINE_ID" to a constant, configuration param...
    // TODO Maybe create a "meta-data" transformer that can add all the info regarding the pipeline used in the output records.
    // TODO Wait a little for Spark 1.6.0 that brings in support for some meta-data in the pipeline

    logInfo(s"$appName: Selecting anomalies: all records having the probability < ${conf.threshold}. ")
    val anomalies = predictions.where(f"probability < ${conf.threshold}%6.4f" )



    logInfo(s"${anomalies.count} anomalies found out of ${predictions.count} analysed records.")

    val outFile = f"${conf.outputPath}/${conf.prefix}_${pipeline.uid}_anomalies"
    logInfo(s"$appName: Saving anomalies records to $outFile.")
    removeHdfsFile(outFile)
    // Save only the exportable columns
    anomalies.select(conf.rawTimestampColName, pipeline.exportableColumns : _*).
      write.json(outFile)

  }

  // TODO implement a configuration validation
  override def validate(implicit sc: SparkContext, config: Config): Try[Boolean] = Success(true)

}




