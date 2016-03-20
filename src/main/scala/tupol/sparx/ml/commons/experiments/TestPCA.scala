package tupol.sparx.ml.commons.experiments

import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import tupol.sparx.ml.commons.SparkRunnable
import tupol.sparx.ml.commons.experiments.configuration.PredictionConfiguration
import tupol.sparx.ml.commons.io._

/**
  *
  */
object TestPCA extends SparkRunnable {

  val appName = "app.kdd"

  val requiredParameters = Seq(
    "prefix",
    "path.wip",
    "path.output",
    "prediction.file",
    "prediction.pipeline"
  )

  def run(implicit sc: SparkContext, config: Config) = {

    val conf = new PredictionConfiguration(config)

    val sqc = new SQLContext(sc)

    import sqc.implicits._

    logInfo(s"$appName: Running prediction on ${conf.inputPredictionData} data file using ${conf.pipeline} pipeline model.")

    val data = sc.textFile(conf.inputPredictionData).
      map(in => (java.time.Instant.now.toString, in)).
      toDF(conf.rawTimestampColName, conf.rawDataColName)

    val pipeline = loadObjectFromHdfsFile[PipelineModel](sc, conf.pipeline).get

    val predictions = pipeline.transform(data).
      withColumn("PIPELINE_ID", lit(pipeline.uid))

    val pca = new PCA().
      setK(3).
      setInputCol(conf.preparedDataColName).
      setOutputCol("pca").
      fit(predictions)

    val projectedData = pca.transform(predictions)

    val lines = projectedData.select("prediction", "pca").map(r => r.getInt(0) + "," + r.getAs[Vector](1).toArray.mkString(",")).collect

    saveLinesToFile(lines, "/tmp/projected-data.csv")
  }

}
