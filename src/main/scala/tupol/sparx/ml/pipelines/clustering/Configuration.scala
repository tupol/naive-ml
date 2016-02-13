package tupol.sparx.ml.pipelines.clustering

import com.typesafe.config.{Config, ConfigRenderOptions}
import tupol.sparx.ml.commons._

/**
  * Convenience wrapper around the Config object.
  */
class Configuration(val config: Config) {

  // General application configuration section
  val prefix: String = config.getString("prefix")
  val workPath: String = config.getString("path.wip")
  val outputPath: String = config.getString("path.output")
  val rawDataColName: String = config.getString("raw.input.col.name")
  val rawTimestampColName: String = config.getString("raw.timestamp.col.name")
  val preparedDataColName: String = config.getString("prepared.data.col.name")

  // Prediction configuration section
  val threshold: Double = config.getDouble("prediction.threshold")
  val pipeline: String = config.getString("prediction.pipeline")
  val inputPredictionData: String = config.getString("prediction.file")

  // Training configuration section
  val inputSchema: String = config.getValue("training.input.schema").render(ConfigRenderOptions.concise)
  val splitRatio: Double = config.getDouble("training.split.ratio")
  val splitSeed: Int = config.getInt("training.split.seed")
  val inputTrainingData: String = config.getString("training.file")
  val trainDataSplitPath = workPath + s"/${prefix}_data_train.parquet"
  val testDataSplitPath = workPath + s"/${prefix}_data_test.parquet"

  // KMeans configuration section
  val clusters: Seq[Int] = parseStringToRange(config.getString("training.kmeans.clusters"))
  val iterations: Seq[Int] = parseStringToRange(config.getString("training.kmeans.iterations"))
  val tolerances: Seq[Double] = parseStringToRange(config.getString("training.kmeans.tolerances")).map(x => math.pow(10, -x))
  val folds: Int = config.getInt("training.kmeans.folds")

}
