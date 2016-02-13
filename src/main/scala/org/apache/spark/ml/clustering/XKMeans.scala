package org.apache.spark.ml.clustering

import org.apache.spark.ml.clustering.evaluation.ClusteringDistanceMetrics
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasProbabilityCol, HasDistanceToCentroidCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Calculates the following:
  * - cluster (prediction)
  * - distance to cluster
  * - probability
  *
  * TODO: Get the probability calculation part into a different class to make it more flexible
  * (easier to plugin in different probability calculation algorithms)
  */
class XKMeans(override val uid: String) extends KMeans with HasDistanceToCentroidCol with HasProbabilityCol {

  def this() = this(Identifiable.randomUID("xkmeans"))

  override def fit(dataset: DataFrame): XKMeansModel = {

    val rdd = dataset.select(col($(featuresCol))).map { case Row(point: Vector) => point }.cache

    val algo = new MLlibKMeans()
      .setK($(k))
      .setInitializationMode($(initMode))
      .setInitializationSteps($(initSteps))
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
      .setEpsilon($(tol))

    val parentModel = algo.run(rdd)
    val predictions = rdd.zip(parentModel.predict(rdd)).
      map { case (vec, k) => (k, Vectors.sqdist(vec, parentModel.clusterCenters(k))) }
    val metrics = new ClusteringDistanceMetrics(predictions)
    val model = new XKMeansModel(uid, parentModel, metrics)
    copyValues(model)
  }

}

class XKMeansModel private[ml](
                                override val uid: String,
                                private val parentModel: MLlibKMeansModel,
                                val metrics: ClusteringDistanceMetrics)
  extends KMeansModel(uid, parentModel) with HasDistanceToCentroidCol with HasProbabilityCol {

  override def copy(extra: ParamMap): XKMeansModel = {
    val copied = new XKMeansModel(uid, parentModel, metrics)
    copyValues(copied, extra)
  }

  override def transform(dataset: DataFrame): DataFrame = {
    val predictUDF = udf((vector: Vector) => predict(vector))

    val distanceUDF = udf((vector: Vector, k: Int) =>
      Vectors.sqdist(vector, parentModel.clusterCenters(k)))

    // TODO This is not the correct formula, but it will do for now
    // Normally this is done for each feature in the cluster rather than for the total distances
    // When improving this, and you don't know where to start, ask Oliver
    val probabilityUDF = udf((k: Int, distance: Double) => {
      val metK = metrics.metricsByCluster(k)
      val sqerr = (distance - metK.avg) * (distance - metK.avg)
      import math._
      (1 / (sqrt(2 * Pi * metK.variance)) *
        math.exp(-sqerr / (2 * metK.variance))
        )
    }
    )

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
      .withColumn($(distanceToCentroidCol), distanceUDF(col($(featuresCol)), col($(predictionCol))))
      .withColumn($(probabilityCol), probabilityUDF(col($(predictionCol)), col($(distanceToCentroidCol))))

  }

}


