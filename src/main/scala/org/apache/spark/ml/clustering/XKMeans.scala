package org.apache.spark.ml.clustering

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasDistanceToCentroidCol, HasProbabilityCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}

/**
  * An extension of the existing KMeans transformer that calculates the distance to cluster for each entry in addition
  * to the prediction (cluster).
  */
class XKMeans(override val uid: String) extends KMeans with HasDistanceToCentroidCol {

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
    val model = new XKMeansModel(uid, parentModel)
    copyValues(model)
  }

}

class XKMeansModel private[ml](
                                override val uid: String,
                                private val parentModel: MLlibKMeansModel)
  extends KMeansModel(uid, parentModel) with HasDistanceToCentroidCol with HasProbabilityCol {

  override def copy(extra: ParamMap): XKMeansModel = {
    val copied = new XKMeansModel(uid, parentModel)
    copyValues(copied, extra)
  }

  override def transform(dataset: DataFrame): DataFrame = {

    val predictUDF = udf((vector: Vector) => predict(vector))

    val distanceUDF = udf((vector: Vector, k: Int) =>
      Vectors.sqdist(vector, parentModel.clusterCenters(k)))

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
      .withColumn($(distanceToCentroidCol), distanceUDF(col($(featuresCol)), col($(predictionCol))))

  }

}


