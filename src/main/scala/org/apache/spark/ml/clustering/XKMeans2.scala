package org.apache.spark.ml.clustering

import org.apache.spark.ml.clustering.evaluation.ClusteringDistanceMetrics
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Calculates the following:
  * - probability
  * - probability by feature
  *
  */
class XKMeans2(override val uid: String) extends Estimator[XKMeans2Model] with XKMeans2Params {

  def this() = this(Identifiable.randomUID("xkmeans2"))

  override def fit(dataset: DataFrame): XKMeans2Model = {

    val rdd = dataset.select(col($(predictionCol)), col($(featuresCol)), col($(distanceToCentroidCol)))
      .map { case Row(cluster: Int, point: Vector, distance: Double) => (cluster, point, distance) }.cache

    val k = rdd.map(_._1).distinct.max

    val statisticalSummaryByCluster = (0 to k)
      .map{ k => Statistics.colStats(rdd.filter(_._1 == k).map(_._2)) }

    val distanceMetrics = new ClusteringDistanceMetrics(rdd.map(x => (x._1, x._3)))

    val model = new XKMeans2Model(uid, distanceMetrics, statisticalSummaryByCluster)

    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): XKMeans2 = defaultCopy(extra)
}

class XKMeans2Model private[ml](
                                override val uid: String,
                                val distanceMetrics: ClusteringDistanceMetrics,
                                val statisticalSummaryByCluster: Seq[MultivariateStatisticalSummary])
  extends Model[XKMeans2Model] with XKMeans2Params {

  override def copy(extra: ParamMap): XKMeans2Model = {
    val copied = new XKMeans2Model(uid, distanceMetrics, statisticalSummaryByCluster)
    copyValues(copied, extra)
  }

  override def transform(dataset: DataFrame): DataFrame = {

    val K = distanceMetrics.byCluster.size
    val featuresNo = statisticalSummaryByCluster(0).variance.size

    val varByFeatures = (0 until featuresNo).map{ f =>
      (0 until K).map{k => statisticalSummaryByCluster(k).variance(f)}}
    val minVariance = varByFeatures.flatten.filter(_ > 0).min
    val minVarianceByFeature = varByFeatures.map{ vs =>
      vs.filter(_ > 0).sorted.headOption match {
        case Some(x) => x
        case None    => minVariance}
    }

    def probabilityByFeature(k: Int, point: Vector) =
    {
      val averages = statisticalSummaryByCluster(k).mean
      val variances = statisticalSummaryByCluster(k).variance

      Vectors.dense((0 until featuresNo).map{ f =>
        val sqerr = (point(f) - averages(f)) * (point(f) - averages(f))
        val variance = if(variances(f) == 0) minVarianceByFeature(f) else variances(f)
        import math._
        val rez = (1 / (sqrt(2 * Pi * variance)) *
          math.exp(-sqerr / (2 * variance))
          )
        rez
      }.toArray)
    }

//    def probability(k: Int, point: Vector) =
//      probabilityByFeature(k, point).toArray.reduce(_ * _)

    def probability(probabilities: Vector) =
      probabilities.toArray.reduce(_ * _)

    val probabilityByFeatureUDF = udf( probabilityByFeature(_: Int, _: Vector) )

    val probabilityUDF = udf( probability(_: Vector) )


    dataset
      .withColumn($(probabilityByFeatureCol), probabilityByFeatureUDF(col($(predictionCol)), col($(featuresCol))))
      .withColumn($(probabilityCol), probabilityUDF(col($(probabilityByFeatureCol))))

  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

}

private [clustering] trait XKMeans2Params extends Params with HasFeaturesCol with HasPredictionCol
with HasDistanceToCentroidCol  with HasProbabilityCol with HasProbabilityByFeature {


  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  /**
    * Validates and transforms the input schema.
    *
    * @param schema input schema
    * @return output schema
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(predictionCol), IntegerType)
    SchemaUtils.appendColumn(schema, $(probabilityCol), DoubleType)
    SchemaUtils.appendColumn(schema, $(probabilityByFeatureCol), new VectorUDT)
  }
}



