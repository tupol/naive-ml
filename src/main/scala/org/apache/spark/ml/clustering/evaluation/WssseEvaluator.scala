package org.apache.spark.ml.clustering.evaluation

import org.apache.spark.ml.clustering.KMeansParams
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.shared.HasDistanceToCentroidCol
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Evaluates a KMeans model based on WSSSE
  */
class WssseEvaluator(override val uid: String)
  extends Evaluator with KMeansParams with HasDistanceToCentroidCol {

  def this() = this(Identifiable.randomUID("wssseEval"))

  /**
    * param for metric name in evaluation
    * Default: averageWSSSE
    *
    * @group param
    */
  val metricName: Param[String] = {
    val allowedParams = ParamValidators.inArray(Array("WSSSE", "averageWSSSE"))
    new Param(
      this, "metricName", "metric name in evaluation (WSSSE|averageWSSSE)", allowedParams)
  }


  /** @group getParam */
  def getMetricName: String = $(metricName)

  /** @group setParam */
  def setMetricName(value: String): this.type = set(metricName, value)

  setDefault(metricName -> "averageWSSSE")

  override def evaluate(dataset: DataFrame): Double = {
    // TODO Implement using just DataFRames, no RDDs
    val distances = dataset.select($(distanceToCentroidCol))
      .map { case Row(distance: Double) =>
        (distance)
      }
    val count = dataset.count()
    val average = distances.sum / count
    val wssse = distances.map(dist => (dist - average)).map(d => d*d).sum
    val avgWssse = wssse / count // a.k.a. variance
    val metric = $(metricName) match {
      case "WSSSE" => wssse
      case "averageWSSSE" => avgWssse
    }
    metric
  }

  override def isLargerBetter: Boolean = false

  override def copy(extra: ParamMap): WssseEvaluator = defaultCopy(extra)
}
