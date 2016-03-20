package org.apache.spark.ml.clustering.evaluation

import org.apache.spark.rdd.RDD

/**
  * Calculate distance statistics, by cluster and for the entire model.
  *
  */
case class ClusteringDistanceMetrics(private val predictions: RDD[(Int, Double)]) {

  // Cache input if necessary
  private[this] val inputNotCached = predictions.getStorageLevel == None
  if(inputNotCached) predictions.persist()

  val byCluster: Seq[DistanceStats] = {
    predictions.groupByKey.
      map { case (clust, dist) =>
        val avgDist = dist.sum / dist.size
        val sse = dist.map(d => d - avgDist).map(d => d * d).sum
        val variance = sse / dist.size
        (clust, DistanceStats(dist.size, dist.min, avgDist, dist.max, sse, variance))
      }.collect.toSeq.sortBy(_._1).map(_._2)
  }

  val byModel: DistanceStats = {
    val count = predictions.count
    val distances = predictions.map(_._2)
    val avgDist = distances.sum / count
    val sse = distances.map(d => d - avgDist).map(d => d * d).sum
    val variance = sse / count
    DistanceStats(
      count,
      byCluster.map(_.min).min,
      avgDist,
      byCluster.map(_.max).max,
      sse,
      variance
    )
  }

  // Release the cache, if necessary
  if(inputNotCached) predictions.unpersist()

}

/**
  * Basic statistics about the distances (minimum, mean, maximum, sse:Sum of Squared Errors and variance)
  *
  * @param min
  * @param avg
  * @param max
  * @param sse
  * @param variance
  */
private[ml] case class DistanceStats(count: Long, min: Double, avg: Double, max: Double, sse: Double, variance: Double)
