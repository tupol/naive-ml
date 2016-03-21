package org.apache.spark.ml.clustering.evaluation

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Calculate distance statistics, by cluster and for the entire model.
  *
  */
case class ClusterStatistics(private val predictions: RDD[(Int, Vector)]) {

  // Cache input if necessary
  private[this] val inputNotCached = predictions.getStorageLevel == None
  if(inputNotCached) predictions.persist()

//  val metricsByCluster: Seq[DistanceStats] = {
//    predictions.groupByKey.
//      map { case (clust, dist) =>
//        val avgDist = dist.sum / dist.size
//        val sse = dist.map(d => d - avgDist).map(d => d * d).sum
//        val variance = sse / dist.size
//        (clust, DistanceStats(dist.size, dist.min, avgDist, dist.max, sse, variance))
//      }.collect.toSeq.sortBy(_._1).map(_._2)
//  }
//
//  val metricsByModel: DistanceStats = {
//    val count = predictions.count
//    val distances = predictions.map(_._2)
//    val avgDist = distances.sum / count
//    val sse = distances.map(d => d - avgDist).map(d => d * d).sum
//    val variance = sse / count
//    DistanceStats(
//      count,
//      metricsByCluster.map(_.min).min,
//      avgDist,
//      metricsByCluster.map(_.max).max,
//      sse,
//      variance
//    )
//  }

  // Release the cache, if necessary
  if(inputNotCached) predictions.unpersist()

}


