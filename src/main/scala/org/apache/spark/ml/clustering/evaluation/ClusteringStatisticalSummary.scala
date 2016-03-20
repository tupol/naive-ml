package org.apache.spark.ml.clustering.evaluation

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

/**
  * Calculate distance statistics, by cluster and for the entire model.
  *
  */
case class ClusteringStatisticalSummary(private val predictions: RDD[(Int, Vector)]) {

  // Cache input if necessary
  private[this] val inputNotCached = predictions.getStorageLevel == None
  if(inputNotCached) predictions.persist()

  private val k = predictions.map(_._1).distinct.max

  val byModel = Statistics.colStats(predictions.map(_._2))

  val byCluster = (0 to k)
    .map{ k => Statistics.colStats(predictions.filter(_._1 == k).map(_._2)) }

  // Release the cache, if necessary
  if(inputNotCached) predictions.unpersist()

}


