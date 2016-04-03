package tupol.ml

import scala.util.Random

/**
  *
  */
case class KMeans(clusterCenters: Seq[KMeans.LabeledPoint]) {

  import KMeans._

  /**
    * Predict the clusters the points belongs to.
    *
    * @param points
    */
  def predict(points: Seq[Point]): Seq[LabeledPoint] = {
    points.map(predict)
  }

  /**
    * Predict the cluster a point belongs to.
    *
    * @param point
    */
  def predict(point: Point): LabeledPoint = {
    val k = clusterCenters.map(k => (k._1, distance2(k._2, point))).minBy(_._2)._1
    (k, point)
  }


}

object KMeans extends App {

  type Point = Array[Double]
  type LabeledPoint = (Int, Point)

  private[ml] def distance2(vector1: Point, vector2: Point): Double = {
    vector1.zip(vector2).map(x => (x._2 - x._1) * (x._2 - x._1)).sum
  }

  private[ml] def mean(vectors: Seq[Point]): Point = {
    require(vectors.size > 0)
    vectors.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
    .map(x => x / vectors.size)
  }

  def initialize(k: Int, points: Seq[Point]): Seq[LabeledPoint] = {
    require(k > 0 && k < points.size)
    Random.shuffle((0 until points.size).toList)
      .foldLeft(Seq[Point]())((clusters, point) => points(point) +: clusters )
        .take(k).zipWithIndex.map(_.swap)
  }


  /**
    * Train a MeanKs model and return the centroids
    *
    * @param k
    * @param maxIter
    * @param tolerance
    * @param dataPoints
    * @return
    */
  def train(k: Int, maxIter: Int, tolerance: Double, dataPoints: Seq[Point]): Seq[LabeledPoint] =
    train(initialize(k, dataPoints), maxIter, tolerance, dataPoints)

  /**
    * Train a MeanKs model and return the centroids
    *
    * @param initialCentroids
    * @param maxIter
    * @param tolerance
    * @param dataPoints
    * @return
    */
  def train(initialCentroids: Seq[LabeledPoint], maxIter: Int, tolerance: Double, dataPoints: Seq[Point]): Seq[LabeledPoint] = {

    def train(oldCentroids: Seq[LabeledPoint], step: Int, done: Boolean): Seq[LabeledPoint] = {
      if (step == maxIter || done)
        oldCentroids
      else {
        val newCentroids = run(dataPoints, oldCentroids)
        val done = clustersMovements(oldCentroids, newCentroids).isEmpty
        train(newCentroids, step + 1, done)
      }
    }

    def clustersMovements(oldCentroids: Seq[LabeledPoint], newCentroids: Seq[LabeledPoint]) =
      newCentroids.map { case (k, point) =>
        oldCentroids.toMap.get(k).map(distance2(_, point))
      }.filter(_.isDefined).map(_.get)

    train(initialCentroids, 0, false)

  }

  private[ml] def run(points: Seq[Point], clusters: Seq[LabeledPoint]): Seq[LabeledPoint] = {
    val pointsByK = KMeans(clusters).predict(points)
    val newClusters = pointsByK.groupBy(_._1).map{ case (k, kfx) => (k, mean(kfx.map(_._2)))}.toSeq
    newClusters
  }



}
