package tupol.ml

import scala.util.Random

/**
  *
  */
case class KMeans(clusterCenters: Seq[LabeledPoint]) extends Predictor[Point, LabeledPoint] {

  /**
    * Predict the cluster a point belongs to.
    *
    * @param point
    */
  def predict(point: Point): LabeledPoint = {
    val k = clusterCenters.map(k => (k._1, point.distance2(k._2))).minBy(_._2)._1
    (k, point)
  }
}

object KMeansTrainer {

  def initialize(k: Int, points: Seq[Point]): Seq[LabeledPoint] = {
    require(k > 0 && k < points.size)
    Random.shuffle((0 until points.size).toList)
      .foldLeft(Seq[Point]())((clusters, point) => points(point) +: clusters )
      .take(k).zipWithIndex.map(_.swap).map(t => (t._1.toDouble, t._2))
  }
}

case class KMeansTrainer(k: Int, maxIter: Int, tolerance: Double) extends Trainer[Point, KMeans] {

  import KMeansTrainer._


  /**
    * Train a MeanKs model and return the centroids
    *
    * @param dataPoints
    * @return
    */
  def train(dataPoints: Seq[Point]): KMeans =
    train(initialize(k, dataPoints), dataPoints)

  /**
    * Train a MeanKs model and return the centroids
    *
    * @param initialCentroids
    * @param dataPoints
    * @return
    */
  def train(initialCentroids: Seq[LabeledPoint], dataPoints: Seq[Point]): KMeans = {

    def train(oldCentroids: Seq[LabeledPoint], step: Int, done: Boolean): Seq[LabeledPoint] = {
      if (step == maxIter + 1 || done)
        oldCentroids
      else {
        val newCentroids = run(dataPoints, oldCentroids)
        val done = clustersMovements(oldCentroids, newCentroids).isEmpty
        train(newCentroids, step + 1, done)
      }
    }

    def clustersMovements(oldCentroids: Seq[LabeledPoint], newCentroids: Seq[LabeledPoint]) =
      newCentroids.map { case (k, point) =>
        oldCentroids.toMap.get(k).map(_.distance2(point))
      }.filter(_.isDefined).map(_.get)

    val centroids = train(initialCentroids, 0, false)
    KMeans(centroids)

  }

  private[ml] def run(points: Seq[Point], clusters: Seq[LabeledPoint]): Seq[LabeledPoint] = {
    val pointsByK = KMeans(clusters).predict(points)
    val newClusters = pointsByK.groupBy(_._1).map{ case (k, kfx) => (k, mean(kfx.map(_._2)))}.toSeq
    newClusters
  }

}

