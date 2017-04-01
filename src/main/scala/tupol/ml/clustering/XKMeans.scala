package tupol.ml.clustering

import tupol.ml._
import tupol.ml.pointops._
import tupol.ml.stats.Stats
import tupol.ml.stats.statsops._

import scala.collection.parallel.ParSeq
import scala.util.Random

case class XKMeansCxPrediction(k: Int, distance: Double, probability: Double, probabilityByDimension: Point)

/**
 * XKMeans predictor
 */
case class XKMeans(clusterCenters: Seq[Stats[Point]]) extends Predictor[Point, XKMeansCxPrediction] {

  /**
   * Predict the cluster a point belongs to.
   *
   */
  override def predict(data: Point): XKMeansCxPrediction = {

    def predict(point: Point) = {
      val (pk, distance) = clusterCenters.zipWithIndex.map(_.swap).map(cc => (cc._1, point.sqdist(cc._2.avg))).minBy(_._2)
      (pk, distance)
    }

    val prediction = predict(data)

    def predictByDimension(data: Point): Point = {

      val k = prediction._1
      val featuresNo = data.size

      val variances = clusterCenters(k).variance

      val averages = clusterCenters(k).avg

      (0 until featuresNo).map { f =>
        val sqerr = (data(f) - averages(f)) * (data(f) - averages(f))
        val variance = if (variances(f) == 0) Double.MinValue else variances(f)
        import math._
        val probability = (1 / (sqrt(2 * Pi * variance)) *
          math.exp(-sqerr / (2 * variance)))
        probability
      }.toArray
    }
    val probability = predictByDimension(data).reduce(_ * _)
    XKMeansCxPrediction(prediction._1, prediction._2, probability, predictByDimension(data))

  }

}

object XKMeansTrainer {

  def initialize(k: Int, points: ParSeq[Point], seed: Long): Seq[Stats[Point]] = {
    require(k > 0 && k < points.size)
    new Random(seed).shuffle((0 until points.size).toList)
      .foldLeft(Seq[Point]())((clusters, point) => points(point) +: clusters)
      .take(k).map(Stats.fromPoint)
  }

  def initialize(k: Int, points: Seq[Point], seed: Long = Random.nextLong): Seq[Stats[Point]] = {
    initialize(k, points.par, seed)
  }
}

/**
 * Traininer for XKMeans predictor
 *
 * @param k
 * @param maxIter
 * @param tolerance
 * @param seed
 */
case class XKMeansTrainer(k: Int, maxIter: Int = 100, tolerance: Double = 1E-6, seed: Long = Random.nextLong) extends Trainer[Point, XKMeans] {

  import XKMeansTrainer._

  private val random = new Random(seed)

  /**
   * Train a MeanKs model and return the centroids
   *
   * @param dataPoints
   * @return
   */
  override def train(dataPoints: ParSeq[Point]): XKMeans =
    train(initialize(k, dataPoints, random.nextLong()), dataPoints)

  override def train(dataPoints: Seq[Point]): XKMeans = {
    val parPoints = dataPoints.par
    train(initialize(k, parPoints, random.nextLong()), parPoints)
  }

  def train(initialCentroids: Seq[Stats[Point]], dataPoints: ParSeq[Point]): XKMeans = {

    def train(oldCentroids: Seq[Stats[Point]], step: Int, done: Boolean): Seq[Stats[Point]] = {
      if (step == maxIter - 1 || done)
        oldCentroids
      else {
        val newKs = newCentroids(dataPoints, oldCentroids)
        val done = centroidsMovements(oldCentroids, newKs).sum <= tolerance
        train(newKs, step + 1, done)
      }
    }

    def newCentroids(points: ParSeq[Point], clusters: Seq[Stats[Point]]): Seq[Stats[Point]] = {
      val pointsByK = points.map(p => (p, XKMeans(clusters).predict(p)))
      // TODO write an "aggregateBy" version
      val newClusters = pointsByK.groupBy(_._2.k).map {
        case (k, kfx) =>
          kfx.map(xp => Stats.fromPoint(xp._1)).reduce(_ |+| _)
      }.toList
      newClusters
    }

    def centroidsMovements(oldCentroids: Seq[Stats[Point]], newCentroids: Seq[Stats[Point]]) = {
      oldCentroids.zip(newCentroids).map {
        case (oc, nc) =>
          oc.avg.sqdist(nc.avg)
      }
    }

    val centroids = train(initialCentroids, 0, false)
    XKMeans(centroids)
  }

}

