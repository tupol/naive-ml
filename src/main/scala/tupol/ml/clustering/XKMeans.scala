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
case class XKMeans(clusterCenters: Map[Int, Cluster]) extends Predictor[Point, XKMeansCxPrediction] {

  def this(centroids: Seq[Cluster]) = this(centroids.map(cc => (cc.k, cc)).toMap)


  /**
   * Predict the cluster a point belongs to.
   *
   */
  override def predict(data: Point): XKMeansCxPrediction = {

    val (k, distance) = clusterCenters.values.map( cc => (cc.k, data.sqdist(cc.point)) ).minBy(_._2)

    def predictByDimension(data: Point): Point = {

      val featuresNo = data.size

      val variances = clusterCenters(k).statsByDim.variance

      val averages = clusterCenters(k).statsByDim.avg

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
    XKMeansCxPrediction(k, distance, probability, predictByDimension(data))

  }

}

case class Cluster(k: Int, stats : Stats[Double], statsByDim: Stats[Point]) {
  lazy val size = stats.count
  lazy val point = statsByDim.avg
}

object XKMeansTrainer {

  def initialize(k: Int, points: ParSeq[Point], seed: Long): Seq[Cluster] = {
    require(k > 0 && k < points.size)
    new Random(seed).shuffle((0 until points.size).toList)
      .foldLeft(Seq[Point]())((clusters, point) => points(point) +: clusters)
      .take(k)
      .zipWithIndex
      .map{ case(p, k) => Cluster(k, Stats.fromDouble(0), Stats.fromPoint(p)) }
  }

  def initialize(k: Int, points: Seq[Point], seed: Long = Random.nextLong): Seq[Cluster] = {
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

  def train(initialCentroids: Seq[Cluster], dataPoints: ParSeq[Point]): XKMeans = {

    def train(oldCentroids: Seq[Cluster], step: Int, done: Boolean): Seq[Cluster] = {
      if (step == maxIter - 1 || done)
        oldCentroids
      else {
        val newKs = newCentroids(dataPoints, oldCentroids)
        val done = centroidsMovements(oldCentroids, newKs).sum <= tolerance
        train(newKs, step + 1, done)
      }
    }

    def newCentroids(points: ParSeq[Point], clusters: Seq[Cluster]): Seq[Cluster] = {
      val pointsByK = points.map(p => (p, new XKMeans(clusters).predict(p)))
      // TODO write an "aggregateBy" version
      pointsByK.groupBy(_._2.k).map {
        case (k, kfx) =>
          val statsByDist = kfx.map(xp => Stats.fromDouble(xp._2.distance)).reduce(_ |+| _)
          val statsByDim = kfx.map(xp => Stats.fromPoint(xp._1)).reduce(_ |+| _)
          Cluster(k, statsByDist, statsByDim)
      }.toList
    }

    def centroidsMovements(oldCentroids: Seq[Cluster], newCentroids: Seq[Cluster]) = {
      oldCentroids.zip(newCentroids).map {
        case (oc, nc) =>
          oc.statsByDim.avg.sqdist(nc.statsByDim.avg)
      }
    }

    val centroids = train(initialCentroids, 0, false)
    new XKMeans(centroids)
  }

}

