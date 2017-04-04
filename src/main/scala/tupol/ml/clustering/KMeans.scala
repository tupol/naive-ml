package tupol.ml.clustering

import tupol.ml._
import tupol.ml.pointops._

import scala.collection.parallel.ParSeq
import scala.util.Random

case class KMeansLabeledPoint(label: (Int, Double), point: Point) extends LabeledPoint[(Int, Double)](label, point)

case class ClusterPoint(k: Int, point: Point) extends LabeledPoint[Int](k, point)

/**
 * KMeans predictor
 */
case class KMeans(clusterCenters: Seq[ClusterPoint]) extends Predictor[Point, KMeansLabeledPoint] {

  /**
   * Predict the cluster a point belongs to.
   *
   * @param point
   */
  def predict(point: Point): KMeansLabeledPoint = {
    val (pk, distance) = clusterCenters.map(cc => (cc.k, point.sqdist(cc.point))).minBy(_._2)
    KMeansLabeledPoint((pk, distance), point)
  }

}

object KMeans {

  /**
   * Helper function for choosing the K.
   *
   * The idea is that a sequence of models will be generated over a range of k (e.g. from 10 to 100 with a step of 5) and calculate a quality parameter for each model (like wssse).
   * With this data we can generate a sequence of tuples of k and quality measurement, which we use to pick the best acceptable k.
   *
   * @param k_measure A sequence of tuples of k and sse representing evolution of a measurement (e.g. WSSSE) over k.
   *              It is recommended to use evenly spread ks (like 10, 20, 30, 40....)
   * @param slope The acceptable decrease ratio of the measurements between two consecutive ks.
   *                  For example, how mny times smaller is wssse(k+1) compared of wssse(k-1).
   * @param maxK Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.
   */
  def chooseK(k_measure: Seq[(Double, Double)], slope: Double = 0.02, maxK: Int = 500): Int = {

    import math._
    val aproxFunction = CxFun(Map(
      //  ("1", (x: Double) => 1),
      ("1 / ln(x)", (x: Double) => 1 / log(x))
    //  ("1 / x^2", (x: Double) => 1 / (x * x)),
    //  ("1 / x^3", (x: Double) => 1 / (x * x * x))
    ))

    chooseK(k_measure, aproxFunction, slope, maxK)
  }

  /**
   * Helper function for choosing the K.
   *
   * The idea is that a sequence of models will be generated over a range of k (e.g. from 10 to 100 with a step of 5) and calculate a quality parameter for each model (like wssse).
   * With this data we can generate a sequence of tuples of k and quality measurement, which we use to pick the best acceptable k.
   *
   * @param k_measure A sequence of tuples of k and sse representing evolution of a measurement (e.g. WSSSE) over k.
   *              It is recommended to use evenly spread ks (like 10, 20, 30, 40....)
   * @param aproxFun A custom function of Double to Double that should be used to approximate the K curve;
   *                 the actual parameters will be found using the normal equation
   * @param slope The acceptable decrease ratio of the measurements between two consecutive ks.
   *                  For example, how mny times smaller is wssse(k+1) compared of wssse(k).
   * @param maxK Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.
   */
  def chooseK(k_measure: Seq[(Double, Double)], aproxFun: CxFun, slope: Double, maxK: Int): Int = {

    require(maxK >= k_measure.map(_._1).max, "The specified maxK should be larger than the maximum k from the input measurements.")

    import breeze.linalg.{ DenseMatrix, pinv }
    val size = k_measure.size
    val X = DenseMatrix.create(1, size, k_measure.map(_._1.toDouble).toArray)
    val Y = DenseMatrix.create(1, size, k_measure.map(_._2).toArray).t

    val cxs = aproxFun.functions.map(_._2).map { f => X.map(f) }.toSeq

    val Xs = DenseMatrix.vertcat(cxs: _*).t

    val T = (pinv(Xs.t * Xs) * Xs.t * Y).toArray

    val kFunction = aproxFun.withParameters(T)

    //    println(s"The values used for approximating the measurement function are:")
    //    println(s"  K, Value")
    //    k_measure.foreach { case (k, v) => println(f"$k%3.0f, $v") }
    //    println(
    //      s"""The approximation function is:
    //         |  f(x) = ${kFunction}""".stripMargin
    //    )

    val secondDerivative = (x: Double) => -T(0) / x / x

    // Start with k == 2
    findK(2, slope, maxK, secondDerivative).toInt
  }

  /**
   * This is the best attempt I found so far to guess K.
   *
   * Until now the best guess come for an epsilon = 0.0003, which is set as default.
   *
   * @param data Training data
   * @param runs How many times should a model be generated for the same input parameters; I recommend a minimum of 3
   * @param kMeansTrainerFactory A function that returns a KMeansTrainer, given a K as an input
   * @param slope The acceptable decrease ratio (derivative) of the measurements for every k.
   * @param maxK Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.
   * @return The best guess for K
   */
  def guessK(data: Seq[Point], runs: Int = 3, kMeansTrainerFactory: (Int) => KMeansTrainer, slope: Double = 0.01, maxK: Int = 500): Int = {

    val ks = (
      (2 to 10 by 3) ++
      (4 to 10).map(x => x * x) ++
      (5 to 20).map(x => x * x * x)
    ).filter(x => x <= data.size / 2 && x <= maxK)

    val sses = ks.map { k =>
      def kMeansTrainer = kMeansTrainerFactory(k)
      val (model, sse) = bestModel(data, runs, kMeansTrainer)
      (k.toDouble, sse)
    }

    KMeans.chooseK(sses, slope)
  }

  /**
   * Choose the best model, by running the prediction `runs` times and picking the model with the minimum SSE.
   *
   * @param trainingData Training data
   * @param runs
   * @param kMeansTrainer
   * @return
   */
  def bestModel(trainingData: Seq[Point], runs: Int, kMeansTrainer: => KMeansTrainer): (KMeans, Double) = {
    (0 until runs).map(_ => {
      val kmeans = kMeansTrainer.train(trainingData)
      val sse = kmeans.predict(trainingData).map(_.label._2).sum
      (kmeans, sse / trainingData.size)
    }).minBy(_._2)
  }

  private def findK(k: Int, slope: Double, maxK: Int, kFunction: (Double) => Double, step: Double = 1E-9): Double = {
    import math._
    val f0 = kFunction(k)
    val f1 = kFunction(k - step)
    val f2 = kFunction(k + step)
    val variation = (f1 - f0) / (1 * step)
    //    println(f"$k%3d,${kFunction(k)}%1.4E,$variation%1.4E")
    // if the derivative is acceptable and the trend is still decreasing of the maxK was reached we stop
    //    if ((variation <= slope && f2 <= f1) || k > maxK) k
    //    println(f"$k%3d, ${kFunction(k)}%1.4E, $variation%1.4E")
    if ((abs(kFunction(k)) <= abs(slope)) || k > maxK) k
    else findK(k + 1, slope, maxK, kFunction)
  }

}

object KMeansTrainer {

  def initialize(k: Int, points: ParSeq[Point], seed: Long): Seq[ClusterPoint] = {
    require(k > 0 && k < points.size)
    new Random(seed).shuffle((0 until points.size).toList)
      .foldLeft(Seq[Point]())((clusters, point) => points(point) +: clusters)
      .take(k).zipWithIndex.map(_.swap).map(t => ClusterPoint(t._1, t._2))
  }

  def initialize(k: Int, points: Seq[Point], seed: Long = Random.nextLong): Seq[ClusterPoint] = {
    initialize(k, points.par, seed)
  }
}

/**
 * Traininer for KMeans predictor
 *
 * @param k
 * @param maxIter
 * @param tolerance
 * @param seed
 */
case class KMeansTrainer(k: Int, maxIter: Int = 100, tolerance: Double = 1E-6, seed: Long = Random.nextLong) extends Trainer[Point, KMeans] {

  import KMeansTrainer._

  private val random = new Random(seed)

  /**
   * Train a MeanKs model and return the centroids
   *
   * @param dataPoints
   * @return
   */
  override def train(dataPoints: Seq[Point]): KMeans = {
    train(initialize(k, dataPoints, random.nextLong()), dataPoints)
  }
  /**
   * Train a MeanKs model and return the centroids
   *
   * @param initialCentroids
   * @param dataPoints
   * @return
   */
  def train(initialCentroids: Seq[ClusterPoint], dataPoints: Seq[Point]): KMeans = {

    def train(oldCentroids: Seq[ClusterPoint], step: Int, done: Boolean): Seq[ClusterPoint] = {
      if (step == maxIter - 1 || done)
        oldCentroids
      else {
        val newKs = newCentroids(dataPoints, oldCentroids)
        val done = centroidsMovements(oldCentroids, newKs).sum <= tolerance
        train(newKs, step + 1, done)
      }
    }

    def newCentroids(points: Seq[Point], clusters: Seq[ClusterPoint]): Seq[ClusterPoint] = {
      val pointsByK = KMeans(clusters).predict(points)
      val newClusters = pointsByK.groupBy(_.label._1).map { case (k, kfx) => ClusterPoint(k, mean(kfx.map(_.point))) }.map(cc => (cc.k, cc)).toMap
      // make sure we preserve the number of clusters
      clusters.map { oc => newClusters.getOrElse(oc.k, oc) }
    }

    val centroids = train(initialCentroids, 0, false)
    KMeans(centroids)

  }

  private def centroidsMovements(oldCentroids: Seq[ClusterPoint], newCentroids: Seq[ClusterPoint]) =
    newCentroids.map {
      case dlp =>
        oldCentroids.map(x => (x.k, x.point)).toMap.get(dlp.k).map(x => x.sqdist(dlp.point))
    }.filter(_.isDefined).map(_.get)

}
