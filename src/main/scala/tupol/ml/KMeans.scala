package tupol.ml

import scala.util.Random

case class KMeansLabeledPoint(label: (Int, Double), point: Point) extends LabeledPoint[(Int, Double)](label, point)

case class ClusterPoint(k: Int, point: Point) extends LabeledPoint[Int](k, point)

/**
 *
 */
case class KMeans(clusterCenters: Seq[ClusterPoint]) extends Predictor[Point, KMeansLabeledPoint] {

  /**
   * Predict the cluster a point belongs to.
   *
   * @param point
   */
  def predict(point: Point): KMeansLabeledPoint = {
    val (pk, distance) = clusterCenters.map(cc => (cc.k, point.distance2(cc.point))).minBy(_._2)
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
   * @param k_measure A sequence of tupples of k and sse representing evolution of a measurement (e.g. WSSSE) over k.
   *              It is recommended to use evenly spread ks (like 10, 20, 30, 40....)
   * @param epsilon The acceptable variance ratio of the measurements between two consecutive ks;.
   *                  For example, how mny times smaller is wssse(k+1) compared of wssse(k).
   * @param maxK Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.
   */
  def chooseK(k_measure: Seq[(Double, Double)], epsilon: Double = 0.005, maxK: Double = 500): Int = {

    require(maxK >= k_measure.map(_._1).max, "The specified maxK should be larger than the maximum k from the input measurements.")

    import breeze.linalg.{ DenseMatrix, pinv }
    val size = k_measure.size
    val I = DenseMatrix.ones[Double](1, k_measure.size).toDenseMatrix
    val X = DenseMatrix.create(1, size, k_measure.map(_._1).toArray)
    val X1 = X.map(1 / math.sqrt(_))
    val Y = DenseMatrix.create(1, size, k_measure.map(_._2).toArray).t

    val Xs = DenseMatrix.vertcat(I, X, X1).t

    val T = (pinv(Xs.t * Xs) * Xs.t * Y).toArray

    /**
     * Aproximation of the measurement function of k
     *
     * @param k
     * @return
     */
    def measureF(k: Double) = T(0) + T(1) * k + T(2) / math.sqrt(k)

    def findK(k: Double): Double = {
      val varianceRatio = 1 - math.abs(measureF(k + 1) / measureF(k))
      if (varianceRatio <= epsilon || k > maxK) k
      else findK(k + 1)
    }

    // Start with k == 2
    findK(2.0).toInt
  }

}

object KMeansTrainer {

  def initialize(k: Int, points: Seq[Point]): Seq[ClusterPoint] = {
    require(k > 0 && k < points.size)
    Random.shuffle((0 until points.size).toList)
      .foldLeft(Seq[Point]())((clusters, point) => points(point) +: clusters)
      .take(k).zipWithIndex.map(_.swap).map(t => ClusterPoint(t._1, t._2))
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
  def train(initialCentroids: Seq[ClusterPoint], dataPoints: Seq[Point]): KMeans = {

    def train(oldCentroids: Seq[ClusterPoint], step: Int, done: Boolean): Seq[ClusterPoint] = {
      if (step == maxIter - 1 || done)
        oldCentroids
      else {
        val newCentroids = newControids(dataPoints, oldCentroids)
        val done = centroidsMovements(oldCentroids, newCentroids).isEmpty
        train(newCentroids, step + 1, done)
      }
    }

    def centroidsMovements(oldCentroids: Seq[ClusterPoint], newCentroids: Seq[ClusterPoint]) =
      newCentroids.map {
        case dlp =>
          oldCentroids.map(x => (x.k, x.point)).toMap.get(dlp.k).map(x => x.distance2(dlp.point))
      }.filter(_.isDefined).map(_.get)

    def newControids(points: Seq[Point], clusters: Seq[ClusterPoint]): Seq[ClusterPoint] = {
      val pointsByK = KMeans(clusters).predict(points)
      val newClusters = pointsByK.groupBy(_.label._1).map { case (k, kfx) => ClusterPoint(k, mean(kfx.map(_.point))) }.toSeq
      newClusters
    }

    val centroids = train(initialCentroids, 0, false)
    KMeans(centroids)

  }

}

