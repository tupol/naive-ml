package tupol.ml.clustering

import tupol.ml._

import scala.annotation.tailrec
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
   * @param k_measure A sequence of tuples of k and sse representing evolution of a measurement (e.g. WSSSE) over k.
   *              It is recommended to use evenly spread ks (like 10, 20, 30, 40....)
   * @param threshold The acceptable decrease ratio of the measurements between two consecutive ks.
   *                  For example, how mny times smaller is wssse(k+1) compared of wssse(k-1).
   * @param maxK Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.
   */
  def chooseK(k_measure: Seq[(Double, Double)], stdSSE: (Double) => Double, threshold: Double = 0.02, maxK: Int = 140)(reportFile: String): Int = {

    import math._
    val aproxFunction = CxFun(Map(
      ("1 / ln(x)", (x: Double) => 1 / log(x)),
      ("1 / x", (x: Double) => 1 / x),
      ("1 / x^2", (x: Double) => 1 / (x * x)),
      ("1 / x^3", (x: Double) => 1 / (x * x * x))
    ))

    chooseK(k_measure, aproxFunction, stdSSE, threshold, maxK)(reportFile)
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
   * @param threshold The acceptable decrease ratio of the measurements between two consecutive ks.
   *                  For example, how mny times smaller is wssse(k+1) compared of wssse(k).
   * @param maxK Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.
   */
  def chooseK(k_measure: Seq[(Double, Double)], aproxFun: CxFun, stdSSE: (Double) => Double, threshold: Double, maxK: Int)(reportFile: String): Int = {

    require(maxK >= k_measure.map(_._1).max, "The specified maxK should be larger than the maximum k from the input measurements.")

    import breeze.linalg.{ DenseMatrix, pinv }
    val size = k_measure.size
    val X = DenseMatrix.create(1, size, k_measure.map(_._1.toDouble).toArray)
    val Y = DenseMatrix.create(1, size, k_measure.map(_._2).toArray).t

    val cxs = aproxFun.functions.map(_._2).map { f => X.map(f) }.toSeq

    val Xs = DenseMatrix.vertcat(cxs: _*).t

    val T = (pinv(Xs.t * Xs) * Xs.t * Y).toArray

    val kFunction = aproxFun.withParameters(T)

    val lines = Seq("", s"The values used for approximating the measurement function are:", "",
      s"K   , Avg WSSE, Approx, WSSE * K^2, StdSSE") ++
      k_measure.map { case (k, v) => f"$k%4.0f, $v, ${kFunction(k)}, ${kFunction(k)*k*k}, ${stdSSE(k)}" } ++
      Seq("", s"""
         |The approximation function is:
         |  f(x) = ${kFunction}
         |  """.stripMargin,
        "",
          s"K   , Avg WSSE, Slope, Avg WSSE * K^2, Slope, Ideal Avg WSSE")

    println(lines.mkString("\n"))

    saveLinesToFile(lines, reportFile)
    // Start with k == 2
    findK(2.0, threshold, maxK, kFunction, stdSSE)(reportFile: String).round.toInt
  }

  /**
   * This is the best attempt I found so far to guess K.
   *
   * Until now the best guess come for an epsilon = 0.0003, which is set as default.
   *
   * @param data Training data
   * @param runs How many times should a model be generated for the same input parameters; I recommend a minimum of 3
   * @param kMeansTrainerFactory A function that returns a KMeansTrainer, given a K as an input
   * @param threshold The acceptable decrease ratio (derivative) of the measurements for every k.
   * @param maxK Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.
   * @return The best guess for K
   */
  def guessK(data: ParSeq[Point], runs: Int = 3, kMeansTrainerFactory: (Int) => KMeansTrainer, threshold: Double = 0.01, maxK: Int = 500)(reportFile: String): Int = {

    val ks = (
      (2 until 10 by 2) ++ (10 to 140 by 10)
    ).filter(x => x <= data.size / 2 && x <= maxK)

    val dataPointsWithLogs = ks.map { k =>
      def kMeansTrainer = kMeansTrainerFactory(k)
      val (model, sse) = bestModel(data, runs, kMeansTrainer)
      val minKdist = model.clusterCenters.map(from => model.clusterCenters.filterNot(_.k == from.k).map(to => from.point.distance2(to.point)).min).sum / 2
      val avgKdist = model.clusterCenters.map(from => model.clusterCenters.filterNot(_.k == from.k).map(to => from.point.distance2(to.point)).sum/model.clusterCenters.size).sum / 2
      val maxKdist = model.clusterCenters.map(from => model.clusterCenters.filterNot(_.k == from.k).map(to => from.point.distance2(to.point)).max).sum / 2

      val line = (f"$k%5f, $minKdist%1.4E, ${avgKdist}%1.4E, ${maxKdist}%1.4E, ${minKdist/k}%1.4E")

      (k.toDouble, sse / data.size, line)
    }

    val lines = Seq("", f"K, SUM MIN^2, SUM AVG^2, SUM MAX^2, AVG MIN^2") ++ dataPointsWithLogs.map(_._3)
    saveLinesToFile(lines, reportFile)

    val dataPoints = dataPointsWithLogs.map(x => (x._1, x._2))

    def stdSSE(k: Double, dataSize : Long, dataRange: Point): Double = {
      (  dataRange * (dataPoints.size + 2 * k)  / 2 / k ).map(x => x * x).sum
    }

    val range = data.maxByDimension :- data.minByDimension

    def stdsse(k: Double) = stdSSE(k, data.size, range) / data.size

    KMeans.chooseK(dataPoints, stdsse, threshold)(reportFile)
  }

  def guessK(data: Seq[Point], runs: Int, kMeansTrainerFactory: (Int) => KMeansTrainer, threshold: Double, maxK: Int)(reportFile: String): Int = {
    guessK(data.par, runs, kMeansTrainerFactory, threshold, maxK)(reportFile)
  }

  def scale(data: Seq[Double], a: Double = 0.0, b: Double = 1.0) = {
    val dMin = data.min
    val dMax = data.max
    def scale(x: Double) = (b - a) * (x - dMin) / (dMax - dMin) + a
    data.map(scale(_))
  }

  /**
   * Choose the best model, by running the prediction `runs` times and picking the model with the minimum SSE.
   *
   * @param data Training data
   * @param runs
   * @param kMeansTrainer
   * @return
   */
  def bestModel(data: ParSeq[Point], runs: Int, kMeansTrainer: => KMeansTrainer): (KMeans, Double) = {
    (0 until runs).map(_ => {


      val size = data.size

      val trainingData = data.take((size * 0.7).toInt)
      val testData = data.drop((size * 0.7).toInt)
      val kmeans = kMeansTrainer.train(data)
      val sse = kmeans.predict(data).map(_.label._2).sum
      (kmeans, sse )
    }).minBy(_._2)
  }

  private def point2Str(p: Point) = p.map(x => f"$x%+5.2f").mkString(", ")

  def bestModel(trainingData: Seq[Point], runs: Int, kMeansTrainer: => KMeansTrainer): (KMeans, Double) = {
    bestModel(trainingData.par, runs, kMeansTrainer)
  }

  @tailrec
  private def findK(k: Double, threshold: Double, maxK: Int, kFunction: (Double) => Double, stdSSE: (Double) => Double, epsilon: Double = 0.1, step: Double= 1.0)(reportFile: String): Double = {
    import math._

    def ksqFun(k: Double) = kFunction(k) * k * k

    val f11 = kFunction(k)
    val f12 = kFunction(k + epsilon)
    val variation1 = (f12 - f11) / (epsilon)


    val f21 = ksqFun(k)
    val f22 = ksqFun(k + epsilon)
    val variation2 = (f22 - f21) / (epsilon)

    val line = f"$k%5.2f, ${kFunction(k)}%1.4E, $variation1%1.4E, ${kFunction(k) * k * k}%1.4E, $variation2%1.4E, ${stdSSE(k)}%1.4E"
    saveLinesToFile(Seq(line), reportFile)
//    if ((variation <= 0 ) || k > maxK) k
    if (k > 200) k
    else findK(k + step, threshold, maxK, kFunction, stdSSE)(reportFile)
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
  override def train(dataPoints: ParSeq[Point]): KMeans =
    train(initialize(k, dataPoints, random.nextLong()), dataPoints)

  override def train(dataPoints: Seq[Point]): KMeans = {
    val parPoints = dataPoints.par
    train(initialize(k, parPoints, random.nextLong()), parPoints)
  }
  /**
   * Train a MeanKs model and return the centroids
   *
   * @param initialCentroids
   * @param dataPoints
   * @return
   */
  def train(initialCentroids: Seq[ClusterPoint], dataPoints: ParSeq[Point]): KMeans = {

    def train(oldCentroids: Seq[ClusterPoint], step: Int, done: Boolean): Seq[ClusterPoint] = {
      if (step == maxIter - 1 || done)
        oldCentroids
      else {
        val newCentroids = newCentroids(dataPoints, oldCentroids)
        val done = centroidsMovements(oldCentroids, newCentroids).sum <= tolerance
        train(newCentroids, step + 1, done)
      }
    }

    def newCentroids(points: ParSeq[Point], clusters: Seq[ClusterPoint]): Seq[ClusterPoint] = {
      // Assign points to closest clusters (predict)
      val pointsByK: ParSeq[KMeansLabeledPoint] = KMeans(clusters).predict(points)
      val newClusters = pointsByK.groupBy(_.label._1).map { case (k, kfx) => ClusterPoint(k, mean(kfx.map(_.point))) }.toSeq
      newClusters.toList
    }

    val centroids = train(initialCentroids, 0, false)
    KMeans(centroids)

  }

  private def centroidsMovements(oldCentroids: Seq[ClusterPoint], newCentroids: Seq[ClusterPoint]) =
    newCentroids.map {
      case dlp =>
        oldCentroids.map(x => (x.k, x.point)).toMap.get(dlp.k).map(x => x.distance2(dlp.point))
    }.filter(_.isDefined).map(_.get)

  def train(initialCentroids: Seq[ClusterPoint], dataPoints: Seq[Point]): KMeans = {
    train(initialCentroids, dataPoints.par)
  }

}

