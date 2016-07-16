package tupol.ml.clustering

import tupol.ml._

import scala.annotation.tailrec
import scala.collection.parallel.ParSeq
import scala.math._
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

  def main(args: Array[String]) = {

    val kmeasure = Array(
//      (2	,582.2416357478),
//      (5	 , 185.4437605004),
//      (8	 , 103.140507434),
//      (16	,41.5307314342),
//      (25	,21.8114009537),
//      (36	,12.2163124606),
//      (49	,7.6140197723),
//      (64	,2.9476137549),
//      (81	,2.2645240378),
//      (100	,0.9918006621),
//      (125	,0.2728564177),
//      (216	,0.0483969821),
//      (343	,0.0311292278)

        (2, 1.918628584474461),
      (10, 0.1918628584474461),
      (20, 0.03690000594866543),
      (30, 0.030480303883404558),
      (40, 0.02055326352243121),
      (50, 0.015053729795239078),
      (60, 0.01385961433299094),
      (70, 0.00922247340114342),
      (80, 0.008408645370093995),
      (90, 0.00703045112075809),
      (100, 0.006798031512871512)

//      (2.0, 1.0061763333816824),
//      (10.0, 0.2549886721024244),
//      (20.0, 0.12273979534565847),
//      (30.0, 0.07543657556372774),
//      (40.0, 0.05606289347127518),
//      (50.0, 0.0390735695704057),
//      (60.0, 0.04016818455148996),
//      (70.0, 0.02649656735771119),
//      (80.0, 0.024835685376829397),
//      (90.0, 0.022481931806457776),
//      (100.0, 0.021041603405182237)
    ).map{case (k,x) => (k.toDouble, 128000*x)}

//    println(chooseK(kmeasure, threshold = 0.1))
//    println(chooseK(kmeasure, threshold = -0.0333))
//    println(chooseK(kmeasure, threshold = 0.01))

  }

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
  def chooseK(k_measure: Seq[(Double, Double)], stdSSE: (Double) => Double, threshold: Double = 0.02, maxK: Int = 140): Int = {

    import math._
    val aproxFunction = CxFun(Map(
      ("1 / ln(x)", (x: Double) => 1 / log(x)),
      ("1 / x", (x: Double) => 1 / x),
      ("1 / x^2", (x: Double) => 1 / (x * x)),
      ("1 / x^3", (x: Double) => 1 / (x * x * x))
    ))

    chooseK(k_measure, aproxFunction, stdSSE, threshold, maxK)
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
  def chooseK(k_measure: Seq[(Double, Double)], aproxFun: CxFun, stdSSE: (Double) => Double, threshold: Double, maxK: Int): Int = {

    require(maxK >= k_measure.map(_._1).max, "The specified maxK should be larger than the maximum k from the input measurements.")

    import breeze.linalg.{ DenseMatrix, pinv }
    val size = k_measure.size
    val X = DenseMatrix.create(1, size, k_measure.map(_._1.toDouble).toArray)
    val Y = DenseMatrix.create(1, size, k_measure.map(_._2).toArray).t

    val cxs = aproxFun.functions.map(_._2).map { f => X.map(f) }.toSeq

    val Xs = DenseMatrix.vertcat(cxs: _*).t

    val T = (pinv(Xs.t * Xs) * Xs.t * Y).toArray

    val kFunction = aproxFun.withParameters(T)

    println(s"The values used for approximating the measurement function are:")
    println(s"K   , Avg WSSE, Approx, WSSE * K^2, StdSSE")
    k_measure.foreach { case (k, v) => println(f"$k%4.0f, $v, ${kFunction(k)}, ${kFunction(k)*k*k}, ${stdSSE(k)}") }
    println(
      s"""
         |The approximation function is:
         |  f(x) = ${kFunction}
         |  """.stripMargin
    )

    println(s"K   , Avg WSSE, Avg WSSE * K^2, Slope, StdSSE")
    // Start with k == 2
    findK(2.0, threshold, maxK, kFunction, stdSSE).round.toInt
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
  def guessK(data: ParSeq[Point], runs: Int = 3, kMeansTrainerFactory: (Int) => KMeansTrainer, threshold: Double = 0.01, maxK: Int = 500): Int = {

    val ks = (
//      (2 to 10 by 3) ++
//      (4 to 10).map(x => x * x) ++
//      (5 to 20).map(x => x * x * x)
//      2 +: ((1 to 10) ++ (15 to 30 by 5)).map(_ * 10)
      (2 until 10 by 2) ++ (10 to 140 by 10)
    ).filter(x => x <= data.size / 2 && x <= maxK)

    println(f"K, avg, MIN, AVG, MAX")
    val dataPoints = ks.map { k =>
      def kMeansTrainer = kMeansTrainerFactory(k)
      val (model, sse) = bestModel(data, runs, kMeansTrainer)

      val kdists = for {
        from <- model.clusterCenters
        to <- model.clusterCenters
      } yield(from.point.distance2(to.point))
      val kdist = kdists.sum / 2 / k

      val minKdist = model.clusterCenters.map(from => model.clusterCenters.filterNot(_.k == from.k).map(to => from.point.distance2(to.point)).min).sum / 2
      val avgKdist = model.clusterCenters.map(from => model.clusterCenters.filterNot(_.k == from.k).map(to => from.point.distance2(to.point)).sum/model.clusterCenters.size).sum / 2
      val maxKdist = model.clusterCenters.map(from => model.clusterCenters.filterNot(_.k == from.k).map(to => from.point.distance2(to.point)).max).sum / 2

      println(f"$k%5f, $kdist%1.4E, $minKdist%1.4E, ${avgKdist}%1.4E, ${maxKdist}%1.4E")

      (k.toDouble, sse / data.size )
    }


    def stdSSE(k: Double, dataSize : Long, dataRange: Point): Double = {
      (  dataRange * (dataPoints.size + 2 * k)  / 2 / k ).map(x => x * x).sum
    }

    val range = data.maxByDimension :- data.minByDimension

    def stdsse(k: Double) = stdSSE(k, data.size, range) / data.size



//    val sses = dataPoints.map(_._2)
//    val dataPointsScaled = ks.map(_.toDouble).zip(scale(sses, 0.0, 1000.0))

    KMeans.chooseK(dataPoints, stdsse, threshold)
  }

  def guessK(data: Seq[Point], runs: Int, kMeansTrainerFactory: (Int) => KMeansTrainer, threshold: Double, maxK: Int): Int = {
    guessK(data.par, runs, kMeansTrainerFactory, threshold, maxK)
  }

//  def scale(data: Seq[Double], a: Double = 0.0, b: Double = 1.0) = {
//    val dMin = data.min
//    val dMax = data.max
//    def scale(x: Double) = (b - a) * (x - dMin) / (dMax - dMin) + a
//    data.map(scale(_))
//  }

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

//      println(s"trainingData = ${trainingData.take(10).map(point2Str(_))}")
//      println(s"testData     = ${testData.take(10).map(point2Str(_))}")

      val kmeans = kMeansTrainer.train(trainingData)
      val sse = kmeans.predict(testData).map(_.label._2).sum
      (kmeans, sse )
    }).minBy(_._2)
  }

  private def point2Str(p: Point) = p.map(x => f"$x%+5.2f").mkString(", ")

  def bestModel(trainingData: Seq[Point], runs: Int, kMeansTrainer: => KMeansTrainer): (KMeans, Double) = {
    bestModel(trainingData.par, runs, kMeansTrainer)
  }

  @tailrec
  private def findK(k: Double, threshold: Double, maxK: Int, kFunction: (Double) => Double, stdSSE: (Double) => Double, epsilon: Double = 0.1, step: Double= 1.0): Double = {
    import math._

    def ksqFun(k: Double) = kFunction(k) * k * k



    val f1 = ksqFun(k)
    val f2 = ksqFun(k + epsilon)
    val variation = (f2 - f1) / (epsilon)

    //    println(f"$k%3d,${kFunction(k)}%1.4E,$variation%1.4E")
    // if the derivative is acceptable and the trend is still decreasing of the maxK was reached we stop
    //    if ((variation <= threshold && f2 <= f1) || k > maxK) k
    println(f"$k%5.2f, ${kFunction(k)}%1.4E, ${kFunction(k) * k * k}%1.4E, $variation%1.4E, ${stdSSE(k)}%1.4E")
    if ((variation <= 0 ) || k > maxK) k
    else findK(k + step, threshold, maxK, kFunction, stdSSE)
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
        val newCentroids = newControids(dataPoints, oldCentroids)
        val done = centroidsMovements(oldCentroids, newCentroids).sum <= tolerance
        train(newCentroids, step + 1, done)
      }
    }

    def newControids(points: ParSeq[Point], clusters: Seq[ClusterPoint]): Seq[ClusterPoint] = {
      val pointsByK = KMeans(clusters).predict(points)
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

