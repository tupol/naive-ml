package tupol.ml.clustering

import org.scalatest.{FunSuite, Matchers}
import tupol.ml._
import tupol.ml.utils.ClusterGen2D

import scala.collection.parallel.ParSeq

/**
 *
 */
class KMeansGuessK extends FunSuite with Matchers {

  import ClusterGen2D._

  val centers_2 = Seq(Array(0.0, 0.0), Array(0.0, 2.0))
  val centers_4 = centers_2 ++ Seq(Array(2.0, 0.0), Array(2.0, 2.0))
  val centers_6 = centers_4 ++ Seq(Array(2.0, 5.0), Array(5.0, 2.0))
  val centers_9 = centers_6 ++ Seq(Array(2.0, 5.0), Array(-5.0, -2.0), Array(-2.0, -5.0), Array(-5.0, -5.0))

  val LARGE_EPSILON = 0.001
  val NORMAL_EPSILON = 0.0003
  val SMALL_EPSILON = 0.00001

  test("KMeans#guessK integration test for 2 clusters with a large epsilon") {

    val dataPoints = centers_2.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 2, expectedKTolerance = 0)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 2 clusters with a normal epsilon") {

    val dataPoints = centers_2.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 2, expectedKTolerance = 0)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 2 clusters with a small epsilon") {

    val dataPoints = centers_2.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 8, expectedKTolerance = 2)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a large epsilon") {

    val dataPoints = centers_4.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 3, expectedKTolerance = 1)

    printOriginalVsActualCentroids(centers_4, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a normal epsilon") {

    val dataPoints = centers_4.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 4, expectedKTolerance = 1)

    printOriginalVsActualCentroids(centers_4, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a small epsilon") {

    val dataPoints = centers_4.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 14, expectedKTolerance = 4)

    printOriginalVsActualCentroids(centers_4, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 9 clusters with a large epsilon") {

    val dataPoints = centers_9.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 6, expectedKTolerance = 3)

    printOriginalVsActualCentroids(centers_9, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 9 clusters with a normal epsilon") {

    val dataPoints = centers_9.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 10, expectedKTolerance = 2)

    printOriginalVsActualCentroids(centers_9, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 9 clusters with a small epsilon") {

    val dataPoints = centers_9.map(p => disc(300, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 44, expectedKTolerance = 4)

    printOriginalVsActualCentroids(centers_9, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 20 clusters with a large epsilon") {

    val Ks = 20
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(300, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 18, expectedKTolerance = 4)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 20 clusters with a normal epsilon ") {

    val Ks = 20
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(300, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 24, expectedKTolerance = 5)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 20 clusters with a small epsilon") {

    val Ks = 20
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(300, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 150, expectedKTolerance = 50)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 50 clusters with a large epsilon ") {

    val Ks = 50
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(300, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 70, expectedKTolerance = 10)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 50 clusters with a normal epsilon ") {

    val Ks = 50
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(300, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 100, expectedKTolerance = 40)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  private def testGuessK(dataPoints: ParSeq[Point], epsilon: Double, expectedK: Int, expectedKTolerance: Int, runs: Int = 5) = {

    val kMeansTrainerFactory = (k: Int) => KMeansTrainer(k, 200, 1E-6)

    val guessedK = KMeans.guessK(dataPoints, runs, kMeansTrainerFactory, epsilon)

    println(s"Best guess K = $guessedK; expected K = $expectedK; tolerance = $expectedKTolerance")

    math.abs(guessedK - expectedK) should be <= expectedKTolerance

    val (bestModel, sse) = KMeans.bestModel(dataPoints, 5, kMeansTrainerFactory(guessedK))
    bestModel
  }

  private def printOriginalVsActualCentroids(original: Seq[Point], actual: Seq[Point]) = {
    val originalKs = original.zipWithIndex.map(_.swap)
    val originalCentroids = originalKs.map(x => ClusterPoint(x._1, x._2))
    val originalModel = KMeans(originalCentroids.toList)

    def point2Str(p: Point) = p.map(x => f"$x%+11.6f").mkString(", ")

    println(f"${"K"}%3s | ${"Distance"}%11s | ${"Actual Center"}%24s | ${"Guessed Center"}%24s")
    originalModel.predict(actual).toList.sortBy(_.label._1).foreach { p =>
      println(f"${p.label._1}%3d | ${p.label._2}%11.6f | ${point2Str(originalKs(p.label._1)._2)}%24s | ${point2Str(p.point)}%24s")
    }
  }

}

