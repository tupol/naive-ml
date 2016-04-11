package tupol.ml.clustering

import org.scalatest.{FunSuite, Matchers}
import tupol.ml._
import tupol.ml.utils.ClusterGen2D

/**
  *
  */
class KMeansGuessKTest extends FunSuite with Matchers {

  import ClusterGen2D._

  val dataPoints2L = disc(1000, Array(0.0, 0.0)) ++ disc(1000, Array(2.0, 2.0))

  val dataPoints4L =  disc(1000, Array(0.0, 2.0)) ++ disc(1000, Array(0.0, 2.0))

  val dataPoints6L = dataPoints4L ++ disc(1000, Array(5.0, 2.0)) ++ disc(1000, Array(2.0, 5.0))

  val dataPoints9L = dataPoints6L ++ disc(1000, Array(-5.0, -2.0)) ++ disc(1000, Array(-2.0, -5.0)) ++ disc(1000, Array(-5.0, -5.0))

  val LARGE_EPSILON = 0.01
  val NORMAL_EPSILON = 0.0005
  val SMALL_EPSILON =  0.000001

  test("KMeans#guessK integration test for 2 clusters with a large epsilon") {

    val dataPoints = disc(300, Array(0.0, 0.0)) ++ disc(300, Array(2.0, 2.0))

    testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 2, expectedKTolerance = 0)

  }

  test("KMeans#guessK integration test for 2 clusters with a normal epsilon") {

    val dataPoints = disc(300, Array(0.0, 0.0)) ++ disc(300, Array(2.0, 2.0))

    testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 2, expectedKTolerance = 0)

  }

  test("KMeans#guessK integration test for 2 clusters with a small epsilon") {

    val dataPoints = disc(300, Array(0.0, 0.0)) ++ disc(300, Array(2.0, 2.0))

    testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 26, expectedKTolerance = 6)

  }


  test("KMeans#guessK integration test for 4 clusters with a large epsilon") {

    val dataPoints = disc(300, Array(0.0, 0.0)) ++ disc(300, Array(2.0, 2.0)) ++
      disc(300, Array(2.0, 0.0)) ++ disc(300, Array(0.0, 2.0))

    testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 3, expectedKTolerance = 1)

  }

  test("KMeans#guessK integration test for 4 clusters with a normal epsilon") {

    val dataPoints = disc(300, Array(0.0, 0.0)) ++ disc(300, Array(2.0, 2.0)) ++
      disc(300, Array(2.0, 0.0)) ++ disc(300, Array(0.0, 2.0))

    testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 4, expectedKTolerance = 1)

  }

  test("KMeans#guessK integration test for 4 clusters with a small epsilon") {

    val dataPoints = disc(300, Array(0.0, 0.0)) ++ disc(300, Array(2.0, 2.0)) ++
      disc(300, Array(2.0, 0.0)) ++ disc(300, Array(0.0, 2.0))

    testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 36, expectedKTolerance = 6)

  }

  test("KMeans#guessK integration test for 9 clusters with a large epsilon") {

    val dataPoints = dataPoints9L

    testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 6, expectedKTolerance = 3)

  }

  test("KMeans#guessK integration test for 9 clusters with a normal epsilon") {

    val dataPoints = dataPoints9L

    testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 10, expectedKTolerance = 2)

  }

  test("KMeans#guessK integration test for 9 clusters with a small epsilon") {

    val dataPoints = dataPoints9L

    testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 160, expectedKTolerance = 10)

  }

  test("KMeans#guessK integration test for 20 clusters with a large epsilon") {

    val Ks = 20
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map{_ => (random.nextInt(Ks), random.nextInt(Ks))}
    val dataPoints = originalCentroids.flatMap(x => disc(500, Array(x._1, x._2), 0.5))

    testGuessK(dataPoints, epsilon = LARGE_EPSILON, expectedK = 10, expectedKTolerance = 4)

  }

  test("KMeans#guessK integration test for 20 clusters with a normal epsilon ") {

    val Ks = 20
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map{_ => (random.nextInt(Ks), random.nextInt(Ks))}
    val dataPoints = originalCentroids.flatMap(x => disc(500, Array(x._1, x._2), 0.5))

    testGuessK(dataPoints, epsilon = NORMAL_EPSILON, expectedK = 24, expectedKTolerance = 5)

  }

  test("KMeans#guessK integration test for 20 clusters with a small epsilon") {

    val Ks = 20
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map{_ => (random.nextInt(Ks), random.nextInt(Ks))}
    val dataPoints = originalCentroids.flatMap(x => disc(500, Array(x._1, x._2), 0.5))

    testGuessK(dataPoints, epsilon = SMALL_EPSILON, expectedK = 500, expectedKTolerance = 50)

  }

  private def testGuessK(dataPoints: Seq[Point], epsilon: Double, expectedK: Int, expectedKTolerance: Int, runs: Int = 3) = {

    def kMeansTrainerFactory(k: Int) = KMeansTrainer(k, 100, 0.1)

    val guessedK = KMeans.guessK(dataPoints, runs, kMeansTrainerFactory, epsilon)

    println(s"Best guess K = $guessedK; expected K = $expectedK; tolerance = $expectedKTolerance")

    math.abs(guessedK - expectedK) should be <= expectedKTolerance
  }

}

