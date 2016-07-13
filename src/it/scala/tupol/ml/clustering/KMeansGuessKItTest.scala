package tupol.ml.clustering

import org.scalatest.{FunSuite, Matchers}
import tupol.ml._
import tupol.ml.utils.ClusterGen2D

import scala.util.Random

/**
 *
 */
class KMeansGuessKItTest extends FunSuite with Matchers {

  import ClusterGen2D._

  val CLUSTER_SIZE = 100

  val centers_2 = Seq(Array(0.0, 0.0), Array(0.0, 2.0))
  val centers_4 = centers_2 ++ Seq(Array(2.0, 0.0), Array(2.0, 2.0))


  val NORMAL_SLOPE = 0.075
  val RELAXED_SLOPE =  NORMAL_SLOPE +  (NORMAL_SLOPE / 2)
  val CONSERVATIVE_SLOPE =  0.000005

  test("KMeans#guessK integration test for 2 clusters with a large epsilon (a.k.a. relaxed).") {

    val dataPoints = centers_2.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _)

    val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 2, expectedKTolerance = 0)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 2 clusters with a normal epsilon") {

    val dataPoints = centers_2.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _)

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 2, expectedKTolerance = 0)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 2 clusters with a small epsilon (a.k.a. conservative).") {

    val dataPoints = centers_2.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _)

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 8, expectedKTolerance = 2)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a large epsilon (a.k.a. relaxed).") {

    val originalCentroids = (0  until 2).flatMap( x=> (0 until 2).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 3, expectedKTolerance = 1)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a normal epsilon") {

    val originalCentroids = (0  until 2).flatMap( x=> (0 until 2).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 4, expectedKTolerance = 1)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a small epsilon (a.k.a. conservative).") {

    val originalCentroids = (0  until 2).flatMap( x=> (0 until 2).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 14, expectedKTolerance = 4)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }


    test("KMeans#guessK integration test for 9 clusters with a large epsilon (a.k.a. relaxed).") {

      val originalCentroids = (0  until 3).flatMap( x=> (0 until 3).map(y => Array(2.0*x, 2.0*y)))
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 6, expectedKTolerance = 3)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 9 clusters with a normal epsilon") {

      val originalCentroids = (0  until 3).flatMap( x=> (0 until 3).map(y => Array(2.0*x, 2.0*y)))
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 10, expectedKTolerance = 2)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 9 clusters with a small epsilon (a.k.a. conservative).") {

      val originalCentroids = (0  until 3).flatMap( x=> (0 until 3).map(y => Array(2.0*x, 2.0*y)))
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 44, expectedKTolerance = 4)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 20 clusters with a large epsilon (a.k.a. relaxed).") {

      val Ks = 20
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 18, expectedKTolerance = 4)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 20 clusters with a normal epsilon ") {

      val Ks = 20
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 20, expectedKTolerance = 5)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 20 random clusters with a small epsilon (a.k.a. conservative).") {

      val Ks = 20
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 150, expectedKTolerance = 50)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

  test("KMeans#guessK integration test for 20 evenly distributed clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 20
    val originalCentroids = (0  until 5).flatMap( x=> (0 until 4  ).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 20, expectedKTolerance = 50)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 30 evenly distributed clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 20
    val originalCentroids = (0  until 5).flatMap( x=> (0 until 6 ).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 20, expectedKTolerance = 50)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 40 evenly distributed clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 20
    val originalCentroids = (0  until 4).flatMap( x=> (0 until 10 ).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 20, expectedKTolerance = 50)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

    test("KMeans#guessK integration test for 50 clusters with a large epsilon (a.k.a. relaxed). (a.k.a. relaxed).") {

      val Ks = 50
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 70, expectedKTolerance = 10)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

  test("KMeans#guessK integration test for 50 clusters with a normal epsilon ") {

    val Ks = 50
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 50, expectedKTolerance = 15)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }



  test("KMeans#guessK integration test for 50 random clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 50
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 100, expectedKTolerance = 40)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 50 evenly distributed clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 50
    val originalCentroids = (0  until 5).flatMap( x=> (0 until 10).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 100, expectedKTolerance = 40)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 60 evenly distributed clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 60
    val originalCentroids = (0  until 6).flatMap( x=> (0 until 10).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 100, expectedKTolerance = 40)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 70 evenly distributed clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 70
    val originalCentroids = (0  until 7).flatMap( x=> (0 until 10).map(y => Array(2.0*x, 2.0*y)))
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 100, expectedKTolerance = 40)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }


  test("KMeans#guessK integration test for 80 evenly distributed clusters with a small epsilon (a.k.a. conservative).") {

      val Ks = 80

      val originalCentroids = (0  until 10).flatMap( x=> (0 until 8).map(y => Array(2.0*x, 2.0*y)))

      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

      val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 90, expectedKTolerance = 20)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

//      println("---------------")
//      originalCentroids.foreach(p => println(point2Str(p)))
//      println("---------------")
//      bestModel.clusterCenters.foreach(p => println(point2Str(p.point)))
//      println("---------------")

    }

  test("KMeans#guessK integration test for 200 clusters with a normal epsilon.") {

    val Ks = 200
    val random = new util.Random()

    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }

    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.25))

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 220, expectedKTolerance = 40)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  private def testGuessK(dataPoints: Seq[Point], epsilon: Double, expectedK: Int, expectedKTolerance: Int, runs: Int = 7) = {

    val kMeansTrainerFactory = (k: Int) => KMeansTrainer(k, 300, 1E-9)

    val shuffled = Random.shuffle(dataPoints).par

    println(s"Expected K, $expectedK, Data Size,${(0.7*dataPoints.size).toInt}")

    val guessedK = KMeans.guessK(shuffled, runs, kMeansTrainerFactory, epsilon)

    println(s"Best guess K = $guessedK; expected K = $expectedK; tolerance = $expectedKTolerance")

//    math.abs(guessedK - expectedK) should be <= expectedKTolerance

//    val (bestModel, sse) = KMeans.bestModel(shuffled, 5, kMeansTrainerFactory(guessedK))
    val (bestModel, sse) = KMeans.bestModel(shuffled, 5, kMeansTrainerFactory(expectedK))
    bestModel
  }

  private def point2Str(p: Point) = p.map(x => f"$x%+11.6f").mkString(", ")

  private def printOriginalVsActualCentroids(original: Seq[Point], actual: Seq[Point]) = {
    val originalKs = original.zipWithIndex.map(_.swap)
    val originalCentroids = originalKs.map(x => ClusterPoint(x._1, x._2))
    val originalModel = KMeans(originalCentroids.toList)

    println(f"${"K"}%3s | ${"Distance"}%11s | ${"Actual Center"}%24s | ${"Guessed Center"}%24s")
    originalModel.predict(actual).toList.sortBy(_.label._1).foreach { p =>
      println(f"${p.label._1}%3d | ${p.label._2}%11.6f | ${point2Str(originalKs(p.label._1)._2)}%24s | ${point2Str(p.point)}%24s")
    }
  }

}

