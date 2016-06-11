package tupol.ml.clustering

import org.scalatest.{FunSuite, Matchers}
import tupol.ml._
import tupol.ml.utils.ClusterGen2D

import scala.collection.parallel.ParSeq

/**
 *
 */
class KMeansGuessKItTest extends FunSuite with Matchers {

  import ClusterGen2D._

  val CLUSTER_SIZE = 100
  
  val centers_2 = Seq(Array(0.0, 0.0), Array(0.0, 2.0))
  val centers_4 = centers_2 ++ Seq(Array(2.0, 0.0), Array(2.0, 2.0))
  val centers_6 = centers_4 ++ Seq(Array(2.0, 5.0), Array(5.0, 2.0))
  val centers_9 = centers_6 ++ Seq(Array(2.0, 5.0), Array(-5.0, -2.0), Array(-2.0, -5.0), Array(-5.0, -5.0))
  
  val RELAXED_SLOPE =  -0.1
  val NORMAL_SLOPE = -0.0333
  val CONSERVATIVE_SLOPE =  -0.01

  test("KMeans#guessK integration test for 2 clusters with a large epsilon (a.k.a. relaxed).") {

    val dataPoints = centers_2.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 2, expectedKTolerance = 0)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 2 clusters with a normal epsilon") {

    val dataPoints = centers_2.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 2, expectedKTolerance = 0)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 2 clusters with a small epsilon (a.k.a. conservative).") {

    val dataPoints = centers_2.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 8, expectedKTolerance = 2)

    printOriginalVsActualCentroids(centers_2, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a large epsilon (a.k.a. relaxed).") {

    val dataPoints = centers_4.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 3, expectedKTolerance = 1)

    printOriginalVsActualCentroids(centers_4, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a normal epsilon") {

    val dataPoints = centers_4.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 4, expectedKTolerance = 1)

    printOriginalVsActualCentroids(centers_4, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 4 clusters with a small epsilon (a.k.a. conservative).") {

    val dataPoints = centers_4.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 14, expectedKTolerance = 4)

    printOriginalVsActualCentroids(centers_4, bestModel.clusterCenters.map(_.point))

  }


    test("KMeans#guessK integration test for 9 clusters with a large epsilon (a.k.a. relaxed).") {

      val dataPoints = centers_9.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

      val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 6, expectedKTolerance = 3)

      printOriginalVsActualCentroids(centers_9, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 9 clusters with a normal epsilon") {

      val dataPoints = centers_9.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

      val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 10, expectedKTolerance = 2)

      printOriginalVsActualCentroids(centers_9, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 9 clusters with a small epsilon (a.k.a. conservative).") {

      val dataPoints = centers_9.map(p => disc(CLUSTER_SIZE, p)).reduce(_ ++ _).par

      val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 44, expectedKTolerance = 4)

      printOriginalVsActualCentroids(centers_9, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 20 clusters with a large epsilon (a.k.a. relaxed).") {

      val Ks = 20
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

      val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 18, expectedKTolerance = 4)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 20 clusters with a normal epsilon ") {

      val Ks = 20
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

      val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 20, expectedKTolerance = 5)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 20 clusters with a small epsilon (a.k.a. conservative).") {

      val Ks = 20
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

      val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 150, expectedKTolerance = 50)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

    test("KMeans#guessK integration test for 50 clusters with a large epsilon (a.k.a. relaxed). (a.k.a. relaxed).") {

      val Ks = 50
      val random = new util.Random()
      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

      val bestModel = testGuessK(dataPoints, epsilon = RELAXED_SLOPE, expectedK = 70, expectedKTolerance = 10)

      printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

    }

  test("KMeans#guessK integration test for 50 clusters with a normal epsilon ") {

    val Ks = 50
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 50, expectedKTolerance = 15)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

  test("KMeans#guessK integration test for 50 clusters with a small epsilon (a.k.a. conservative).") {

    val Ks = 50
    val random = new util.Random()
    val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }
    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = CONSERVATIVE_SLOPE, expectedK = 100, expectedKTolerance = 40)

    printOriginalVsActualCentroids(originalCentroids, bestModel.clusterCenters.map(_.point))

  }

    test("KMeans#guessK integration test for 80 clusters with a normal epsilon.") {

      val Ks = 80
      val random = new util.Random()

      val originalCentroids = (0 until Ks).map { _ => Array(random.nextInt(Ks).toDouble, random.nextInt(Ks).toDouble) }

//      val originalCentroids = (0  until 20 by 2).flatMap( x=> (0 until 16 by 2).map(y => Array(x.toDouble, y.toDouble)))

      val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

      val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 90, expectedKTolerance = 20)

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

    val dataPoints = originalCentroids.flatMap(p => disc(CLUSTER_SIZE, p, 0.5)).par

    val bestModel = testGuessK(dataPoints, epsilon = NORMAL_SLOPE, expectedK = 220, expectedKTolerance = 40)

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

