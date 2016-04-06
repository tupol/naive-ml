package tupol.ml

import org.scalatest.{FunSuite, Matchers}
import utils.ClusterGen2D

/**
  *
  */
class KMeansGaussianSpec extends FunSuite with Matchers {

  import ClusterGen2D._

  val dataPoints1 = Seq(
    Array(0.0, 0.0)
  )

  val dataPoints2 = Seq(
    Array(1.0, 1.0),
    Array(2.0, 2.0)
  )

  val dataPoints3 = Seq(
        Array(0.0, 0.0),
        Array(1.0, 1.0),
        Array(0.0, 1.0)
      )


  val dataPoints1L = disc(1000, Array(0.0, 0.0))

  val dataPoints2L = (disc(1000, Array(0.0, 0.0)) ++ disc(1000, Array(2.0, 2.0)))

  val dataPoints3L = (disc(1000, Array(0.0, 0.0)) ++ disc(1000, Array(2.0, 0.0)) ++ disc(1000, Array(0.0, 2.0)))


  test("KMeansGaussian#train test 1 disc K=1") {

    val initialCentroids = Seq((0.0, Array(0.5, 0.5)))

    val kmeans = KMeansTrainer(1, 500, 0.1).train(initialCentroids, dataPoints1L)

    val kmg = KMeansGaussianTrainer(kmeans).train(dataPoints1L)

    val points = Seq(
      Array(0.0, 0.0),
      Array(0.0, 1.0),
      Array(1.0, 1.0),
      Array(0.0, 2.0),
      Array(0.0, 3.0)
    )

    val expectedProbabilities = Seq(1.0, 0.6, 0.005, 0.000009, 0.0000009)
    val actualProbabilities = points.map(kmg.predict)

    expectedProbabilities.zip(actualProbabilities).forall { case (e, a) => e >= a }

    println("Clusters")
    kmeans.clusterCenters.map(lpointToStr).foreach(println)
    println("Predictions")
    points.foreach { point =>
      println(f"${lpointToStr(kmeans.predict(point))}  ${kmg.predict(point)}%12.10f  ${pointToStr(kmg.predictByDimension(point))}")
    }
  }
  test("KMeansGaussian#train test 1 disc K=2") {

    val initialCentroids = Seq((0.0, Array(0.5, 0.5)))

    val kmeans = KMeansTrainer(2, 500, 0.1).train(dataPoints1L)

    val kmg = KMeansGaussianTrainer(kmeans).train(dataPoints1L)

    val points = Seq(
      Array(0.0, 0.0),
      Array(0.0, 1.0),
      Array(1.0, 1.0),
      Array(0.0, 2.0),
      Array(0.0, 3.0)
    )

    val expectedProbabilities = Seq(1.0, 0.6, 0.005, 0.000009, 0.0000009)
    val actualProbabilities = points.map(kmg.predict)

    expectedProbabilities.zip(actualProbabilities).forall { case (e, a) => e >= a }

    println("Clusters")
    kmeans.clusterCenters.map(lpointToStr).foreach(println)
    println("Predictions")
    points.foreach { point =>
      println(f"${lpointToStr(kmeans.predict(point))}  ${kmg.predict(point)}%12.10f  ${pointToStr(kmg.predictByDimension(point))}")
    }
  }

  test("KMeansGaussian#train test 2 discs K=2") {

    val initialCentroids = Seq((0.0, Array(0.5, 0.5)), (1.0, Array(2.5, 2.5)))

    val kmeans = KMeansTrainer(2, 500, 0.1).train(initialCentroids, dataPoints2L)

    val kmg = KMeansGaussianTrainer(kmeans).train(dataPoints2L)

    val points = Seq(
      Array(0.0, 0.0),
      Array(0.0, 1.0),
      Array(1.0, 1.0),
      Array(0.0, 2.0),
      Array(0.0, 3.0)
    )

    val expectedProbabilities = Seq(1.0, 0.6, 0.005, 0.000009, 0.0000009)
    val actualProbabilities = points.map(kmg.predict)

    expectedProbabilities.zip(actualProbabilities).forall { case (e, a) => e >= a }

    println("Clusters")
    kmeans.clusterCenters.map(lpointToStr).foreach(println)
    println("Predictions")
    points.foreach { point =>
      println(f"${lpointToStr(kmeans.predict(point))}  ${kmg.predict(point)}%12.10f  ${pointToStr(kmg.predictByDimension(point))}")
    }
  }


  private def pointToStr(point: Point) = point.map(x => f"$x%+12.10f").mkString("[", ", ", "]")
  private def lpointToStr(lp: LabeledPoint) = f"(${lp._1}%.0f ${pointToStr(lp._2)})"

}
