package tupol.ml.clustering

import org.scalatest.{ FunSuite, Matchers }
import tupol.ml._
import tupol.ml.utils.ClusterGen2D

/**
 *
 */
class KMeansSpec extends FunSuite with Matchers {

  import ClusterGen2D._
  import KMeansTrainer._

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

  def dataPoints(centroids: Seq[Point]): Seq[Point] = centroids.flatMap(p => disc(300, p, 0.5))

  test("KMeans#mean test 1") {

    val expected = Array(0.0, 0.0)
    val actual = mean(dataPoints1)

    assert(actual === expected)
  }

  test("KMeans#mean test 2") {

    val expected = Array(1.5, 1.5)
    val actual = mean(dataPoints2)

    assert(actual === expected)
  }

  test("KMeans#mean test 3") {

    val expected = Array(1 / 3.0, 2 / 3.0)
    val actual = mean(dataPoints3)

    assert(actual === expected)
  }

  test("KMeans#initialize test k = 0") {

    val k = 0
    intercept[IllegalArgumentException](initialize(k, dataPoints3))
  }

  test("KMeans#initialize test k = 1") {

    val k = 1
    val actual = initialize(k, dataPoints3)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => dataPoints3.contains(x.point)))
  }

  test("KMeans#initialize test k = 2") {

    val k = 2
    val actual = initialize(k, dataPoints3)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => dataPoints3.contains(x.point)))
  }

  test("KMeans#initialize test k = 3") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 0.0))
    val k = kCenters.size
    val clusters = dataPoints(kCenters)
    val actual = initialize(k, clusters)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => clusters.contains(x.point)))
  }

  test("KMeans#initialize test k = 200") {

    val k = 200
    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 0.0))
    val clusters = dataPoints(kCenters)
    val actual = initialize(k, clusters)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => clusters.contains(x.point)))
  }

  test("KMeans#distance2 test 1") {

    val v1 = Array(0.0, 0.0)
    val v2 = Array(0.0, 1.0)

    val expected = 1
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("KMeans#distance2 test 2") {

    val v1 = Array(1.0, 0.0)
    val v2 = Array(0.0, 1.0)

    val expected = 2
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("KMeans#distance2 test 3") {

    val v1 = Array(1.0, 1.0)
    val v2 = Array(1.0, 1.0)

    val expected = 0
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("KMeans#train test 2 discs") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0))
    val k = kCenters.size
    val clusters = dataPoints(kCenters)

    val actual = KMeansTrainer(2, 200, 1E-6).train(clusters).clusterCenters.map(_.point)

    val epsilon = 0.1

    kCenters foreach { e =>
      val (cc, distance) = actual.map(p => (p, math.sqrt(distance2(p, e)))).sortWith(_._2 < _._2).head
      println(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }

  }

  test("KMeans#train test 3 discs") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 2.0))
    val k = kCenters.size
    val clusters = dataPoints(kCenters)
    val actual = KMeansTrainer(3, 200, 1E-8).train(clusters).clusterCenters.map(_.point)

    val epsilon = 0.1

    kCenters foreach { e =>
      val (cc, distance) = actual.map(p => (p, math.sqrt(distance2(p, e)))).sortWith(_._2 < _._2).head
      println(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }
  }

  test("KMeans#chooseK from a synthetic data set of SSEs") {

    val sses = Seq(
      (20, 2.290424E+05),
      (30, 1.710782E+05),
      (40, 1.319024E+05),
      (50, 1.044779E+05),
      (60, 9.526762E+04),
      (70, 8.612571E+04),
      (80, 7.384531E+04),
      (90, 6.687644E+04),
      (100, 6.144810E+04),
      (110, 5.601230E+04),
      (120, 5.439095E+04),
      (130, 5.187239E+04),
      (140, 4.892317E+04),
      (150, 4.542728E+04),
      (160, 4.289776E+04),
      (170, 4.147563E+04),
      (180, 3.914782E+04),
      (190, 3.830164E+04),
      (200, 3.681869E+04)
    )

    val expectedK = 135
    val guessedK = KMeans.chooseK(sses, 0.0003)
    println(s"guessedK = $guessedK")
    math.abs(guessedK - expectedK) should be <= 10
  }

}

