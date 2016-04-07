package tupol.ml

import org.scalatest.{ FunSuite, Matchers }
import utils.ClusterGen2D

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

  val dataPoints2L = (disc(100, Array(0.0, 0.0)) ++ disc(100, Array(2.0, 2.0)))

  val dataPoints3L = (disc(100, Array(0.0, 0.0)) ++ disc(100, Array(2.0, 0.0)) ++ disc(100, Array(0.0, 2.0)))

  test("KMeans#mean test 1") {

    val expected = Array(0., 0.)
    val actual = mean(dataPoints1)

    assert(actual === expected)
  }

  test("KMeans#mean test 2") {

    val expected = Array(1.5, 1.5)
    val actual = mean(dataPoints2)

    assert(actual === expected)
  }

  test("KMeans#mean test 3") {

    val expected = Array(1 / 3., 2 / 3.)
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

    val k = 3
    val actual = initialize(k, dataPoints3L)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => dataPoints3L.contains(x.point)))
  }

  test("KMeans#initialize test k = 200") {

    val k = 200
    val actual = initialize(k, dataPoints3L)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => dataPoints3L.contains(x.point)))
  }

  test("KMeans#distance2 test 1") {

    val v1 = Array(0., 0.)
    val v2 = Array(0., 1.)

    val expected = 1
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("KMeans#distance2 test 2") {

    val v1 = Array(1., 0.)
    val v2 = Array(0., 1.)

    val expected = 2
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("KMeans#distance2 test 3") {

    val v1 = Array(1., 1.)
    val v2 = Array(1., 1.)

    val expected = 0
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("KMeans#train test 2 discs") {

    val expected = Seq(Array(0.0, 0.0), Array(2.0, 2.0)).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)
    val actual = KMeansTrainer(2, 100, 0.1).train(dataPoints2L).clusterCenters.map(_.point).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)

    val epsilon = 0.01

    actual.zip(expected).forall(t => math.sqrt(distance2(t._1, t._2)) < epsilon)
  }

  test("KMeans#train test 3 discs") {

    val expected = Seq(Array(0.0, 0.0), Array(1.0, 0.0), Array(3.0, 0.0)).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)
    val actual = KMeansTrainer(3, 100, 0.1).train(dataPoints3L).clusterCenters.map(_.point).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)

    val epsilon = 0.01

    actual.zip(expected).forall(t => math.sqrt(distance2(t._1, t._2)) < epsilon)
  }

  test("KMeans#chooseK ") {

    val sses = Seq(
      (020.0, 2.290424E+05),
      (030.0, 1.710782E+05),
      (040.0, 1.319024E+05),
      (050.0, 1.044779E+05),
      (060.0, 9.526762E+04),
      (070.0, 8.612571E+04),
      (080.0, 7.384531E+04),
      (090.0, 6.687644E+04),
      (100.0, 6.144810E+04),
      (110.0, 5.601230E+04),
      (120.0, 5.439095E+04),
      (130.0, 5.187239E+04),
      (140.0, 4.892317E+04),
      (150.0, 4.542728E+04),
      (160.0, 4.289776E+04),
      (170.0, 4.147563E+04),
      (180.0, 3.914782E+04),
      (190.0, 3.830164E+04),
      (200.0, 3.681869E+04)
    )

    val expectedK = 140
    KMeans.chooseK(sses, 0.005, 200.) should be(expectedK)
  }

}
