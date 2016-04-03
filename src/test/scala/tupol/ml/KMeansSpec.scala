package tupol.ml

import org.scalatest.{FunSuite, Matchers}
import utils.ClusterGen2D

/**
  *
  */
class KMeansSpec extends FunSuite with Matchers {

  import ClusterGen2D._
  import KMeans._

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

    val expected = Array(1/3., 2/3.)
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
    assert(actual.map(_._2).toSeq.size === k)
    assert(actual.forall(x => dataPoints3.contains(x._2)))
  }

  test("KMeans#initialize test k = 2") {

    val k = 2
    val actual = initialize(k, dataPoints3)

    assert(actual.size === k)
    assert(actual.map(_._2).toSeq.size === k)
    assert(actual.forall(x => dataPoints3.contains(x._2)))
  }

  test("KMeans#initialize test k = 3") {

    val k = 3
    val actual = initialize(k, dataPoints3L)

    assert(actual.size === k)
    assert(actual.map(_._2).toSeq.size === k)
    assert(actual.forall(x => dataPoints3L.contains(x._2)))
  }

  test("KMeans#initialize test k = 200") {

    val k = 200
    val actual = initialize(k, dataPoints3L)

    assert(actual.size === k)
    assert(actual.map(_._2).toSeq.size === k)
    assert(actual.forall(x => dataPoints3L.contains(x._2)))
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

  test("MeansKs#train test 2 discs") {

    val expected = Seq(Array(0.0, 0.0), Array(2.0, 2.0)).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)
    val actual = train(2, 100, 0.1, dataPoints2L).map(_._2).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)

    val epsilon = 0.01

    actual.zip(expected).forall(t => math.sqrt(distance2(t._1, t._2)) < epsilon)
  }

  test("MeansKs#train test 3 discs") {

    val expected = Seq(Array(0.0, 0.0), Array(1.0, 0.0), Array(3.0, 0.0)).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)
    val actual = train(3, 100, 0.1, dataPoints3L).map(_._2).sortWith((a, b) =>
      a.zip(b).find(t => t._1 < t._2).isDefined)

    val epsilon = 0.01

    actual.zip(expected).forall(t => math.sqrt(distance2(t._1, t._2)) < epsilon)
  }

}
