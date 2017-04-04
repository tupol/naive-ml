package tupol.ml.clustering

import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{ FunSuite, Matchers }
import tupol.ml._
import tupol.ml.pointops._
import tupol.ml.utils.ClusterGen2D
/**
 *
 */
class KMeansSpec extends FunSuite with Matchers with LazyLogging {

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
      logger.debug(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }

  }

  test("KMeans#train test 3 discs") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 2.0))
    val clusters = dataPoints(kCenters)
    val actual = KMeansTrainer(3, 200, 1E-8, 77977).train(clusters).clusterCenters.map(_.point)

    val epsilon = 0.2

    kCenters foreach { e =>
      val (cc, distance) = actual.map(p => (p, math.sqrt(distance2(p, e)))).sortWith(_._2 < _._2).head
      logger.debug(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }
  }

}

