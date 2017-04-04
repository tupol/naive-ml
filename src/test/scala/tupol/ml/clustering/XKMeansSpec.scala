package tupol.ml.clustering

import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{ FunSuite, Matchers }
import tupol.ml._
import tupol.ml.pointops._
import tupol.ml.stats.Stats
import tupol.ml.utils.ClusterGen2D

/**
 *
 */
class XKMeansSpec extends FunSuite with Matchers with LazyLogging {

  import ClusterGen2D._
  import XKMeansTrainer._

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

  def dataPoints(centroids: Seq[Point]): Seq[Point] = centroids.flatMap(p => disc(600, p, 0.5, 77777))

  test("XKMeans#initialize test k = 0") {

    val k = 0
    intercept[IllegalArgumentException](initialize(k, dataPoints3))
  }

  test("XKMeans#initialize test k = 1") {

    val k = 1
    val actual = initialize(k, dataPoints3)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => dataPoints3.contains(x.statsByDim.avg)))
  }

  test("XKMeans#initialize test k = 2") {

    val k = 2
    val actual = initialize(k, dataPoints3)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => dataPoints3.contains(x.statsByDim.avg)))
  }

  test("XKMeans#initialize test k = 3") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 0.0))
    val k = kCenters.size
    val clusters = dataPoints(kCenters)
    val actual = initialize(k, clusters)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => clusters.contains(x.statsByDim.avg)))
  }

  test("XKMeans#initialize test k = 200") {

    val k = 200
    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 0.0))
    val clusters = dataPoints(kCenters)
    val actual = initialize(k, clusters)

    assert(actual.size === k)
    assert(actual.toSet.size === k)
    assert(actual.forall(x => clusters.contains(x.statsByDim.avg)))
  }

  test("XKMeans#distance2 test 1") {

    val v1 = Array(0.0, 0.0)
    val v2 = Array(0.0, 1.0)

    val expected = 1
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("XKMeans#distance2 test 2") {

    val v1 = Array(1.0, 0.0)
    val v2 = Array(0.0, 1.0)

    val expected = 2
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("XKMeans#distance2 test 3") {

    val v1 = Array(1.0, 1.0)
    val v2 = Array(1.0, 1.0)

    val expected = 0
    val actual = distance2(v1, v2)

    assert(actual === expected)
  }

  test("KMeansGaussian#train test 1 disc K=1") {

    val kCenters = Seq(Array(0.5, 0.5))
    val clusters = dataPoints(kCenters)

    val model = XKMeansTrainer(1, 500, 0.1).train(clusters)

    val expectedProbabilities = Seq(1.0, 0.6, 0.005, 0.000009, 0.0000009)
    val actualProbabilities = kCenters.map(model.predict)

    expectedProbabilities.zip(actualProbabilities.map(_.probability)).forall { case (e, a) => a <= 1.1 * e && a >= 0.9 * e }

  }

  test("XKMeans#train and predict on 2 discs") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0))
    val clusters = dataPoints(kCenters)

    val model = XKMeansTrainer(2, 200, 1E-6).train(clusters)
    val actual = model.clusterCenters.values.map(_.statsByDim.avg).toSeq

    val epsilon = 0.2

    kCenters foreach { e =>
      val (cc, distance) = actual.map(p => (p, math.sqrt(distance2(p, e)))).sortWith(_._2 < _._2).head
      logger.debug(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }

    val expectedProbabilities = Seq(1.0, 0.6, 0.005, 0.000009, 0.0000009)
    val actualProbabilities = kCenters.map(model.predict)

    expectedProbabilities.zip(actualProbabilities.map(_.probability)).forall { case (e, a) => a <= 1.1 * e && a >= 0.9 * e }

  }

  test("XKMeans#train test 3 discs") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 2.0))
    val clusters = dataPoints(kCenters)
    val actual = XKMeansTrainer(3, 200, 1E-8, 77977).train(clusters).clusterCenters.values.toSeq

    val epsilon = 0.2

    kCenters foreach { e =>
      val (cc, distance) = actual.map(p => (p, math.sqrt(distance2(p.point, e)))).sortWith(_._2 < _._2).head
      logger.debug(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.point.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }
  }

  test("XKMeans#metaclustering???") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 2.0))
    val k = kCenters.size
    val clusters = dataPoints(kCenters)
    val model = XKMeansTrainer(14, 200, 1E-8).train(clusters)
    val actual = model.clusterCenters.values

    def stats2Str(stats: Stats[Point]) = s"Stats(${stats.count},${stats.min.toSeq},${stats.avg.toSeq},${stats.max.toSeq},${stats.variance.toSeq},${stats.stdev.toSeq},${stats.sse.toSeq})"

    def createMetaClusters(pairs: Iterable[(Int, Int)]): Seq[Set[Int]] = pairs
      .foldLeft(Seq[Set[Int]]()) { (acc, p) =>
        val accf = acc.filter(seq => seq.contains(p._1) || seq.contains(p._2))
        val accnf = acc.filterNot(x => accf.contains(x))
        if (accf.isEmpty) Set(p._1, p._2) +: accnf
        else accf.map(x => x + p._1 + p._2) ++ accnf
      }

    val dists = actual.map { ok =>
      val (dist, closestK) = actual.filterNot(_.k == ok.k).map { x => (x.statsByDim.avg.sqdist(ok.statsByDim.avg), x) }.minBy(_._1)
      (ok, closestK, dist)
    }

    //    println("-----------------------")
    //    actual.toSeq.sortBy(_.k).foreach(p => println(f"${p.k}%2d | ${stats2Str(p.statsByDim)}"))

    //    println("-----------------------")
    //    MetaCluster.fromModel(model, 0.001).map(x => x.name).foreach(println)

  }

}

case class MetaCluster(clusters: Set[Cluster]) {
  def name = clusters.toSeq.map(_.k).sorted.mkString(",")
}

object MetaCluster {

  def fromModel(model: XKMeans, probabilityThreshold: Double = 0.01): Seq[MetaCluster] = {
    val clusters = model.clusterCenters.values
    val closeClusters: Iterable[(Cluster, Cluster)] = clusters.map { ok =>
      val newKs = clusters.filterNot(_ == ok)
      (ok, new XKMeans(newKs.toSeq).predict(ok.point))
    }
      .filter(_._2.probability > probabilityThreshold)
      .map(x => (x._1, model.clusterCenters(x._2.k)))

    val metaK = createMetaClusters(closeClusters)

    createMetaClusters(clusters.map(cc => (cc, cc)), metaK.map(_.clusters))

  }

  private def createMetaClusters(pairs: Iterable[(Cluster, Cluster)], initialMKs: Seq[Set[Cluster]] = Seq[Set[Cluster]]()): Seq[MetaCluster] = pairs
    .foldLeft(initialMKs) { (acc, p) =>
      val accf = acc.filter(seq => seq.contains(p._1) || seq.contains(p._2))
      val accnf = acc.filterNot(x => accf.contains(x))
      if (accf.isEmpty) Set(p._1, p._2) +: accnf
      else accf.map(x => x + p._1 + p._2) ++ accnf
    }.map(MetaCluster(_))

}

