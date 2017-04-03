package tupol.ml.clustering

import org.scalatest.{FunSuite, Matchers}
import tupol.ml._
import tupol.ml.pointops._
import tupol.ml.stats.Stats
import tupol.ml.utils.ClusterGen2D

/**
 *
 */
class XKMeansSpec extends FunSuite with Matchers {

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

  val dataPoints1L = disc(1000, Array(0.0, 0.0))

  val dataPoints2L = (disc(1000, Array(0.0, 0.0)) ++ disc(1000, Array(2.0, 2.0)))

  val dataPoints3L = (disc(1000, Array(0.0, 0.0)) ++ disc(1000, Array(2.0, 0.0)) ++ disc(1000, Array(0.0, 2.0)))


  def dataPoints(centroids: Seq[Point]): Seq[Point] = centroids.flatMap(p => disc(600, p, 0.5))

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

  test("XKMeans#train test 2 discs") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0))
    val clusters = dataPoints(kCenters)

    val model = XKMeansTrainer(2, 200, 1E-6).train(clusters)
    val actual = model.clusterCenters.values.map(_.statsByDim.avg).toSeq

    val epsilon = 0.1

    kCenters foreach { e =>
      val (cc, distance) = actual.map(p => (p, math.sqrt(distance2(p, e)))).sortWith(_._2 < _._2).head
      println(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }


    val expectedProbabilities = Seq(1.0, 0.6, 0.005, 0.000009, 0.0000009)
    val actualProbabilities = kCenters.map(model.predict)

    expectedProbabilities.zip(actualProbabilities.map(_.probability)).forall { case (e, a) => e >= a }

    println("Clusters")
    model.clusterCenters.toSeq.map(cPointToStr).foreach(println)
    println("Predictions")
    points.toSeq.foreach { point =>
      println(f"${predPointToStr(kmeans.predict(point))}  ${kmg.predict(point).probability}%12.10f  ${(kmg.predict(point))}")
    }

  }

  test("XKMeans#train test 3 discs") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 2.0))
    val clusters = dataPoints(kCenters)
    val actual = XKMeansTrainer(3, 200, 1E-8).train(clusters).clusterCenters.values

    val epsilon = 0.2

    kCenters foreach { e =>
      val (cc, distance) = actual.map(p => (p, math.sqrt(distance2(p, e)))).sortWith(_._2 < _._2).head
      println(s"For expected centroid ${e.mkString("[", ",", "]")}, the closest predicted cluster was found " +
        s"at ${cc.mkString("[", ",", "]")}; distance=$distance.")
      distance should be < epsilon
    }
  }

    test("XKMeans#metaclustering???") {

    val kCenters = Seq(Array(0.0, 0.0), Array(0.0, 2.0), Array(2.0, 2.0))
    val k = kCenters.size
    val clusters = dataPoints(kCenters)
    val actual = XKMeansTrainer(8, 200, 1E-8).train(clusters).clusterCenters.values

    val epsilon = 0.2


    def stats2Str(stats: Stats[Point]) = s"Stats(${stats.count},${stats.min.toSeq},${stats.avg.toSeq},${stats.max.toSeq},${stats.variance.toSeq},${stats.stdev.toSeq},${stats.sse.toSeq})"

    def createMetaClusters(pairs: Iterable[(Int, Int)]) = pairs
      .foldLeft(Seq[Set[Int]]()) { (acc, p) =>
        val accf = acc.filter(seq => seq.contains(p._1) || seq.contains(p._2))
        val accnf = acc.filterNot(x => accf.contains(x))
        if (accf.isEmpty) Set(p._1, p._2) +: accnf
        else accf.map(x => x + p._1 + p._2) ++ accnf
      }


    actual.toSeq.sortBy(_.k).foreach(p => println(f"${p.k}%2d | ${stats2Str(p.statsByDim)}"))

    val ks = actual

    val dists = ks.map { ok =>
      val (dist, closestK) = ks.filterNot(_.k == ok.k).map { x => (x.statsByDim.avg.sqdist(ok.statsByDim.avg), x) }.minBy(_._1)
      (ok, closestK, dist)
    }


    println("-----------------------")


    val mkp0 = dists.map { case (ok, ck, dist) =>
      //      val threshold = 2.0 * (ok.stats.stdev + ck.stats.stdev)
      val threshold = 3.0 * math.max(ok.stats.stdev, ck.stats.stdev)
      (ok, ck, dist, threshold)
    }.filter(x => x._3 <= x._4)
      .map { case (ok, ck, dist, threshold) => (ok.k, ck.k) }

    dists.map { case (ok, ck, dist) =>
      val threshold = 3.0 * math.max(ok.stats.stdev, ck.stats.stdev)
      (ok, ck, dist, threshold)
    }.foreach { case (ok, ck, dist, threshold) =>
      println(s"* ${ok.k} to ${ck.k}: $dist  |  $threshold")
      if (dist <= threshold) {
        println(s"    2 * max(${(ok.stats.stdev)}, ${(ck.stats.stdev)}) = $threshold")
      }
    }

    val mks0 = createMetaClusters(mkp0)
    ks.map(_.k).filterNot(x => mks0.flatMap(z => z).toSeq.contains(x)).foreach(println)

    println("-----------------------")

    val zero = ks.map(cc => Set(cc.k))
    val mkp1 = ks.map { ok =>
      val newKs = ks.filterNot(_ == ok).toSeq
      (ok.k, new XKMeans(newKs).predict(ok.point))
    }
      .filter(_._2.probability > 0.01)
      .map(x => (x._1, x._2.k))

    val mks1 = createMetaClusters(mkp1)

    mks1.foreach(println)
    ks.map(_.k).filterNot(x => mks1.flatMap(z => z).toSeq.contains(x)).foreach(println)

  }

  private def pointToStr(point: Point) = point.toSeq.map(x => f"$x%+12.10f").mkString("[", ", ", "]")
  private def cPointToStr(lp: ClusterPoint) = f"(${lp.k}%3d ${pointToStr(lp.point)})"
  private def predPointToStr(lp: KMeansLabeledPoint) = f"(${lp.label._1}%3d (${lp.label._2}%+12.10f) ${pointToStr(lp.point)})"

}

