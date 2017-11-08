package tupol.ml.stats

import tupol.ml.Point

object statsops {

  implicit def double2Stats(value: Double): Stats[Double] = Stats.fromDouble(value)
  implicit def point2Stats(value: Point): Stats[Point] = Stats.fromPoint(value)
  implicit def points2Stats(values: Iterable[Point]): Stats[Point] = values.map(point2Stats).reduce(_ |+| _)
  implicit def doubleIterable2Stats(value: Iterable[Double]): Stats[Double] = Stats.fromDoubles(value)

  trait StatsOps[T] {
    def append(x: Stats[T], y: Stats[T]): Stats[T]
  }

  implicit object DoubleStatsOps extends StatsOps[Double] {

    def append(x: Stats[Double], y: Stats[Double]): Stats[Double] = {
      if(x.count == 0) y
      else if (y.count == 0) x
      else {
        val newCount = (x.count + y.count)
        val newMin = math.min(x.min, y.min)
        val newMax = math.max(x.max, y.max)
        val newAvg = (x.count * x.avg + y.count * y.avg) / newCount
        val delta = y.avg - x.avg
        val sse = x.sse + y.sse + delta * delta * x.count * y.count / newCount
        // See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        val newVariance = sse / (newCount - 1)
        val newStdev = math.sqrt(newVariance)
        Stats(newCount, newMin, newAvg, newMax, newVariance, newStdev, sse)
      }
    }
  }

  implicit object ArrayStatsOps extends StatsOps[Point] {

    def append(x: Stats[Point], y: Stats[Point]): Stats[Point] = {
      import tupol.ml.pointops._
      if(x.count == 0) y
      else if (y.count == 0) x
      else {
        val newCount = (x.count + y.count)
        val newMin = x.min.zip(y.min).map { case (a, b) => math.min(a, b) }
        val newMax = x.max.zip(y.max).map { case (a, b) => math.max(a, b) }
        val newAvg = x.avg.zip(y.avg).map { case (a, b) => (x.count * a + y.count * b) / newCount }
        // (x.count * x.avg + y.count * y.avg) / newCount
        val delta = x.avg.zip(y.avg).map { case (a, b) => a - b }
        val sse = x.sse + y.sse + delta * delta * x.count * y.count / newCount
        // See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        val newVariance = sse / (newCount - 1)
        val newStdev = newVariance.sqrt
        Stats(newCount, newMin, newAvg, newMax, newVariance, newStdev, sse)
      }
    }

  }

  trait OnlineStatsOps[T] {
    def |+|(stats: Stats[T])(implicit ops: StatsOps[T]): Stats[T]
  }

  implicit class DoubleStatsOnlineOps(val stats: Stats[Double]) extends OnlineStatsOps[Double] {
    override def |+|(stats: Stats[Double])(implicit ops: StatsOps[Double]): Stats[Double] = ops.append(this.stats, stats)
  }

  implicit class ArrayStatsOnlineOps(val stats: Stats[Point]) extends OnlineStatsOps[Point] {
    override def |+|(stats: Stats[Point])(implicit ops: StatsOps[Point]): Stats[Point] = ops.append(this.stats, stats)
  }

}
