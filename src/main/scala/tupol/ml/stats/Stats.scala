package tupol.ml.stats

import tupol.ml._
import tupol.ml.stats.statsops._

import scala.language.implicitConversions

/**
 * Represents statistics data
 *
 * @param min minimal value
 * @param avg mean value
 * @param max maximal value
 * @param sse sum of squared errors
 * @param variance variance
 * @param stdev standard deviation
 * @tparam T either [[tupol.ml.Point]] or Double
 */
case class Stats[T](count: Long, min: T, avg: T, max: T, variance: T, stdev: T, sse: T)

/** Calculates statistics data */
object Stats {

  import scala.math.sqrt


  def fromPoint(population: Point): Stats[Point] = {
    val zero = (0 until population.size).map(_ => 0.0).toArray
    new Stats(1, population, population, population, zero, zero, zero)
  }

  def fromPoints(population: Iterable[Point]): Stats[Point] = {
    population.map(fromPoint).reduce(_ |+| _)
  }

  /**
   * Calculates statistics data of the [[Iterable]] of Double's
   *
   * @note doesn't work with empty Iterable's
   * @param population non-empty Iterable[Double]
   * @return [[Stats]]
   */
  def fromDoubles(population: Iterable[Double]): Stats[Double] = {
    val count = population.size
    val avg = population.sum / count
    val sse = population.map { d => val er = (d - avg); er * er }.sum
    val variance = sse / (count - 1) // See https://en.wikipedia.org/wiki/Bessel%27s_correction
    val sd = sqrt(variance)
    new Stats(count, population.min, avg, population.max, variance, sd, sse)
  }

  /**
   * Initialises the stats from a single double
   * @param value
   * @return
   */
  def fromDouble(value: Double): Stats[Double] = Stats(1, value, value, value, 0.0, 0.0, 0.0)

  val zeroDouble: Stats[Double] = Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  val zeroPoint = {
    val zero = Array.emptyDoubleArray
    Stats(0, zero, zero, zero, zero, zero, zero)
  }

  def zeroPoint(point: Point) = {
    val zero = (0 until point.size).map(_ => 0.0).toArray
    new Stats(0, point, point, point, zero, zero, zero)
  }

}

