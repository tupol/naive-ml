package tupol.ml.utils

/**
 * Generate 2D cluster data in different shapes
 */
object ClusterGen2D {

  type Coordinate = Array[Double]

  def rectangle(points: Int = 200, origin: Coordinate = Array(-0.5, -0.5),
    width: Double = 1, height: Double = 1): Seq[Coordinate] = {
    require(points > 0, s"The number of points to be generated ($points) must be greater than 0.")
    (0 until points).
      map(x => Array(math.random * width + origin(0), math.random * height + origin(1)))
  }

  def square(points: Int = 200, origin: Coordinate = Array(-0.5, -0.5), side: Double = 1): Seq[Coordinate] =
    rectangle(points, origin, side, side)

  def disc(points: Int = 200, center: Coordinate = Array(0, 0), radius: Double = 1): Seq[Coordinate] =
    ring(points, center, 0, radius)

  def ring(points: Int = 200, center: Coordinate = Array(0, 0), minRadius: Double = 0.5, maxRadius: Double = 1): Seq[Coordinate] =
    sector(points, center, minRadius, maxRadius, 0, 2 * math.Pi)

  def sector(points: Int = 200, center: Coordinate = Array(0, 0), minRadius: Double = 0.5, maxRadius: Double = 1,
    minTheta: Double = 0, maxTheta: Double = 2 * math.Pi): Seq[Coordinate] = {
    require(points > 0, s"The number of points to be generated ($points) must be greater than 0.")
    require(minRadius >= 0, s"The minimum radius ($minRadius) must be greater or equal to 0.")
    require(minRadius < maxRadius, s"The minimum radius ($minRadius) must be smaller than the maximum radius ($maxRadius).")
    import math._
    (0 until points).
      map(x => (math.random * (maxRadius - minRadius) + minRadius, math.random * (maxTheta - minTheta) + minTheta)).
      map {
        case (r, t) =>
          Array(center(0) + r * cos(t), center(1) + r * sin(t))
      }
  }

}
