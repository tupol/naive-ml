package tupol

/**
 *
 */
package object ml {

  type Point = Array[Double]

  case class DoubleLabeledPoint(label: Double, point: Point) extends LabeledPoint[Double](label, point)

  case class KMeansLabeledPoint(label: (Double, Double), point: Point) extends LabeledPoint[(Double, Double)](label, point)

  abstract class LabeledPoint[L](label: L, point: Point)

  trait Predictor[T, P] {
    def predict(data: T): P
    def predict(data: Seq[T]): Seq[P] =
      data.map(predict)
  }

  trait Trainer[T, P] {
    def train(data: Seq[T]): P
  }

  implicit class PointOps(thisPoint: Point) {

    def *(scalar: Double) = thisPoint.map(_ * scalar)

    def *(thatPoint: Point) = {
      require(thisPoint.size == thatPoint.size)
      thisPoint.zip(thatPoint).map { case (t, x) => t * x }.sum
    }

    def -(thatPoint: Point) = {
      require(thisPoint.size == thatPoint.size)
      thisPoint.zip(thatPoint).map { case (t, x) => t - x }
    }

    def /(scalar: Double) = thisPoint.map(_ / scalar)

    def distance2(thatPoint: Point): Double = {
      distance2ByDimension(thatPoint).sum
    }
    def distance2ByDimension(thatPoint: Point): Point = {
      thisPoint.zip(thatPoint).map(x => (x._2 - x._1) * (x._2 - x._1))
    }
  }

  implicit class PointsOps(points: Seq[Point]) {

    lazy val size = points.size

    def *(scalar: Double) = points.map(_ * scalar)

    def /(scalar: Double) = points.map(_ / scalar)

    def variance(mean: Point): Point = {
      points.map(_.distance2ByDimension(mean)).sumByDimension / size
    }

    lazy val mean: Point = {
      require(size > 0)
      sumByDimension / size
    }

    lazy val sumByDimension: Point = {
      require(size > 0)
      points.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
    }

    lazy val variance: Point = {
      points.map(_.distance2ByDimension(mean)).sumByDimension / size
    }

    def normalize() = {
      val sigma = variance.map(math.sqrt)
      points.map(v => v.zip(mean).zip(sigma).map { case ((x, mu), sig) => (x - mu) / sig })
    }

  }

  /**
   * Square distance between two vectors
   * @param vector1
   * @param vector2
   * @return
   */
  def distance2(vector1: Point, vector2: Point): Double = {
    vector1.zip(vector2).map(x => (x._2 - x._1) * (x._2 - x._1)).sum
  }

  def mean(vectors: Seq[Point]): Point = {
    require(vectors.size > 0)
    vectors.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
      .map(x => x / vectors.size)
  }

}
