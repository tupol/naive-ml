package tupol

import scala.collection.parallel.ParSeq

/**
 *
 */
package object ml {

  type Point = Array[Double]

  abstract class LabeledPoint[L](label: L, point: Point) {
    override def toString = s"($label) ${point.mkString("[", ", ", "]")})"
  }

  case class DoubleLabeledPoint(label: Double, point: Point) extends LabeledPoint[Double](label, point)

  trait Predictor[T, P] {
    def predict(data: T): P

    def predict(data: Seq[T]): Seq[P] =
      predict(data.par).toList

    def predict(data: ParSeq[T]): ParSeq[P] =
      data.map(predict)
  }

  trait Trainer[T, P] {
    def train(data: Seq[T]): P = train(data.par)
    def train(data: ParSeq[T]): P
  }

  /**
   * Decorate the Point with new operations
   *
   * @param point
   */
  implicit class PointOps(point: Point) {

    def +(scalar: Double) = point.map(_ + scalar)

    def -(scalar: Double) = point.map(_ - scalar)

    def *(scalar: Double) = point.map(_ * scalar)

    def /(scalar: Double) = point.map(_ / scalar)

    def *(thatPoint: Point) = {
      (point :* thatPoint).sum
    }

    def :+(thatPoint: Point) = {
      require(point.size == thatPoint.size)
      point.zip(thatPoint).map { case (t, x) => t + x }
    }

    def :-(thatPoint: Point) = {
      require(point.size == thatPoint.size)
      point.zip(thatPoint).map { case (t, x) => t - x }
    }

    def :*(thatPoint: Point) = {
      require(point.size == thatPoint.size)
      point.zip(thatPoint).map { case (x, t) => x * t }
    }

    def distance2(thatPoint: Point): Double = {
      distance2ByDimension(thatPoint).sum
    }
    def distance2ByDimension(thatPoint: Point): Point = {
      point.zip(thatPoint).map(x => (x._2 - x._1) * (x._2 - x._1))
    }

    override def toString() = point.mkString("[", ", ", "]")
  }

  /**
   * Decorate the Sequences of Points with new operations
   * @param points
   */
  implicit class PointsOps(points: Seq[Point]) {

    lazy val size = points.size

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
      points.map(_.distance2ByDimension(mean)).sumByDimension / (size - 1)
    }

    def normalize() = {
      val sigma = variance.map(math.sqrt)
      points.map(v => v.zip(mean).zip(sigma).map { case ((x, mu), sig) => (x - mu) / sig })
    }
  }
  /**
   * Decorate the Sequences of Points with new operations
   * @param points
   */
  implicit class ParPointsOps(points: ParSeq[Point]) {

    val size = points.size

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
      points.map(_.distance2ByDimension(mean)).sumByDimension / (size - 1)
    }

    def normalize() = {
      val sigma = variance.map(math.sqrt)
      points.map(v => v.zip(mean).zip(sigma).map { case ((x, mu), sig) => (x - mu) / sig })
    }

  }

  /**
   * Square distance between two vectors
   *
   * @param vector1
   * @param vector2
   * @return
   */
  private[ml] def distance2(vector1: Point, vector2: Point): Double = {
    vector1.zip(vector2).map(x => (x._2 - x._1) * (x._2 - x._1)).sum
  }

  private[ml] def mean(vectors: ParSeq[Point]): Point = {
    require(vectors.size > 0)
    vectors.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
      .map(x => x / vectors.size)
  }

  private[ml] def mean(vectors: Seq[Point]): Point = mean(vectors.par)

  /**
   * This is a pseudo polynomial of 1 variable x
   * The functions map keys are function names or text representations; for example for a function f(x) = x * x,
   * the text representation should be x * x
   */
  case class CxFun(functions: Map[String, Double => Double]) extends Function1[Double, Double] {

    def this(functions: Seq[Double => Double]) =
      this(functions.zipWithIndex.map(_.swap).map {
        case (id, f) =>
          (f"$id%02d", f)
      }.toMap)

    /**
     * Create a new CxFun by mutiplying each function with a constant parameter
     *
     * @param parameters
     * @return
     */
    def withParameters(parameters: Seq[Double]): CxFun = {
      require(parameters.size == functions.size)
      val newFunctions = parameters.zip(functions).map { case (p, cxf) => (s"$p * ${cxf._1}", (x: Double) => p * cxf._2(x)) }
      CxFun(newFunctions.toMap)
    }

    private lazy val assembledFunction = functions.map(_._2).reduce((f1, f2) => (x: Double) => f1(x) + f2(x))

    override def apply(x: Double): Double = assembledFunction(x)

    override def toString: String = functions.map(_._1).mkString(" + ")

  }

}
