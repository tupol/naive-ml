package tupol.ml

import scala.collection.parallel.ParSeq

/**
 * Additional operations for linalg.Point
 */
package object pointops {

  /**
   * Additional operations for Double with Point
   */
  implicit class DoubleOps(val scalar: Double) extends AnyVal {

    def +(vector: Point): Point = {
      opByDim(vector, (x1: Double, x2: Double) => x1 + x2)
    }

    def *(vector: Point): Point = {
      opByDim(vector, (x1: Double, x2: Double) => x1 * x2)
    }

    def -(vector: Point): Point = {
      opByDim(vector, (x1: Double, x2: Double) => x1 - x2)
    }

    def /(vector: Point): Point = {
      opByDim(vector, (x1: Double, x2: Double) => x1 / x2)
    }

    private[pointops] def opByDim(vector: Point, op: (Double, Double) => Double) =
      vector.toSeq.map { op(scalar, _) }.toArray

  }

  /**
   * Added operations by dimension
   * @param self
   */
  implicit class PointOps(val self: Point) extends AnyVal {

    /**
     * Square distance between two vectors
     *
     * @param that
     * @return
     */
    def sqdist(that: Point): Double = {
      self.zip(that).map(x => (x._2 - x._1) * (x._2 - x._1)).sum
    }

    /**
     * Squared distances by dimension
     *
     * @param that
     * @return
     */
    def sqdistByDim(that: Point): Point = {
      require(self.size == that.size, "vectors should have same size")
      self.zip(that).map {
        case (x1, x2) => math.pow(x1 - x2, 2)
      }
    }

    /**
     * Add 2 vectors, dimension by dimension
     *
     * @param that
     * @return
     */
    def +(that: Point): Point = {
      require(self.size == that.size)
      op(self, that, (x1: Double, x2: Double) => x1 + x2)
    }

    /**
     * Add each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def +(scalar: Double): Point = {
      op(self, scalar, (x1: Double, x2: Double) => x1 + x2)
    }

    /**
     * Subtract that vector from this vector, dimension by dimension
     *
     * @param that
     * @return
     */
    def -(that: Point): Point = {
      require(self.size == that.size)
      op(self, that, (x1: Double, x2: Double) => x1 - x2)
    }

    /**
     * Subtract each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def -(scalar: Double): Point = {
      op(self, scalar, (x1: Double, x2: Double) => x1 - x2)
    }

    /**
     * Change the sign of each value inside the vector
     *
     * @return
     */
    def unary_- : Point = self.toSeq.map(-_).toArray

    /**
     * Calculate the exponential by dimension
     *
     * @return
     */
    def exp: Point = map(math.exp)

    /**
     * Calculates the square root by dimension
     *
     * @return new vector
     */
    def sqrt: Point = map(math.sqrt)

    /**
     * Calculate the square by dimension
     *
     * @return
     */
    def sqr: Point = map(x => x * x)

    /**
     * Multiply this vector with that vector, dimension by dimension
     *
     * @param that
     * @return
     */
    def *(that: Point): Point = {
      require(self.size == that.size)
      op(self, that, (x1: Double, x2: Double) => x1 * x2)
    }

    def |*|(that: Point): Double = {
      require(self.size == that.size)
      op(self, that, (x1: Double, x2: Double) => x1 * x2).sum
    }

    /**
     * Multiply each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def *(scalar: Double): Point = {
      op(self, scalar, (x1: Double, x2: Double) => x1 * x2)
    }

    /**
     * Divide each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def /(scalar: Double): Point = {
      op(self, scalar, (x1: Double, x2: Double) => x1 / x2)
    }

    /**
     * Divide this vector with that vector, dimension by dimension
     *
     * @param that
     * @return
     */
    def /(that: Point): Point = {
      require(self.size == that.size)
      op(self, that, (x1: Double, x2: Double) => x1 / x2)
    }

    def map(op: (Double) => Double): Point = this.op(self, op)

    private[pointops] def op(v1: Point, v2: Point, op: (Double, Double) => Double): Point =
      v1.zip(v2).map { case (x1, x2) => op(x1, x2) }

    private[pointops] def op(v1: Point, scalar: Double, op: (Double, Double) => Double): Point =
      v1.toSeq.map { op(_, scalar) }.toArray

    private[pointops] def op(v1: Point, op: (Double) => Double): Point =
      v1.toSeq.map(op).toArray
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
    vectors.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2)) / vectors.size
  }

  private[ml] def mean(vectors: Seq[Point]): Point = mean(vectors.par)

  /**
   * Decorate the Sequences of Points with new operations
   * @param points
   */
  implicit class ParPointsOps(points: ParSeq[Point]) {

    lazy val size = points.size

    def *(scalar: Double) = points.map(_ * scalar)

    def /(scalar: Double) = points.map(_ / scalar)

    def variance(mean: Point): Point = points.map(_.sqdistByDim(mean)).sumByDimension / size

    lazy val mean: Point = {
      require(size > 0)
      sumByDimension / size
    }

    lazy val sumByDimension: Point = {
      require(size > 0)
      points.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
    }

    lazy val variance: Point = points.map(_.sqdistByDim(mean)).sumByDimension / (size - 1)

    lazy val sigma: Point = variance.toSeq.map(math.sqrt).toArray

    lazy val normalize = points.map(v => v.zip(mean).zip(sigma).map { case ((x, mu), sig) => (x - mu) / sig })


  }
}
