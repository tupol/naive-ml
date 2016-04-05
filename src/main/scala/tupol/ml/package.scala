package tupol

/**
  *
  */
package object ml {

  type Point = Array[Double]
  type LabeledPoint = (Double, Point)

  trait Predictor[T, P] {
    def predict(data: T): P
    def predict(data: Seq[T]): Seq[P] =
      data.map(predict)
  }

  trait Trainer[T, P] {
    def train(data: Seq[T]): P
  }

  implicit class XPoint(thisPoint: Point) {

    def *(scalar: Double) = thisPoint.map(_ * scalar)

    def /(scalar: Double) = thisPoint.map(_ / scalar)

    def distance2(thatPoint: Point): Double = {
      distance2ByDimension(thatPoint).sum
    }
    def distance2ByDimension(thatPoint: Point): Point = {
      thisPoint.zip(thatPoint).map(x => (x._2 - x._1) * (x._2 - x._1))
    }
  }

  implicit class XPoints(points: Seq[Point]) {

    lazy val size = points.size

    def *(scalar: Double) = points.map(_ * scalar)

    def /(scalar: Double) = points.map(_ / scalar)

    def mean(): Point = {
      require(size > 0)
      sumByDimension / size
    }

    def sumByDimension : Point = {
      require(size > 0)
      points.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
    }

    def variance(mean: Point): Point = {
      points.map(_.distance2ByDimension(mean)).sumByDimension / size
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
