package tupol.ml

/**
  *
  */
case class LinearRegression(theta: Point) extends Predictor[Point, LabeledPoint] {

  def predict(point: Point): LabeledPoint = {
    require (point.size == theta.size || point.size + 1 == theta.size)
    val pred = if(theta.size == point.size + 1)
      (theta.tail).zip(point).map{case(t, x) => t * x}.sum
    else
      (theta).zip(point).map{case(t, x) => t * x}.sum
    (pred, point)
  }

}

object LinearRegression {

  def normalize(data: Seq[Point]) = {
    val size = data.size
    val sum = data.reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
    val mean = sum.map(_ / size)
    val variance = data.map(v => v.zip(mean)
      .map{ case(x, avg) => (x - avg) * (x - avg)})
      .reduce((v1, v2) => v1.zip(v2).map(x => x._1 + x._2))
      .map(_ / size)
    val sigma = variance.map(math.sqrt)
    data.map(v => v.zip(mean).zip(sigma).map{case ((x, mu), sig) => (x - mu) / sig})
  }

  def cost(data: Seq[LabeledPoint], theta: Point) = {
    val sqrErrors = errors(data, theta).map(e => e * e)
    sqrErrors.sum / data.size / 2
  }

  def errors(data: Seq[LabeledPoint], theta: Point) = {
    val X = data.map(_._2)
    val Y = data.map(_._1)
    val predictions = LinearRegression(theta).predict(X)
    predictions.map(_._1).zip(Y).map{ case(p, y) => (p - y) }
  }


  def train(data: Seq[LabeledPoint], theta: Seq[Point], maxIter: Int = 10, epsilon: Double = 10E-4) = {


    def train(theta: Seq[Point], step: Int, done: Boolean) = {
      if(step == maxIter || done)
        theta
      else {
//        val differential =
      }
    }
  }


}
