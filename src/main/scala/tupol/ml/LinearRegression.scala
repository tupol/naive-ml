package tupol.ml

/**
  *
  */
case class LinearRegression(thetaHistory: Seq[Point]) extends Predictor[Point, LabeledPoint] {

  import LinearRegression._

  def this(theta: Point) = this(Seq(theta))

  lazy val theta = thetaHistory.head

  def predict(point: Point): LabeledPoint = {
    require (point.size == theta.size || point.size + 1 == theta.size)
    val pred = hypothesys(point, theta)
    (pred, point)
  }

}

object LinearRegression {

  def hypothesys(point: Point, theta: Point): Double = {
    if(theta.size == point.size + 1)
      theta.head + theta.tail * point
    else
      theta * point
  }

  def sse(data: Seq[LabeledPoint], theta: Point) =
    errors(data, theta).map(e => e * e).sum

  def cost(data: Seq[LabeledPoint], theta: Point) = {
    sse(data, theta) / data.size / 2
  }

  def errors(data: Seq[LabeledPoint], theta: Point) = {
    val X = data.map(_._2)
    val Y = data.map(_._1)
    val predictions = new LinearRegression(theta).predict(X)
    predictions.map(_._1).zip(Y).map{ case(p, y) => (p - y) }
  }

}

case class LinearRegressionTrainer(theta: Point, maxIter: Int = 10, learningRate: Double = 0.01, hypothesys: (Point, Point) => Double)
  extends Trainer[LabeledPoint, LinearRegression] {

  def train(data: Seq[LabeledPoint]) = {

    def train(thetas: Seq[Point], step: Int, done: Boolean): Seq[Point] = {
      if(step == maxIter - 1 || done)
        thetas
      else {
        val oldTheta = thetas.head
        val differential = data.map{case (y, x) => (x * (hypothesys(x, oldTheta) - y))}.sumByDimension / data.size
        val newTheta = oldTheta - differential  * learningRate

        val oldCost = LinearRegression.cost(data, oldTheta)
        val newCost = LinearRegression.cost(data, newTheta)
        val done = newCost >= oldCost

        train(newTheta +: thetas, step+1, done)
      }
    }
    val thetas = train(Seq(theta), 0, false)
    LinearRegression(thetas)

  }
}


case class LinearRegressionOptimizedTrainer(theta: Point, maxIter: Int = 10, learningRate: Double = 0.01, tolerance: Double = 10E-4, hypothesys: (Point, Point) => Double)
  extends Trainer[LabeledPoint, LinearRegression] {

  def train(data: Seq[LabeledPoint]) = {

    def train(thetas: Seq[Point], step: Int, learningRate: Double, done: Boolean): Seq[Point] = {
      if(step == maxIter - 1 || done)
        thetas
      else {
        val oldTheta = thetas.head
        val differential = data.map{case (y, x) => (x * (hypothesys(x, oldTheta) - y))}.sumByDimension / data.size
        val newTheta = oldTheta - differential  * learningRate

        val oldCost = LinearRegression.cost(data, oldTheta)
        val newCost = LinearRegression.cost(data, newTheta)

        if(newCost - oldCost > 0) {
          // the cost is increasing, so let's reduce the learning rate, but change nothing else
          train(thetas, step, learningRate/3, false)
        } else {
          val done = oldCost - newCost <= tolerance
          train(newTheta +: thetas, step+1, learningRate, done)
        }
      }
    }
    val thetas = train(Seq(theta), 0, learningRate, false)
    LinearRegression(thetas)

  }
}
