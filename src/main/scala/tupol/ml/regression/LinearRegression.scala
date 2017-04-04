package tupol.ml.regression

import tupol.ml._
import tupol.ml.pointops._

/**
 *
 */
case class LinearRegression(thetaHistory: Seq[Point]) extends Predictor[Point, DoubleLabeledPoint] {

  import LinearRegression._

  def this(theta: Point) = this(Seq(theta))

  lazy val theta = thetaHistory.head

  def predict(point: Point): DoubleLabeledPoint = {
    require(point.size == theta.size || point.size + 1 == theta.size)
    val pred = hypothesys(point, theta)
    DoubleLabeledPoint(pred, point)
  }

}

object LinearRegression {

  def hypothesys(point: Point, theta: Point): Double = {
    if (theta.size == point.size + 1)
      theta.head + theta.tail |*| point
    else
      theta |*| point
  }

  def sse(data: Seq[DoubleLabeledPoint], theta: Point): Double =
    errors(data, theta).map(e => e * e).sum

  def cost(data: Seq[DoubleLabeledPoint], theta: Point): Double = {
    sse(data, theta) / data.size / 2
  }

  def errors(data: Seq[DoubleLabeledPoint], theta: Point): Seq[Double] = {
    val X = data.map(_.point)
    val Y = data.map(_.label)
    val predictions = new LinearRegression(theta).predict(X)
    predictions.map(_.label).zip(Y).map { case (p, y) => (p - y) }
  }

  def gradient(data: Seq[DoubleLabeledPoint], theta: Point): Point =
    data.map { dlp => (dlp.point * (hypothesys(dlp.point, theta) - dlp.label)) }.sumByDimension / data.size

}

case class LinearRegressionTrainer(theta: Point, maxIter: Int = 10, learningRate: Double = 0.01, hypothesys: (Point, Point) => Double)
    extends Trainer[DoubleLabeledPoint, LinearRegression] {

  import LinearRegression._

  def train(data: Seq[DoubleLabeledPoint]) = {

    def train(thetas: Seq[Point], step: Int, done: Boolean): Seq[Point] = {
      if (step == maxIter || done)
        thetas
      else {
        val oldTheta = thetas.head
        val grad = gradient(data, oldTheta)
        val newTheta = oldTheta - grad * learningRate
        val oldCost = cost(data, oldTheta)
        val newCost = cost(data, newTheta)
        val done = newCost >= oldCost
        train(newTheta +: thetas, step + 1, done)
      }
    }
    val thetas = train(Seq(theta), 0, false)
    LinearRegression(thetas)

  }
}

/** This is a naive optimization, nothing fancy. Maybe in the future some BFGS... */
case class LinearRegressionOptimizedTrainer(theta: Point, maxIter: Int = 10, learningRate: Double = 0.01, tolerance: Double = 10E-4, hypothesys: (Point, Point) => Double)
    extends Trainer[DoubleLabeledPoint, LinearRegression] {

  import LinearRegression._

  def train(data: Seq[DoubleLabeledPoint]) = {

    def train(thetas: Seq[Point], step: Int, learningRate: Double, done: Boolean): Seq[Point] = {
      if (step == maxIter || done)
        thetas
      else {
        val oldTheta = thetas.head
        val grad = gradient(data, oldTheta)
        val newTheta = oldTheta - grad * learningRate
        val oldCost = cost(data, oldTheta)
        val newCost = cost(data, newTheta)
        if (newCost - oldCost > 0) {
          // the cost is increasing, so let's reduce the learning rate, but change nothing else
          train(thetas, step, learningRate / 3, false)
        } else {
          val done = oldCost - newCost <= tolerance
          train(newTheta +: thetas, step + 1, learningRate, done)
        }
      }
    }
    val thetas = train(Seq(theta), 0, learningRate, false)
    LinearRegression(thetas)

  }
}
