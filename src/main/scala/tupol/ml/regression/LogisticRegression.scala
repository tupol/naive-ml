package tupol.ml.regression

import tupol.ml._

import scala.collection.parallel.ParSeq
import scala.math._

/**
 *
 */
case class LogisticRegression(thetaHistory: ParSeq[Point]) extends Predictor[Point, DoubleLabeledPoint] {

  import LogisticRegression._

  def this(theta: Point) = this(ParSeq(theta))

  lazy val theta = thetaHistory.head

  def predict(point: Point): DoubleLabeledPoint = {
    require(point.size == theta.size || point.size + 1 == theta.size)
    val pred = hypothesys(point, theta)
    DoubleLabeledPoint(if (pred < 0.5) 0.0 else 1.0, point)
  }

}

object LogisticRegression {

  def sigmoid(x: Double): Double = 1 / (1 + exp(-x))

  def sigmoid(x: ParSeq[Double]): ParSeq[Double] = x.map(sigmoid)

  def hypothesys(point: Point, theta: Point): Double = {
    if (theta.size == point.size + 1)
      theta.head + sigmoid(theta * point)
    else
      sigmoid(theta * point)
  }

  def cost(data: ParSeq[DoubleLabeledPoint], theta: Point, lambda: Double): Double = {
    val X = data.map(_.point)
    val Y = data.map(_.label)
    val m = X.length
    val hyp = X.map(hypothesys(_, theta))
    val cost = -1 * (
      Y.zip(hyp).map { case (a, b) => a * log(b) }.sum +
      Y.map(1 - _).zip(hyp.map(1 - _)).map { case (a, b) => a * log(b) }.sum
    ) / m
    val regularization = lambda * theta.tail.map(x => x * x).sum / 2 / m

    cost + regularization

  }

  def gradient(data: ParSeq[DoubleLabeledPoint], theta: Point, lambda: Double): Point = {
    val X = data.map(_.point)
    val Y = data.map(_.label)
    val m = X.length
    val hyp = X.map(hypothesys(_, theta))

    val regularization = 0.0 +: theta.tail * (lambda / m)
    val err = hyp.zip(Y).map(hy => hy._1 - hy._2)
    val diff = X.zip(err).map { case (x, e) => x * e }.sumByDimension / m

    diff.zip(regularization).map { case (d, r) => d + r }
  }

}

case class LogisticRegressionTrainer(theta: Point, maxIter: Int = 10, learningRate: Double = 1, lambda: Double = 1, tolerance: Double = 1E-3)
    extends Trainer[DoubleLabeledPoint, LogisticRegression] {

  def train(data: ParSeq[DoubleLabeledPoint]) = {

    import LogisticRegression._

    def train(thetas: ParSeq[Point], step: Int, learningRate: Double, done: Boolean): ParSeq[Point] = {
      if (step == maxIter || done)
        thetas
      else {
        val oldTheta = thetas.head
        val grad = gradient(data, oldTheta, lambda)

        val newTheta = oldTheta :- grad * learningRate
        val oldCost = cost(data, oldTheta, lambda)
        val newCost = cost(data, newTheta, lambda)

        if (newCost - oldCost > 0) {
          // the cost is increasing, so let's reduce the learning rate, but change nothing else
          train(thetas, step, learningRate / 3, false)
        } else {
          val done = oldCost - newCost <= tolerance
          train(newTheta +: thetas, step + 1, learningRate, done)
        }

      }
    }
    val thetas = train(ParSeq(theta), 0, learningRate, false)
    LogisticRegression(thetas)

  }
}

