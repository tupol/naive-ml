package tupol.ml.regression

import org.scalatest.{ FunSuite, Matchers }
import tupol.ml.DoubleLabeledPoint

/**
 *
 */
class LinearRegressionSpec extends FunSuite with Matchers {

  import LinearRegression._

  test("LinearRegression#cost test 1") {

    val data = Seq(
      DoubleLabeledPoint(7.0, Array(1.0, 2.0)),
      DoubleLabeledPoint(6.0, Array(1.0, 3.0)),
      DoubleLabeledPoint(5.0, Array(1.0, 4.0)),
      DoubleLabeledPoint(4.0, Array(1.0, 5.0))
    )
    val theta = Array(0.1, 0.2)
    val tolerance = 0.001
    val expected = 11.9450
    val actual = cost(data, theta)
    val actualEpsilon = math.abs(actual - expected)

    actualEpsilon should be <= tolerance

  }

  test("LinearRegression#cost test 2") {

    val data = Seq(
      DoubleLabeledPoint(7.0, Array(1.0, 2.0, 3.0)),
      DoubleLabeledPoint(6.0, Array(1.0, 3.0, 4.0)),
      DoubleLabeledPoint(5.0, Array(1.0, 4.0, 5.0)),
      DoubleLabeledPoint(4.0, Array(1.0, 5.0, 6.0))
    )
    val theta = Array(0.1, 0.2, 0.3)
    val tolerance = 0.001
    val expected = 7.0175
    val actual = cost(data, theta)
    val actualEpsilon = math.abs(actual - expected)

    actualEpsilon should be <= tolerance

  }

  test("LinearRegression#train theta is zero") {
    val data = Seq(
      DoubleLabeledPoint(1.0, Array(1.0, 5.0)),
      DoubleLabeledPoint(6.0, Array(1.0, 2.0)),
      DoubleLabeledPoint(4.0, Array(1.0, 4.0)),
      DoubleLabeledPoint(2.0, Array(1.0, 5.0))
    )
    val theta = Array(0.0, 0.0)
    val maxIter = 1000
    val learningRate = 0.01

    val expectedTheta = Array(5.211465984067477, -0.5725906341810969)
    val expectedCost = 0.8554025700312651

    val rtheta = LinearRegressionTrainer(theta, maxIter, learningRate, LinearRegression.hypothesys).train(data)
    rtheta.theta should be(expectedTheta)

    cost(data, rtheta.theta) should be(expectedCost)

  }

  test("LinearRegression#train theta is non zero") {
    val data = Seq(
      DoubleLabeledPoint(1.0, Array(1.0, 5.0)),
      DoubleLabeledPoint(6.0, Array(1.0, 2.0))
    )
    val theta = Array(0.5, 0.5)
    val maxIter = 10
    val learningRate = 0.1

    val expectedTheta = Array(1.596465801944336, 0.21987429259863284)
    val expectedCost = 4.646865103463483

    val rtheta = LinearRegressionTrainer(theta, maxIter, learningRate, LinearRegression.hypothesys).train(data)

    rtheta.theta should be(expectedTheta)

    cost(data, rtheta.theta) should be(expectedCost)
  }

  test("LinearRegressionOptimized#train theta is non zero") {
    val data = Seq(
      DoubleLabeledPoint(1.0, Array(1.0, 5.0)),
      DoubleLabeledPoint(6.0, Array(1.0, 2.0))
    )
    val theta = Array(0.5, 0.5)
    val maxIter = 10
    val learningRate = 1
    val tolerance = 0.1

    val expectedTheta = Array(1.7098632193403809, 0.19229353765009757)
    val expectedCost = 4.5116663759254

    val rtheta = LinearRegressionOptimizedTrainer(theta, maxIter, learningRate, tolerance, LinearRegression.hypothesys).train(data)

    //    rtheta.theta should be (expectedTheta)
    //
    //    cost(data, rtheta.theta) should be (expectedCost)

    println(rtheta.thetaHistory.size)
    rtheta.thetaHistory.map(_.mkString(", ")).foreach(println)
    rtheta.thetaHistory.map(LinearRegression.cost(data, _)).foreach(println)
  }

}
