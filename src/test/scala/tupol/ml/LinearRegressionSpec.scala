package tupol.ml

import org.scalatest.{FunSuite, Matchers}

/**
  *
  */
class LinearRegressionSpec extends FunSuite with Matchers {

  import LinearRegression._

  test("LinearRegression#cost test 1") {

    val data = Seq(
      (7.0, Array(1.0, 2.0)),
      (6.0, Array(1.0, 3.0)),
      (5.0, Array(1.0, 4.0)),
      (4.0, Array(1.0, 5.0))
    )
    val theta =  Array(0.1, 0.2)
    val tolerance = 0.001
    val expected = 11.9450
    val actual = cost(data, theta)
    val actualEpsilon = math.abs(actual - expected)

    actualEpsilon should be <= tolerance

  }

  test("LinearRegression#cost test 2") {

    val data = Seq(
      (7.0, Array(1.0, 2.0, 3.0)),
      (6.0, Array(1.0, 3.0, 4.0)),
      (5.0, Array(1.0, 4.0, 5.0)),
      (4.0, Array(1.0, 5.0, 6.0))
    )
    val theta =  Array(0.1, 0.2, 0.3)
    val tolerance = 0.001
    val expected = 7.0175
    val actual = cost(data, theta)
    val actualEpsilon = math.abs(actual - expected)

    actualEpsilon should be <= tolerance

  }

  test("LinearRegression#train test is zero") {
    val data = Seq(
      (1.0, Array(1.0, 5.0)),
      (6.0, Array(1.0, 2.0)),
      (4.0, Array(1.0, 4.0)),
      (2.0, Array(1.0, 5.0))
    )
    val theta =  Array(0.0, 0.0)
    val maxIter = 1000
    val learningRate = 0.01

    val expectedTheta = Array(5.214754949594046, -0.573345912562104)
    val expectedCost = 0.8542642597709095

    val rtheta = LinearRegressionTrainer(theta, maxIter, learningRate, LinearRegression.hypothesys).train(data)
    rtheta.theta should be (expectedTheta)

    cost(data, rtheta.theta) should be (expectedCost)

  }

  test("LinearRegression#train theta is non zero") {
    val data = Seq(
      (1.0, Array(1.0, 5.0)),
      (6.0, Array(1.0, 2.0))
    )
    val theta =  Array(0.5, 0.5)
    val maxIter = 10
    val learningRate = 0.1

    val expectedTheta = Array(1.7098632193403809, 0.19229353765009757)
    val expectedCost = 4.5116663759254

    val rtheta = LinearRegressionTrainer(theta, maxIter, learningRate, LinearRegression.hypothesys).train(data)

    rtheta.theta should be (expectedTheta)

    cost(data, rtheta.theta) should be (expectedCost)
  }


}
