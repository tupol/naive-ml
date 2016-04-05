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

    assert(actualEpsilon < tolerance)

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

    assert(actualEpsilon < tolerance)

  }

  test("LinearRegression#train test 1") {

    //gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);

    val data = Seq(
      (1.0, Array(1.0, 5.0)),
      (6.0, Array(1.0, 2.0)),
      (4.0, Array(1.0, 4.0)),
      (2.0, Array(1.0, 5.0))
    )
    val theta =  Array(0.0, 0.0)
    val tolerance = 0.001
    val expected = 11.9450
    val maxIter = 1000
    val epsilon = 0.01

//    train(data, theta, maxIter, epsilon)

  }


}
