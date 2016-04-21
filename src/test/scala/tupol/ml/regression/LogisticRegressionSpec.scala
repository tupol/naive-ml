package tupol.ml.regression

import org.scalatest.{ FunSuite, Matchers }
import tupol.ml.DoubleLabeledPoint

import scala.collection.parallel.ParSeq

/**
 *
 */
class LogisticRegressionSpec extends FunSuite with Matchers {

  import LogisticRegression._

  test("LogisticRegression#sigmoid ") {

    sigmoid(-10000.0) should be(0)
    sigmoid(0.0) should be(0.5)
    sigmoid(4.0) should be(0.9820137900379085)
    sigmoid(5.0) should be(0.9933071490757153)
    sigmoid(6.0) should be(0.9975273768433653)
    sigmoid(61.0) should be(1.0)
    sigmoid(75.0) should be(1.0)
    sigmoid(10000.0) should be(1.0)
    sigmoid(10000.0) should be(1.0)
  }

  test("LogisticRegression#cost un-regularized 1") {

    val data = ParSeq(
      DoubleLabeledPoint(1.0, Array(1.0, 8.0, 1.0, 6.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 3.0, 5.0, 7.0)),
      DoubleLabeledPoint(1.0, Array(1.0, 4.0, 9.0, 2.0))
    )

    val theta = Array(-2.0, -1.0, 1.0, 2)

    LogisticRegression.cost(data, theta, 0.0) should be(4.683166549810689)

  }

  test("LogisticRegression#cost un-regularized 2") {

    val data = ParSeq(
      DoubleLabeledPoint(1.0, Array(1.0, 16.0, 2.0, 3.0, 13.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 5.0, 11.0, 10.0, 8.0)),
      DoubleLabeledPoint(1.0, Array(1.0, 9.0, 7.0, 6.0, 12.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 4.0, 14.0, 15.0, 1.0))
    )

    val theta = Array(-1.0, 2.0, -3.0, 4.0, -5.0)

    LogisticRegression.cost(data, theta, 0.0) should be(21.999999991552066)

  }

  test("LogisticRegression#cost regularized") {

    val data = ParSeq(
      DoubleLabeledPoint(1.0, Array(1.0, 8.0, 1.0, 6.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 3.0, 5.0, 7.0)),
      DoubleLabeledPoint(1.0, Array(1.0, 4.0, 9.0, 2.0))
    )

    val theta = Array(-2.0, -1.0, 1.0, 2)

    LogisticRegression.cost(data, theta, 3.0) should be(7.683166549810689)

  }

  test("LogisticRegression#gradient un-regularized 1") {

    val data = ParSeq(
      DoubleLabeledPoint(1.0, Array(1.0, 8.0, 1.0, 6.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 3.0, 5.0, 7.0)),
      DoubleLabeledPoint(1.0, Array(1.0, 4.0, 9.0, 2.0))
    )

    val theta = Array(-2.0, -1.0, 1.0, 2)

    val expectedGradient = Array(0.317220748033335, 0.8723154384059271, 1.6481235028108963, 2.2378722792832018)

    LogisticRegression.gradient(data, theta, 0.0) should be(expectedGradient)

  }

  test("LogisticRegression#gradient un-regularized 2") {

    val data = ParSeq(
      DoubleLabeledPoint(1.0, Array(1.0, 16.0, 2.0, 3.0, 13.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 5.0, 11.0, 10.0, 8.0)),
      DoubleLabeledPoint(1.0, Array(1.0, 9.0, 7.0, 6.0, 12.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 4.0, 14.0, 15.0, 1.0))
    )

    val theta = Array(-1.0, 2.0, -3.0, 4.0, -5.0)

    val expectedGradient = Array(-0.2500000005056777, -5.250000002011198, 1.2499999928901242, 1.4999999923655705, -6.000000000437538)

    LogisticRegression.gradient(data, theta, 0.0) should be(expectedGradient)

  }

  test("LogisticRegression#gradient regularized") {

    val data = ParSeq(
      DoubleLabeledPoint(1.0, Array(1.0, 8.0, 1.0, 6.0)),
      DoubleLabeledPoint(0.0, Array(1.0, 3.0, 5.0, 7.0)),
      DoubleLabeledPoint(1.0, Array(1.0, 4.0, 9.0, 2.0))
    )

    val theta = Array(-2.0, -1.0, 1.0, 2)

    val expectedGradient = Array(0.317220748033335, -0.12768456159407293, 2.648123502810896, 4.237872279283202)

    LogisticRegression.gradient(data, theta, 3.0) should be(expectedGradient)

  }

  test("LogisticRegression#predict 1") {

    val data = ParSeq(
      Array(8.0, 1.0, 6.0),
      Array(3.0, 5.0, 7.0),
      Array(4.0, 9.0, 2.0)
    )

    val theta = Array(0.0, 1.0, 10.0)

    val expectedPredictions = Array(1.0, 1.0, 1.0)
    val actualPredictions = LogisticRegression(ParSeq(theta)).predict(data).map(_.label).toArray

    actualPredictions should be(expectedPredictions)

  }

  test("LogisticRegression#predict 2") {

    val data = ParSeq(
      Array(8.0, 1.0, 6.0),
      Array(3.0, 5.0, 7.0),
      Array(4.0, 9.0, 2.0)
    )

    val theta = Array(4.0, 3.0, -8.0)

    val expectedPredictions = Array(0.0, 0.0, 1.0)
    val actualPredictions = LogisticRegression(ParSeq(theta)).predict(data).map(_.label).toArray

    actualPredictions should be(expectedPredictions)

  }

  test("LogisticRegression#predict 3") {

    val data = ParSeq(
      Array(8.0, 1.0, 6.0),
      Array(3.0, 5.0, 7.0),
      Array(4.0, 9.0, 2.0)
    )

    val theta = Array(3.0, 0.0, -8.0)

    val expectedPredictions = Array(0.0, 0.0, 0.0)
    val actualPredictions = LogisticRegression(ParSeq(theta)).predict(data).map(_.label).toArray

    actualPredictions should be(expectedPredictions)

  }

}
