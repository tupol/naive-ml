package tupol.ml.regression

import org.scalatest.{FunSuite, Matchers}
import tupol.ml._
import tupol.ml.pointops._

/**
  *
  */
class LinearRegressionItTest extends FunSuite with Matchers {


  test("Data Set 1") {
    val input = scala.io.Source.fromInputStream(this.getClass.getResourceAsStream("/linreg_data1.csv")).
      getLines.map(_.split(",").map(_.toDouble)).toSeq

    val X = input.map(arr => arr.take(2)).normalize().map(arr => 1.0 +: arr)
    val Y = input.map(arr => arr.takeRight(1).head)

    val trainingData = (X zip Y).map{ case (point, label) => DoubleLabeledPoint(label, point)}

    val initialTheta = Array(0.0, 0.0, 0.0)
    //This is flimsy... fails occasionally
    val expectedTheta = Array(340412.65957446786, 110631.04594373259, -6649.469935706276)

    import LinearRegression._
    val linearRegressor = LinearRegressionTrainer(initialTheta, 400, 0.1, hypothesys).train(trainingData)

  }

}
