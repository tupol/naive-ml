package tupol.ml.regression

import org.scalatest.{FunSuite, Matchers}
import tupol.ml.DoubleLabeledPoint

/**
  *
  */
class LogisticRegressionItTest extends FunSuite with Matchers {

  test("Data Set 1") {
    val input = scala.io.Source.fromInputStream(this.getClass.getResourceAsStream("/logreg_data1.csv")).
      getLines.map(_.split(",").map(_.toDouble))

    val X = input.map(arr => 1.0 +: arr.take(2))
    val Y = input.map(arr => arr.takeRight(1).head)

    val trainingData = X.zip(Y).map{ case (point, label) => DoubleLabeledPoint(label, point)}.toStream

    val initialTheta = Array(0.0, 0.0, 0.0)

    val logisticRegressor = LogisticRegressionTrainer(initialTheta, 100, 1, 0.0, 1E-4).
      train(trainingData)
    logisticRegressor.theta

    // Not sure yet what to expect... sometimes in the future... maybe
  }

  test("Data Set 2") {
    val input = scala.io.Source.fromInputStream(this.getClass.getResourceAsStream("/logreg_data2.csv")).
      getLines.map(_.split(",").map(_.toDouble))

    val X = input.map(arr => arr.take(2)).toSeq
    val Y = input.map(arr => arr.takeRight(1).head).toSeq

    def makePoly(x: Double, y: Double, degree: Int = 6): Array[Double] = {
      import math._
      val rez = for{
        i <- (1 to degree)
        j <- (0 to i)
      } yield pow(x, i-j) * pow(y, j)
      rez.toArray
    }

    val nX = X.map(x => 1.0 +: makePoly(x(0), x(1))).toList


    val trainingData = nX.zip(Y).map{ case (point, label) => DoubleLabeledPoint(label, point)}.toStream

    val initialTheta = Array.fill(nX.head.size)(0.0)

    LogisticRegressionTrainer(initialTheta, 700, 1.0, 1.0, 1E-3).
      train(trainingData)
    // Not sure yet what to expect... sometimes in the future... maybe
  }

}
