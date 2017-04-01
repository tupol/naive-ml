package tupol

import scala.collection.parallel.ParSeq

/**
 *
 */
package object ml {

  type Point = Array[Double]

  abstract class LabeledPoint[L](label: L, point: Point) {
    override def toString = s"($label) ${point.mkString("[", ", ", "]")})"
  }

  case class DoubleLabeledPoint(label: Double, point: Point) extends LabeledPoint[Double](label, point)

  trait Predictor[T, P] {
    def predict(data: T): P

    def predict(data: Seq[T]): Seq[P] =
      predict(data.par).toList

    def predict(data: ParSeq[T]): ParSeq[P] =
      data.map(predict)
  }

  trait Trainer[T, P] {
    def train(data: Seq[T]): P = train(data.par)
    def train(data: ParSeq[T]): P
  }

  /**
   * This is a pseudo polynomial of 1 variable x
   * The functions map keys are function names or text representations; for example for a function f(x) = x * x,
   * the text representation should be x * x
   */
  case class CxFun(functions: Map[String, Double => Double]) extends Function1[Double, Double] {

    def this(functions: Seq[Double => Double]) =
      this(functions.zipWithIndex.map(_.swap).map {
        case (id, f) =>
          (f"$id%02d", f)
      }.toMap)

    /**
     * Create a new CxFun by mutiplying each function with a constant parameter
     *
     * @param parameters
     * @return
     */
    def withParameters(parameters: Seq[Double]): CxFun = {
      require(parameters.size == functions.size)
      val newFunctions = parameters.zip(functions).map { case (p, cxf) => (s"$p * ${cxf._1}", (x: Double) => p * cxf._2(x)) }
      CxFun(newFunctions.toMap)
    }

    private lazy val assembledFunction = functions.map(_._2).reduce((f1, f2) => (x: Double) => f1(x) + f2(x))

    override def apply(x: Double): Double = assembledFunction(x)

    override def toString: String = functions.map(_._1).mkString(" + ")

  }

}
