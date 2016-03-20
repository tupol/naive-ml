package tupol.sparx.ml.commons.experiments

import breeze.linalg.{DenseMatrix, pinv}

/**
  *
  */
object TestAprox extends App {


  // import breeze.linalg._
  // import breeze.numerics._

  val mat = Array(
    Array(020.0, 1.969651E+14, 2.290424E+05, 4.378416E+07, 3.805730E+07),
    Array(030.0, 5.675540E+13, 1.710782E+05, 1.700446E+07, 2.160023E+07),
    Array(040.0, 2.316503E+13, 1.319024E+05, 1.120996E+07, 1.149426E+07),
    Array(050.0, 9.721460E+12, 1.044779E+05, 9.091417E+06, 8.632630E+06),
    Array(060.0, 5.794071E+12, 9.526762E+04, 7.229374E+06, 7.056353E+06),
    Array(070.0, 4.161690E+12, 8.612571E+04, 5.923035E+06, 5.756613E+06),
    Array(080.0, 2.914639E+12, 7.384531E+04, 5.092197E+06, 4.925615E+06),
    Array(090.0, 1.952307E+12, 6.687644E+04, 4.471485E+06, 4.573427E+06),
    Array(100.0, 1.301934E+12, 6.144810E+04, 4.012595E+06, 3.902935E+06),
    Array(110.0, 1.046782E+12, 5.601230E+04, 3.407562E+06, 3.370927E+06),
    Array(120.0, 8.690001E+11, 5.439095E+04, 3.116815E+06, 3.185722E+06),
    Array(130.0, 7.286499E+11, 5.187239E+04, 2.937207E+06, 2.844567E+06),
    Array(140.0, 5.955503E+11, 4.892317E+04, 2.630652E+06, 2.610710E+06),
    Array(150.0, 4.904656E+11, 4.542728E+04, 2.415579E+06, 2.385341E+06),
    Array(160.0, 4.014726E+11, 4.289776E+04, 2.312963E+06, 2.219629E+06)//,
//    Array(170.0, 3.500741E+11, 4.147563E+04, 2.135472E+06, 2.152693E+06),
//    Array(180.0, 3.168889E+11, 3.914782E+04, 1.997441E+06, 2.037492E+06),
//    Array(190.0, 2.744621E+11, 3.830164E+04, 1.892084E+06, 1.860991E+06),
//    Array(200.0, 2.341774E+11, 3.681869E+04, 1.749163E+06, 1.775982E+06)
  )

//  mat.foreach(r => println(r.mkString(",")))


  val dm = DenseMatrix.create(mat.head.size, mat.size, mat.flatten)

  val dmt = dm.t

  println(dmt)

  val X = dmt(::, 0).toDenseMatrix
  val Y = dmt(::, 2).toDenseMatrix.t
  val X2 = X :* X
  val X3 = X2 :* X
  val X4 = X3 :* X
  val X5 = X4 :* X
  val X6 = X5 :* X
  val X7 = X6 :* X
  val X8 = X7 :* X

  val X1 = DenseMatrix.vertcat( DenseMatrix.ones[Double](X.rows, X.cols).toDenseMatrix, X, X2
    , X3//, X4, X5//, X6, X7 //, X8
    ).t

  println(s"X = \n$X")
  println(s"Y = \n$Y")
  println(s"X1 = \n$X1")
  println(s"X2 = \n${X1.t * X1}")
  println(s"PX2 = \n${pinv(X1.t * X1)}")

  val theta = pinv(X1.t * X1) * X1.t * Y

  println(s"theta = \n$theta")

  val tht = theta.toArray

  def f(x: Double) = tht(0) + x * tht(1) + x * x * tht(2) +
   x * x * x * tht(3) //+ x * x * x * x * tht(4) +
//   x * x * x * x * x * tht(5) //+
//    x * x * x * x * x * x * tht(6) +
//    x * x * x * x * x * x * x * tht(7) //+
//    x * x * x * x * x * x * x * x * tht(8)

//  println(pinv(X.t * X) * X.t * Y)

  (20 to 100 by 10).foreach(x => println(s"$x => ${f(x)}"))

  (0 until X.size).foreach{i =>
    val x = X.toArray(i)
    val y = Y.toArray(i)
    val fx = f(x)
    println(f"$x%4.0f | $fx%12.3f | $y%12.3f ")
  }




}




