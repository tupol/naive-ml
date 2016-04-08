package utils

import breeze.plot._
import tupol.ml.Point

/**
  *
  */
object Plotter {


  def myPlot(plottableData: Seq[Point], plotFile: String = "/tmp/my-plot.png"): Unit = {

  val f = Figure()
  //  f.visible = false

  val cols = Array("")

  val dim = plottableData.head.length

  for{
    x <- 0 until dim
    y <- 0 until dim
  } yield {
    val p = f.subplot(dim, dim, x*dim + y)

    val fx = plottableData.map(a => a(x))
    val fy = plottableData.map(a => a(y))
    println(fx.mkString(", "))
    println(fy.mkString(", "))
    p += plot(fx, fy, '+')
    p.xlabel = f"C$x%03d"
    p.ylabel = f"C$y%03d"
  }

  f.saveas(plotFile, 300)

  }
}
