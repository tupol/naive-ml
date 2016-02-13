package tupol.sparx.ml

/**
  * A few common functions that might be useful.
  *
  * This should not be a repository for all common functions. As more and more similar functions
  * appear here they should be moved to different objects/classes...
  *
  * When unsure, ask a colleague.
  *
  */
package object commons {

  /**
    * Transform a string into a sequence of ints.
    *
    * Case 1: A sequence of ints
    * - first int: 'from'
    * - second int: 'to' (smaller than from)
    * - third int: 'step' a positive number
    *
    * Case 2: A single int
    *
    * @param argument -> start,stop,step or just an int value
    * @return
    */
  def parseStringToRange(argument: String): Seq[Int] = {
    val arg = argument.split(",").map(_.trim.toInt).toSeq
    arg match {
      case start +: stop +: step +: Nil =>
        require(start < stop, "The start should be smaller than the stop.")
        require(step > 0, "The step should be a positive number.")
        ((start to stop) by step)
      case value +: Nil => Seq(value)
      case _ => throw new IllegalArgumentException(
        """
          |The input should contain either an Integer or a comma separated list of 3 integers.
        """.stripMargin)
    }
  }

  /**
    * Run a block and return the block result and the runtime in millis
    *
    * @param block
    * @return
    */
  def timeCode[T](block: => T): (T, Long) = {
    val start = new java.util.Date
    val result = block
    val runtime = (new java.util.Date).toInstant.toEpochMilli - start.toInstant.toEpochMilli
    (result, runtime)
  }

}
