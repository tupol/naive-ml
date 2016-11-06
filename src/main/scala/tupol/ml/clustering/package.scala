package tupol.ml

import java.io.{BufferedOutputStream, FileOutputStream, PrintWriter}

import scala.util.Try

/**
  *
  */
package object clustering {

  def saveLinesToFile(lines: Iterable[String], path: String, overwrite: Boolean = false) = {

    Try {
      // Create a writer
      val writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(path, !overwrite)), true)
      //write each line
      lines.foreach(line => Try(writer.print(line + "\n")))
      // Close the streams
      Try(writer.close())
    }
  }

}
