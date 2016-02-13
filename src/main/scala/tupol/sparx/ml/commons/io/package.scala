package tupol.sparx.ml.commons

import java.io.PrintWriter

import org.apache.hadoop.conf.{Configuration => HadoopConfiguration}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.clustering.KMeansModel

import scala.reflect.ClassTag
import scala.util.Try


/**
  * A few common IO functions that might be useful.
  *
  */
package object io {

    trait SaveableAsObject {
      def saveAsObject(sc: SparkContext, path: String, overwrite: Boolean = true): Boolean
    }

    /**
      * Decorate the PipelineModel with a save method
      *
      * @param pipelineModel
      */
    implicit class BinarySerialisedPipelineModel(val pipelineModel: PipelineModel) extends SaveableAsObject {
      def saveAsObject(sc: SparkContext, path: String, overwrite: Boolean = true): Boolean =
        saveObjectToHdfsFile(sc, pipelineModel, path, ".plm", overwrite)
    }

    /**
      * Decorate the PipelineModel with a save method
      *
      * @param model
      */
    implicit class BinarySerialisedKMeansModel(val model: KMeansModel) extends SaveableAsObject {
      def saveAsObject(sc: SparkContext, path: String, overwrite: Boolean = true): Boolean =
        saveObjectToHdfsFile(sc, model, path, ".kmm", overwrite)
    }

    /**
      * Save the given object (that) to the given file path with hte given extension
      *
      * @param that what are we saving?
      * @param path where are we saving?
      * @param extension what is the extension, if any?
      * @param overwrite should we overwrite the previous file if exists?
      * @tparam T type of 'that'
      * @return true if the save succeeded or false otherwise
      */
    def saveObjectToHdfsFile[T: ClassTag](sc: SparkContext, that: T, path: String, extension: String = "", overwrite: Boolean = true): Boolean = {
      sc.parallelize(Seq(that)).saveAsObjectFile(addExtension(path, extension))
      true
    }

    /**
      * Load an object of type T from the given path
      *
      * @param path where are we loading from?
      * @param extension what is the extension, if any?
      * @param mf
      * @tparam T
      * @return
      */
    def loadObjectFromHdfsFile[T](sc: SparkContext, path: String, extension: String = "")(implicit mf: Manifest[T]): Try[T] = {
      Try(sc.objectFile[T](addExtension(path, extension)).first())
    }

    def removeHdfsFile(path: String) = {
      val hdfs = FileSystem.get(new HadoopConfiguration())
      val workingPath = new Path(path)
      hdfs.delete(workingPath, true) // delete recursively
    }

    /**
      * Save a sequence of Strings as lines in an HDFS file
      *
      * @param rez
      * @param path
      * @return
      */
    def saveLinesToFile(rez: Iterable[String], path: String, extension: String = "", overwrite: Boolean = true) = {
      import java.io.BufferedOutputStream
      Try {
        // Create the HDFS file system handle
        val hdfs = FileSystem.get(new HadoopConfiguration())
        // Create a writer
        val writer = new PrintWriter(new BufferedOutputStream(hdfs.create(new Path(addExtension(path, extension)), overwrite)), true)
        //write each line
        rez.foreach(line => Try(writer.print(line + "\n")))
        // Close the streams
        Try(writer.close())
      }
    }

    /**
      * Set the extension to the given path preventing the double extension
      *
      * @param path
      * @param extension
      * @return
      */
    private[this] def addExtension(path: String, extension: String) = {
      val normExt =  extension.trim
      val ext = if (extension.startsWith(".") || normExt.isEmpty) extension else "." + extension
      if (path.endsWith(ext)) path
      else path + ext
    }
}
