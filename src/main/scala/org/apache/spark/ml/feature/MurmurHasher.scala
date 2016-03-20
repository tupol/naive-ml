package org.apache.spark.ml.feature

import org.apache.commons.lang.NotImplementedException
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._

import scala.util.hashing.MurmurHash3

/**
 * This hasher, though far from being a correct solution, overcomes the problem of StringIndexer,
 * that can not manage unknown labels.
 *
 * This hasher improves on the old StringHasher by treating numbers differently to minimize the
 * collision rate for numeric types.
 *
 * @param uid
 * @param seed
 */
@Experimental
class MurmurHasher(override val uid: String, val seed: Int)
    extends UnaryTransformer[Any, Int, MurmurHasher] {

  def this(seed: Int) = this(Identifiable.randomUID("symHash"), seed)

  def this() = this(1259)

  override protected def createTransformFunc: (Any) => Int = { in => MurmurHasher.hash(in, seed) }

  override protected def outputDataType: DataType = IntegerType
}

object MurmurHasher {

  import org.apache.spark.ml.Byteable._

  /**
   * Hash any given input into a signed integer
   *
   * @param in
   * @param seed
   * @return
   */
  def hash(in: Any, seed: Int): Int = in match {
    case x: Short => MurmurHash3.bytesHash(x.toByteArray, seed)
    case x: Int => MurmurHash3.bytesHash(x.toByteArray, seed)
    case x: Long => MurmurHash3.bytesHash(x.toByteArray, seed)
    case x: Float => MurmurHash3.bytesHash(x.toByteArray, seed)
    case x: Double => MurmurHash3.bytesHash(x.toByteArray, seed)
    case x: Boolean => MurmurHash3.stringHash(x.toString, seed)
    case x: String => MurmurHash3.stringHash(x, seed)
    case any => throw new NotImplementedException(s"No hashing is implemented for objects of type ${any.getClass.getName}.")
    //TODO It might be necessary to add different other implementations in the future (e.g. collections)
  }

}
