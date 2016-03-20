package org.apache.spark.ml

import java.nio.ByteBuffer

/**
 * Trait that helps classes to be serialised to a byte array.
 */
trait Byteable extends Any {
  def toByteArray: Array[Byte]
}

object Byteable {

  implicit class ByteableShort(val x: Short) extends AnyVal with Byteable {
    def toByteArray: Array[Byte] = {
      val buf = ByteBuffer.allocate(2)
      buf.putShort(x)
      buf.array
    }
  }

  implicit class ByteableInt(val x: Int) extends AnyVal with Byteable {
    def toByteArray: Array[Byte] = {
      val buf = ByteBuffer.allocate(4)
      buf.putInt(x)
      buf.array
    }
  }

  implicit class ByteableLong(val x: Long) extends AnyVal with Byteable {
    def toByteArray: Array[Byte] = {
      val buf = ByteBuffer.allocate(8)
      buf.putLong(x)
      buf.array
    }
  }

  implicit class ByteableFloat(val x: Float) extends AnyVal with Byteable {
    def toByteArray: Array[Byte] = {
      val buf = ByteBuffer.allocate(4)
      buf.putFloat(x)
      buf.array
    }
  }

  implicit class ByteableDouble(val x: Double) extends AnyVal with Byteable {
    def toByteArray: Array[Byte] = {
      val buf = ByteBuffer.allocate(8)
      buf.putDouble(x)
      buf.array
    }
  }

}
