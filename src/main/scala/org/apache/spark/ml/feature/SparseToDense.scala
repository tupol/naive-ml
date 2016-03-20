package org.apache.spark.ml.feature

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, VectorUDT}
import org.apache.spark.sql.types.{DataType, StructType}

/**
  * Converts a column of Vectors from SparseVector to Dense Vector or the other way around.
  *
  * This class exists because the VectorAssembler produces only SparseVector columns
  *
  * No new column is created, the column is converted on the spot.
  *
  */
class SparseToDense(override val uid: String)
  extends UnaryTransformer[Vector, Vector, SparseToDense] {

  def this() = this(Identifiable.randomUID("sparToDens"))

  val toDense: BooleanParam = new BooleanParam(this, "toDense",
    "Produce a DenseVector")

  /** @group setParam */
  def setToDense: this.type = set(toDense, true)

  /** @group setParam */
  def setToSparse: this.type = set(toDense, false)

  setDefault(toDense -> true)

  override def transformSchema(schema: StructType): StructType = schema

  override protected def createTransformFunc: (Vector) => Vector = { vector =>
    (vector, $(toDense)) match {
      case (dv: DenseVector, false) => dv.toSparse
      case (sv: SparseVector, true) => sv.toDense
      case _ => vector
    }
  }

  override protected def outputDataType: DataType = new VectorUDT()
}

