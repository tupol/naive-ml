package org.apache.spark.ml.feature

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/**
  * Converts a column of Vectors from SparseVector to Dense Vector or the other way around.
  *
  * This class exists because the VectorAssembler produces only SparseVector columns
  *
  * No new column is created, the column is converted on the spot.
  *
  */
class SparseToDense(override val uid: String)
  extends Transformer {

  def this() = this(Identifiable.randomUID("sparToDens"))

  final val inputCol: Param[String] = new Param[String](this, "inputVectorCol", "input column name")

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  val toDense: BooleanParam = new BooleanParam(this, "toDense",
    "Produce a DenseVector")

  /** @group setParam */
  def setToDense: this.type = set(toDense, true)

  /** @group setParam */
  def setToSparse: this.type = set(toDense, false)

  setDefault(toDense -> true)

  override def transform(dataset: DataFrame): DataFrame = {
    val scale = udf {
      transformVector _
    }
    dataset.withColumn($(inputCol), scale(col($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  private def transformVector(vector: linalg.Vector): linalg.Vector = {
    (vector, $(toDense)) match {
      case (dv: DenseVector, false) => dv.toSparse
      case (sv: SparseVector, true) => sv.toDense
      case _ => vector
    }
  }
}
