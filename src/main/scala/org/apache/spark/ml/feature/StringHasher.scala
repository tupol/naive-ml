package org.apache.spark.ml.feature

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.util.hashing.MurmurHash3


/**
  * This hasher, though far from being a correct solution, overcomes the problem of StringIndexer,
  * that can not manage unknown labels.
  */
@Experimental
class StringHasher(override val uid: String, val seed: Int)
  extends Transformer with HasInputCol with HasOutputCol {

  def this(seed: Int) = this(Identifiable.randomUID("symHash"), seed)

  def this() = this(1259)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {

    val hashUDF = udf { term: String => MurmurHash3.stringHash(term, seed) }
    dataset.withColumn($(outputCol), hashUDF(col($(inputCol))))
  }


  /** Validates and transforms the input schema. */
  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == StringType || inputDataType.isInstanceOf[NumericType],
      s"The input column $inputColName must be either string type or numeric type, " +
        s"but got $inputDataType.")
    val inputFields = schema.fields
    val outputColName = $(outputCol)
    require(inputFields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    SchemaUtils.appendColumn(schema, $(outputCol), LongType)
  }

  override def copy(extra: ParamMap): StringHasher = defaultCopy(extra)
}
