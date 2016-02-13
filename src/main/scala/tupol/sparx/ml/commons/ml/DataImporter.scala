package tupol.sparx.ml.commons.ml

import java.sql.Date

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Transform a column containing parsed input in the form of arrays of string into multiple columns,
  * matching the field description provided by the inputSchema parameter.
  *
  * The schema itself is normally coming from a configuration file that the user produced in advance and persisted.
  * Data is also converted to appropriate types.
  */
class DataImporter(override val uid: String)
  extends Transformer {

  def this() = this(Identifiable.randomUID("import"))

  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")

  final val inputSchema: Param[StructType] = new Param[StructType](this, "inputSchema", "schema description object of the input data ")

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setNewSchema(parser: StructType): this.type = set(inputSchema, parser)

  def getNewSchema: StructType = $(inputSchema)

  override def transform(dataset: DataFrame): DataFrame = {

    // TODO deal with type conversion exceptions
    def arrayToRow(input: Seq[String]) = {
      input.zip($(inputSchema).fields).
        map { case (in, field) => typeConversion(in, field) }
    }

    val data = dataset.rdd.map{row =>
      // The following padding part is a bit of a dirty hack
      // TODO Think of a better way to address labels and no labels as well as just missing columns
      val input = row.getAs[Seq[String]]($(inputCol))
      val sizeDiff = $(inputSchema).fields.size - input.size
      val padding = if(sizeDiff > 0) Array.fill[String](sizeDiff)(null) else Array[String]()

      Row.fromSeq( row.toSeq ++ arrayToRow(input ++ padding))
    }

    val schema = transformSchema(dataset.schema)

    dataset.sqlContext.createDataFrame(data, schema)
  }

  override def transformSchema(schema: StructType): StructType =
    $(inputSchema).fields.foldLeft(schema){ (xs, field) =>
    require(!xs.fieldNames.contains(field.name), s"Column ${field.name} already exists.")
    StructType(xs.fields :+ field) }

  /**
    * Basic type conversions
    *
    * @param input
    * @param targetField
    * @return
    */
  private def typeConversion(input: String, targetField: StructField): Any = {
    import java.sql.Timestamp
    if (input == null && targetField.nullable)
      null
    else
      targetField.dataType match {
        // TODO make this list exhaustive and manage exceptions
        case ShortType => input.toShort
        case IntegerType => input.toInt
        case LongType => input.toLong
        case FloatType => input.toFloat
        case DoubleType => input.toDouble
        case BooleanType => input.toBoolean
        case StringType => input
        case TimestampType => Timestamp.valueOf(input)
        case DateType => Date.valueOf(input)
        case t => throw new IllegalArgumentException(s"Parameter type unknown: ${t.typeName}")
      }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
