package tupol.sparx.ml.pipelines.clustering.preprocessors

import tupol.sparx.ml.commons.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, StructType}
import tupol.sparx.ml.commons.ml.PreProcessorBuilder
import tupol.sparx.ml.commons.ml.PreProcessorBuilder

/**
  * Pre-processing pipeline builder for the KDD data
  */
object KddPreProcessorBuilder extends PreProcessorBuilder {

  def createPreProcessor(inputDataFrame: DataFrame,
                         rawDataColName: String, inputJsonSchema: String,
                         outputColName: String): PipelineModel = {

    // Get the schema that we use for assembling the typed columns out of the parsed data
    val schema: StructType = DataType.fromJson(inputJsonSchema).asInstanceOf[StructType]

    val continuousCols = schema.unlabeledContinuousCols
    val discreteCols = schema.unlabeledDiscreteCols

    // Transform the raw input string into an array of string, give a tokenizer
    val tokenizer = new RegexTokenizer().
      setPattern(",").
      setInputCol(rawDataColName)

    // Split the array of string into multiple columns as described by the schema
    val dataImporter = new DataImporter().
      setInputCol(tokenizer.getOutputCol).
      setNewSchema(schema)

    // Transform the discrete columns into columns of numbers by applying a hashing algorithm.
    // I was asked why, fair enough. The main reason is the limitation of the StringIndexer, which I already tried,
    // but StringIndexer manages unknown labels by throwing an exception, which is undesirable.
    val stringHasher = discreteCols.map { colName =>
      new StringHasher().
        setInputCol(colName)
    }

    // Out of the columns produced by the StringHasher and the columns already containing numbers
    // and produce a column of Vectors
    val toFeaturesVector = new VectorAssembler().
      setInputCols((stringHasher.map(_.getOutputCol) ++ continuousCols).toArray)

    // Scale the vectors so they will fit between -1 and 1
    val scaleAndCenter = new MinMaxScaler().
      setMin(-1).setMax(1).
      setInputCol(toFeaturesVector.getOutputCol).
      setOutputCol(outputColName)

    // Build the pre-processing pipeline
    new Pipeline().
      setStages(
        tokenizer +:
          dataImporter +:
          stringHasher.toArray :+
          toFeaturesVector :+
          scaleAndCenter
      ).fit(inputDataFrame)
  }
}
