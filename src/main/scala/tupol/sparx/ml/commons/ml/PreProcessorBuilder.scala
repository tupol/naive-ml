package tupol.sparx.ml.commons.ml

import org.apache.spark.Logging
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/**
  * Build pre-processing pipeline.
  */
trait PreProcessorBuilder extends Logging {


  /**
    * Create a input data preparation pipeline.
    *
    * The output pipeline model should take input DataFrame, the raw input data column
    * along with the schema that should be produced from it and the final output column
    * of the transformations, which should be a column of Vector.
    *
    * This might be extracted into a trait later on.
    *
    * @param inputDataFrame
    * @param rawDataColName
    * @param inputJsonSchema
    * @param outputColName
    * @return
    */

  def createPreProcessor(inputDataFrame: DataFrame,
                         rawDataColName: String, inputJsonSchema: String,
                         outputColName: String): PipelineModel

}
