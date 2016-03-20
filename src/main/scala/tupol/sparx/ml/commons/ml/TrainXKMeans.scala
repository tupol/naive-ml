package tupol.sparx.ml.commons.ml

import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineAssembler
import org.apache.spark.ml.clustering.evaluation.WssseEvaluator
import org.apache.spark.ml.clustering.{XKMeans2, XKMeans, XKMeansModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SQLContext
import tupol.sparx.ml.commons.SparkRunnable
import tupol.sparx.ml.commons.io._
import tupol.sparx.ml.pipelines.clustering.Configuration


/**
  * KMeans trainer given the app name (as reflected in the configuration)
  * and the corresponding pre-processor builder.
  */
class TrainXKMeans(val appName: String, preProcessorBuilder: PreProcessorBuilder) extends SparkRunnable {

  val requiredParameters = Seq(
    "prefix",
    "path.wip",
    "path.output",
    "training.file"
  )

  def run(implicit sc: SparkContext, config: Config) = {

    val conf = new Configuration(config)

    val sqc = new SQLContext(sc)

    import sqc.implicits._

    logDebug(s"$appName: Reading raw input data from ${conf.inputTrainingData}")
    val dataRDD = sc.textFile(conf.inputTrainingData)

    val splitRatio = conf.splitRatio
    val splitSeed = conf.splitSeed

    // Convert the input RDD into a dataframe and name the input column
    val data = dataRDD.toDF(conf.rawDataColName)

    // Split the data into training and test sets (last part is held out for testing)
    val Array(trainData, testData) = data.randomSplit(Array(splitRatio, 1 - splitRatio), splitSeed)

    logDebug(s"$appName: Creating a pre-processor that transforms ${conf.rawDataColName} into a ${conf.preparedDataColName} features column Vector typed")
    // Build the pre-processing pipeline
    val preparationModel = preProcessorBuilder.createPreProcessor(trainData,
      conf.rawDataColName, conf.inputSchema, conf.preparedDataColName)

    // Create a KMeans template that can be used in cross validation
    val kmeans = new XKMeans().
      setFeaturesCol(conf.preparedDataColName)

    // Create the cross validation parameters
    val paramGrid = new ParamGridBuilder()
      .addGrid(kmeans.k, conf.clusters)
      .addGrid(kmeans.tol, conf.tolerances)
      .addGrid(kmeans.maxIter, conf.iterations)
      .build()

    // Build a cross validator that will give us the best model
    val cv = new CrossValidator()
      .setEstimator(kmeans)
      .setEvaluator(new WssseEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(conf.folds) // Use 3+ in practice

    val preparedData = preparationModel.transform(trainData)

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(preparedData)

    // This is the best model that we were able to come up with
    val bestModel = cvModel.bestModel.asInstanceOf[XKMeansModel]

    val predictedData = bestModel.transform(preparedData)

    val kmeans2 = new XKMeans2()
      .setFeaturesCol(bestModel.getFeaturesCol)
      .setPredictionCol(bestModel.getPredictionCol)
      .fit(predictedData)

    kmeans2.transform(predictedData).show


    // TODO we need to add the build info in the pipeline as well
    // TODO Wait a little for Spark 1.6.0 that brings in support for some meta-data in the pipeline
    val bestPipeline = PipelineAssembler(preparationModel, bestModel, kmeans2)

    val outFile = f"${conf.outputPath}/${conf.prefix}_${bestPipeline.uid}"
    logInfo(s"$appName: Saving pipeline to $outFile.pml")
    bestPipeline.saveAsObject(sc, outFile)

    // TODO save a companion text file with the entire pipeline description next to the pipeline object as it should be easier to see what we got.
    // Right now we are saving just a small bit of the big picture
    logInfo(s"$appName: Saving pipeline info to $outFile.txt")
    saveLinesToFile(kmeans2.distanceStatsReportMarkdown, outFile, "txt")

    // TODO rethink the naming so we don't need to add the extension here; maybe the save should return a Try[Path] instead of a boolean.
    // Since we can return any, we can return the best pipeline file path
    s"$outFile.plm"

  }

}
