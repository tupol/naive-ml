
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.{XKMeans, XKMeans2}
import org.apache.spark.mllib.linalg.Vector
import tupol.sparx.ml.commons.SparkRunnable
import utils.PointsGenerator

/**
  *
  */
object RandomTests extends SparkRunnable {

  import PointsGenerator._


  /**
    * This is the key for basically choosing a certain app and it should have
    * the form of 'app.....', reflected also in the configuration structure.
    *
    * @return
    */
  override def appName: String = ""

  /**
    * The list of required application parameters. If defined they are checked in the `validate()` function
    *
    * @return
    */
  override def requiredParameters: Seq[String] = Seq()

  override def run(implicit sc: SparkContext, conf: Config): Unit = {

    //  arc(minRadius=0.8, minTheta = 0, maxTheta = Pi/2).
    //      ring(minRadius=0.8).
    //  (ring(points = 100, minRadius=0.2, maxRadius = 0.4) ++ disc(points = 100, radius=0.4, center = (1,1))).
    val rawData = (disc(points = 1000, radius = 1.0)
//      ++ring(points = 1000, minRadius=0.2, maxRadius = 0.4)
//      ++ square(points = 1000, origin = (1,1), side=1)
      ).
      map(x => (x._1, x._2))
//      .map(x => Array(x._1, x._2))


    import org.apache.spark.sql._
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val df = sc.parallelize(rawData).toDF


    import org.apache.spark.ml.feature._
    val toFeaturesVector = new VectorAssembler().setInputCols(Array("_1", "_2")).setOutputCol("_3")
    val dfp = toFeaturesVector.transform(df)


    import org.apache.spark.ml.clustering.XKMeansModel
    import org.apache.spark.ml.clustering.evaluation.WssseEvaluator
    import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

    val kmeans = new XKMeans().setFeaturesCol("_3").setMaxIter(100).setK(2)
    val paramGrid = new ParamGridBuilder()
//      .addGrid(kmeans.k, (2 to 60 by 2))
      .build()
    val cv = new CrossValidator().setEstimator(kmeans).setEvaluator(new WssseEvaluator).setNumFolds(20).setEstimatorParamMaps(paramGrid)

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(dfp)

    // This is the best model that we were able to come up with
    val bestModel = cvModel.bestModel.asInstanceOf[XKMeansModel]

    println("Metrics")
    cvModel.avgMetrics.foreach(println)
    println("----------------")

    val trainPred = bestModel.transform(dfp)

    trainPred.show()

    val points0 = trainPred.select("prediction", "_3").map(r => (r.getInt(0), r.getAs[Vector](1)))


    bestModel.clusterCenters.foreach{x => println(x.toArray.mkString(","))}


    val xkmeans2 = new XKMeans2().setFeaturesCol(bestModel.getFeaturesCol).setPredictionCol(bestModel.getPredictionCol).fit(trainPred)

//    val anom = Seq((0.0, 0.0), (0.2, 0.0), (0.3, 0.0), (0.1, 0.0), (0.45, 0.0),
//      (0.9, 0.9), (2.0, 2.0), (2.1, 2.1), (1.3, 1.3), (1.5, 1.0), (1.5, 0.9), (1.5, 0.8))

        val anom = (0.0 to 1.8 by 0.1).map(x => (0.0, x)
        )

    val adf = sc.parallelize(anom).toDF
    val adfp = bestModel.transform(toFeaturesVector.transform(adf))

    val apred = xkmeans2.transform(adfp)
    apred.show
//    apred.collect.foreach(println)

    apred.select("_1", "_2", "probability", "probabilityByFeature")
    .map(r => (r.getDouble(0),r.getDouble(1),r.getDouble(2), r.getAs[Vector](3)))
    .map(x => (x._1, x._2, x._3, x._4))
    .collect.foreach{ case(x1, x2, x3, x4) =>
      println(f"$x1%+5.3f  $x2%+5.3f  $x3%+7.5f  $x4")
    }



  }
}

