package tupol.sparx.ml.pipelines.clustering

import tupol.sparx.ml.commons.ml._
import tupol.sparx.ml.pipelines.clustering.preprocessors.{KddPreProcessorBuilder, NabPreProcessorBuilder}


object KddTrainXKMeans extends TrainXKMeans(KDD, KddPreProcessorBuilder)

object NabTrainXKMeans extends TrainXKMeans(NAB, NabPreProcessorBuilder)


object TrainXKMeans {
  def apply(appName: String): TrainXKMeans = appName.trim.toLowerCase match {
    case NAB => new TrainXKMeans(appName, NabPreProcessorBuilder){}
    case KDD => new TrainXKMeans(appName, KddPreProcessorBuilder)
    case _ => throw new IllegalArgumentException("Unknown application")
  }
}

