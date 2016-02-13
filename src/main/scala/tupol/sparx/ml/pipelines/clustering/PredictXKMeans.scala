package tupol.sparx.ml.pipelines.clustering

import tupol.sparx.ml.commons.ml.PredictXKMeans

object KddPredictXKMeans extends PredictXKMeans(KDD)

object NabPredictXKMeans extends PredictXKMeans(NAB)

object PredictXKMeans {
  def apply(app: String): PredictXKMeans = app.trim.toLowerCase match {
    case NAB => new PredictXKMeans(app)
    case KDD => new PredictXKMeans(app)
    case _ => throw new IllegalArgumentException("Unknown application")
  }
}


