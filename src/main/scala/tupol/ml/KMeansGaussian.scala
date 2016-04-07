package tupol.ml

/**
 *
 */
case class KMeansGaussian(kmeans: KMeans, varianceByCluster: Seq[DoubleLabeledPoint]) extends Predictor[Point, Double] {

  override def predict(data: Point): Double = {
    predictByDimension(data).reduce(_ * _)
  }

  def predictByDimension(data: Point): Point = {

    val predictedPoint = kmeans.predict(data)
    val k = predictedPoint.label
    val point = predictedPoint.point
    val featuresNo = point.size

    val variances = varianceByCluster.map(dlp => (dlp.label, dlp.point)).toMap.get(k) match {
      case Some(x) => x
      case None => throw new Exception(s"Could not find variance for k = $k")
    }
    val averages = kmeans.clusterCenters.map(dlp => (dlp.label, dlp.point)).toMap.get(k) match {
      case Some(x) => x
      case None => throw new Exception(s"Could not find centroid for k = $k")
    }

    (0 until featuresNo).map { f =>
      val sqerr = (point(f) - averages(f)) * (point(f) - averages(f))
      val variance = if (variances(f) == 0) Double.MinValue else variances(f)
      import math._
      val probability = (1 / (sqrt(2 * Pi * variance)) *
        math.exp(-sqerr / (2 * variance)))
      probability
    }.toArray
  }
}

case class KMeansGaussianTrainer(kmeans: KMeans) extends Trainer[Point, KMeansGaussian] {

  override def train(data: Seq[Point]): KMeansGaussian = {
    val clusteredData = kmeans.predict(data)
    val varianceByK = clusteredData.map(dlp => (dlp.label, dlp.point)).groupBy(_._1).map {
      case (k, kfx) =>
        val centroid = kmeans.clusterCenters.map(dlp => (dlp.label, dlp.point)).toMap.get(k) match {
          case Some(x) => x
          case None => throw new Exception(s"Could not find centroid for k = $k")
        }
        DoubleLabeledPoint(k, kfx.map(_._2).variance(centroid))
    }.toSeq
    KMeansGaussian(kmeans, varianceByK)
  }

}
