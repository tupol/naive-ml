package tupol.ml.clustering

import tupol.ml._

import scala.collection.parallel.ParSeq

case class KMeansCxPrediction(k: Int, distance: Double, probability: Double, probabilityByDimension: Point)

/**
 *
 */
case class KMeansGaussian(kmeans: KMeans, varianceByCluster: ParSeq[ClusterPoint]) extends Predictor[Point, KMeansCxPrediction] {

  override def predict(data: Point): KMeansCxPrediction = {

    val predictedPoint = kmeans.predict(data)

    def predictByDimension(data: Point): Point = {

      val k = predictedPoint.label._1
      val point = predictedPoint.point
      val featuresNo = point.size

      val variances = varianceByCluster.map(dlp => (dlp.k, dlp.point)).toMap.get(k) match {
        case Some(x) => x
        case None => throw new Exception(s"Could not find variance for k = $k")
      }
      val averages = kmeans.clusterCenters.map(dlp => (dlp.k, dlp.point)).toMap.get(k) match {
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
    val probability = predictByDimension(data).reduce(_ * _)
    KMeansCxPrediction(predictedPoint.label._1, predictedPoint.label._2, probability, predictByDimension(data))

  }

}

/**
 * Essentially calculate the variance for each cluster; the means for each cluster we already know to be the centroid.
 *
 * @param kmeans
 */
case class KMeansGaussianTrainer(kmeans: KMeans) extends Trainer[Point, KMeansGaussian] {

  override def train(data: ParSeq[Point]): KMeansGaussian = {
    val clusteredData = kmeans.predict(data)
    val varianceByK = clusteredData.map(dlp => (dlp.label._1, dlp.point)).groupBy(_._1).map {
      case (k, kfx) =>
        val centroid = kmeans.clusterCenters.map(cc => (cc.k, cc.point)).toMap.get(k) match {
          case Some(x) => x
          case None => throw new Exception(s"Could not find centroid for k = $k")
        }
        ClusterPoint(k, kfx.map(_._2).variance(centroid))
    }.toSeq
    KMeansGaussian(kmeans, varianceByK)
  }

}
