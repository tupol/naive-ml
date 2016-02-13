package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{Param, Params}

/**
  * Trait for shared param distanceToCentroidCol (default: "distanceToCentroid").
  */
private[ml] trait HasDistanceToCentroidCol extends Params  {
  /**
    * Param for distanceToCentroid column name.
    *
    * @group param
    */
  final val distanceToCentroidCol: Param[String] = new Param[String](this, "distanceToCentroid", "distance to closest centroid column name")

  setDefault(distanceToCentroidCol, "distanceToCentroid")

  /** @group getParam */
  final def getDistanceToCentroidCol: String = $(distanceToCentroidCol)

}
