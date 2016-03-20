package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{Param, Params}

/**
  * Trait for shared param distanceToCentroidCol (default: "probabilityByFeature").
  */
private[ml] trait HasProbabilityByFeature extends Params  {
  /**
    * Param for probabilityByFeature column name.
    *
    * @group param
    */
  final val probabilityByFeatureCol: Param[String] = new Param[String](this, "probabilityByFeatureCol", "the probability by feature (dimension)")

  setDefault(probabilityByFeatureCol, "probabilityByFeature")

  /** @group getParam */
  final def getProbabilityByFeatureCol: String = $(probabilityByFeatureCol)

}
