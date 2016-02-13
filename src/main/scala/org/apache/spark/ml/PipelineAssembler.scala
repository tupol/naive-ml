package org.apache.spark.ml

import org.apache.spark.ml.util.Identifiable

/**
  * If we know what we are doing we should be able to assemble already fitted pipelines (a.k.a. PipelineModel(s))
  */
object PipelineAssembler {

  def apply(pipelineModels :PipelineModel*): PipelineModel =
    new PipelineModel(Identifiable.randomUID("pipeline"), pipelineModels.foldLeft(Array[Transformer]())((acc, model) => acc ++ model.stages))

  def apply(pipelineModel :PipelineModel, transformers: Transformer*): PipelineModel =
    new PipelineModel(Identifiable.randomUID("pipeline"), pipelineModel.stages ++ transformers)

}
