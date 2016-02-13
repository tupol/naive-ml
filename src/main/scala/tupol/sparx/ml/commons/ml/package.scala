package tupol.sparx.ml.commons

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.clustering.XKMeansModel
import org.apache.spark.sql.types.StructType

/**
  *
  */
package object ml {


  implicit class XStructType(val schema: StructType) extends AnyVal {

    // TODO test the behavior when continuous and label metadata are not present
    def unlabeledContinuousCols = schema.fields.
      filter{_.metadata.getBoolean("continuous")}.
      filterNot{_.metadata.getBoolean("label")}.
      map(_.name)

    // TODO test the behavior when continuous and label metadata are not present
    def unlabeledDiscreteCols = schema.fields.
      filterNot{_.metadata.getBoolean("continuous")}.
      filterNot{_.metadata.getBoolean("label")}.
      map(_.name)

    def unlabeledCols = unlabeledContinuousCols ++ unlabeledDiscreteCols

    // TODO test the behavior when label metadata is not present
    def label = schema.fields.filter{_.metadata.getBoolean("label")}.map(_.name)

  }

  implicit class TrainXKMeansMarkdown(val model: XKMeansModel) {

    def distanceStatsReportMarkdown: Seq[String] = {

      val modelSummary = "" :: "" ::
        "### Model Summary" :: "" ::
        f"| Results Info     | Value         |" ::
        f"| :--------------- | ------------: |" ::
        f"| WSSSE            | ${model.metrics.metricsByModel.sse}%.7E |" ::
        f"| Min. Distance    | ${model.metrics.metricsByModel.min}%13.11f |" ::
        f"| Avg. Distance    | ${model.metrics.metricsByModel.avg}%13.11f |" ::
        f"| Max. Distance    | ${model.metrics.metricsByModel.max}%13.11f |" ::
        f"| Variance         | ${model.metrics.metricsByModel.variance}%13.11f |" ::
        Nil

      val tabClustInfoHeader = "" :: "" ::
        "### Clusters Info" ::
        "" ::
        f"| K     | ${"Count"}%-10s | ${"Min. Dist."}%-12s | ${"Avg. Dist."}%-12s | ${"Max. Dist."}%-12s | ${"SSE"}%-12s | ${"Variance"}%-12s | " ::
        f"| ----: | ---------: | -----------: | -----------: | -----------: | -----------: | -----------: | " ::
        Nil


      val tabClustInfo = (0 until model.getK).
        zip(model.metrics.metricsByCluster).
        map { case (k, cds) =>
          if (cds.count > 0) {
            f"| $k%5d | ${cds.count}%10d | ${cds.min}%12.4f | ${cds.avg}%12.4f | ${cds.max}%12.4f | ${cds.sse}%12.4f | ${cds.variance}%12.4f | "
          } else {
            f"| $k%5d | ${" "}%10s | ${" "}%10s | ${" "}%10s | ${" "}%12s | ${" "}%12s | ${" "}%12s | ${" "}%12s | ${" "}%12s | "
          }

        }

      modelSummary ++ tabClustInfoHeader ++ tabClustInfo

    }
  }

  /**
    * Decorate the PipelineModel with useful methods and functions
    *
    * @param pipelineModel
    */
  implicit class XPipelineModel(val pipelineModel: PipelineModel) extends AnyVal {

    // Define the column names that we are going to use for exporting to ES
    // This is mainly because in hte current version of elasticsearch-spark some column types
    // can not be serialized (as as column of Vector).
    def exportableColumns: Seq[String] =
      pipelineModel.stages.
        flatMap{
          case d : DataImporter => d.getNewSchema.fields.map(_.name).toList
          case x : XKMeansModel => x.getPredictionCol :: x.getDistanceToCentroidCol :: x.getProbabilityCol :: Nil
          // TODO Add more prediction models here to make this more generic
          case _                => Nil
        }
  }

}
