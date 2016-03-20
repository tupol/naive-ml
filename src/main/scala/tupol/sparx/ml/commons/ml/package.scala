package tupol.sparx.ml.commons

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.clustering.{XKMeans2Model, XKMeansModel}
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

  implicit class TrainXKMeansMarkdown(val model: XKMeans2Model) {

    def distanceStatsReportMarkdown: Seq[String] = {

      val modelSummary = "" :: "" ::
        "### Model Summary" :: "" ::
        f"| Results Info     | Value         |" ::
        f"| :--------------- | ------------: |" ::
        f"| WSSSE            | ${model.distanceMetrics.byModel.sse}%.7E |" ::
        f"| Min. Distance    | ${model.distanceMetrics.byModel.min}%13.11f |" ::
        f"| Avg. Distance    | ${model.distanceMetrics.byModel.avg}%13.11f |" ::
        f"| Max. Distance    | ${model.distanceMetrics.byModel.max}%13.11f |" ::
        f"| Variance         | ${model.distanceMetrics.byModel.variance}%13.11f |" ::
        Nil

      val tabClustInfoHeader = "" :: "" ::
        "### Clusters Info" ::
        "" ::
        f"| K     | ${"Count"}%-10s | ${"Min. Dist."}%-12s | ${"Avg. Dist."}%-12s | ${"Max. Dist."}%-12s | ${"SSE"}%-12s | ${"Variance"}%-12s | " ::
        f"| ----: | ---------: | -----------: | -----------: | -----------: | -----------: | -----------: |  " ::
        Nil

      val tabClustInfoHeader2 = "" :: "" ::
        "### Clusters Info By Feture" ::
        "" ::
        f"| K     | ${"Count"}%-10s | ${"Min. Value"}%-12s | ${"Avg. Value"}%-12s | ${"Max. Value"}%-12s | ${"Variance"} | " ::
        f"| ----: | ---------: | -----------: | -----------: | -----------: | -----------: | " ::
        Nil


      val tabClustInfo = (0 until model.distanceMetrics.byCluster.size).
        zip(model.distanceMetrics.byCluster).
        map { case (k, cds) =>
          if (cds.count > 0) {
            f"| $k%5d | ${cds.count}%10d | ${cds.min}%+1.5E | ${cds.avg}%+1.5E | ${cds.max}%+1.5E | ${cds.sse}%+1.5E | ${cds.variance}%+1.5E | "
          } else {
            f"| $k%5d | ${" "}%10s | ${" "}%10s | ${" "}%10s | ${" "}%12s | ${" "}%12s | ${" "}%12s | ${" "}%12s | ${" "}%12s | "
          }
        }


      val tabClustInfo2 = (0 until model.statisticalSummaryByCluster.size).
        zip(model.statisticalSummaryByCluster).
        map { case (k, cds) =>
          if (cds.count > 0) {
            val min = cds.min.toArray.map(x => f"$x%+1.6E").mkString("[", ",", "]")
            val avg = cds.mean.toArray.map(x => f"$x%+1.6E").mkString("[", ",", "]")
            val max = cds.max.toArray.map(x => f"$x%+1.6E").mkString("[", ",", "]")
            val variance = cds.variance.toArray.map(x => f"$x%+1.6E").mkString("[", ",", "]")
            f"| $k%5d | ${cds.count}%10d | $min | $avg | $max | $variance | "
          } else {
            f"| $k%5d | ${" "}%10s | ${" "}%10s | ${" "}%10s | ${" "}%12s | ${" "}%12s |  "
          }
        }

      modelSummary ++ tabClustInfoHeader ++ tabClustInfo ++ tabClustInfoHeader2 ++ tabClustInfo2

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
          case x : XKMeansModel => x.getPredictionCol :: x.getDistanceToCentroidCol :: Nil
          case x : XKMeans2Model => x.getProbabilityCol :: x.getProbabilityByFeatureCol :: Nil
          // TODO Add more prediction models here to make this more generic
          case _                => Nil
        }
  }

}
