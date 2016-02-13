# Model Generation

## Introduction

This demo is based on the KDD Cup 1999 exercise, which can be found [here](https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data).

## Configuration

**WARNING!**
The application is mainly a prototype, so there are no extensive input validations done on the configuration, so just make sure that the configuration is done right.
If the application does not work, it is usually a problem with the configuration parameters.

All the configuration parameters need to be prefixed by 'app.APPLICATION_NAME' (e.g. app.kdd or app.nab).

### Required Parameters

| Parameter            | Description                                                                      |
| -------------------- | -------------------------------------------------------------------------------- |
| `prefix`             | The prefix should be set for each app type (e.g 'kdd' for KDD and 'nab' for NAB) |
| `path.wip`           | The path holding the temporary or "work in progress" files |
| `path.output`        | The root output path for the generated model, predictions... |
| `training.file`      | File used for training the model |
| `prediction.file`    | File used as input for the prediction |
| `prediction.pipeline`| Pipeline used for predicting data; this should be the pipeline built by TranXKMeans runnable |

**Note** 
Please keep in mind, when running in a cluster to set the proper URLs for the HDFS files and paths.

**Note** 
If passing these arguments as application arguments, to escape quote the paths and property values containing standard JSON separators (like the `:` in the URL).
For example, an hdfs path parameter should be specified like this: 
`app.kdd.path.wip=\"hdfs:///my/path\"`

### Recommended Adjustable Parameters

| Parameter                    | Default    | Description                                                                      |
| ---------------------------- | ---------: | -------------------------------------------------------------------------------- |
| `training.split.ratio`       | 0.7        | Training data split ratio (a number between 0 and 1, representing the percentage of data that will be used for actual training (not cross validation nor test) |
| `training.split.seed`        | 7774777    | Random seed used for splitting the data |
| `training.kmeans.clusters`   | 140,150,10 | Clusters or cluster range; acceptable values are: <br>- a single int (e.g. 140) <br>- an inclusive range, defined as following: from, to, step (e.g. 140, 150, 5 will produce a sequence containing 140, 145 and 150) |
| `training.kmeans.iterations` | 100        | Iterations or iterations range; acceptable values are: <br>- a single int (e.g. 140) <br>- an inclusive range, defined as following: from, to, step (e.g. 140, 150, 5 will produce a sequence containing 140, 145 and 150) ||
| `training.kmeans.tolerances` | 10         | Tolerance or tolerances range; acceptable values are: <br>- a single int (e.g. 140) <br>- an inclusive range, defined as following: from, to, step (e.g. 140, 150, 5 will produce a sequence containing 140, 145 and 150) |
| `training.kmeans.folds`      | 3          | Folds represents the number of time that the cross validation will run the algorithm with the same parameters, choosing the best out of them; In practice it should be greater than 3.  |
| `prediction.threshold`       | 0.003      | Threshold that delimits normal records from anomalies. Anomalies have a probability smaller than this threshold.  |

### Other Properties

| Parameter                | Description                                                                      |
| ------------------------ | -------------------------------------------------------------------------------- |
| `raw.input.col.name`     | Column name of the dataframe column containing the raw data to be processed.     |
| `raw.timestamp.col.name` | Column name of the dataframe column containing the timestamp of the data input.  |
| `prepared.data.col.name` | Column name of the column of vectors that will be used for model generation / prediction. |

**WARNING!**
The parameters above should not be changed, unless there are external constraints (e.g. a third party application using the persisted column names can not work; for example Kibana does not like columns starting with '_', even though ElasticSearch has not problem with it). 

### `training.input.schema`

This property is the most delicate property in the configuration and for the KDD demo it should not be changed.
It represents the json schema description of the parsed input file (see the org.apache.spark.sql.types.StructType).

An easy way to produce this schema is to run a tool that does csv importing in order to infer the column data types, then map the column names with the actual column names.
In the future we can build a tool that will help the user generate the schema in an easy manner, but for now this will do.


## Usage / Process Description

The application is built using the Spark ML library for machine learning.
 
The objective is to create a pipeline that can be used to produce predictions starting with the raw input data, in the form of a DataFrame object containing the `raw.input.col.name` column.

The raw input string can be parsed using the available `RegexTokenizer` transformer into an array of string, which will have to match the `training.input.schema` description. A quick look at the `KddPreProcessorBuilder` can give some ideas on how the process can work.

Following this point, the rest of the transformers are chained into the pipeline producing a pre-processing or data transformation pipeline.

The data transformation pipeline is used to provide input to the model training code, which runs a cross validation to find the best model, based on a custom `WssseEvaluator`. The input needs to be a column of `Vector`.

The `WssseEvaluator` is a very limited evaluator, as it will always picks the model with the lowest sum of squared errors (WSSSE). This works just fine for multiple folds (choosing the best model running multiple folds on the same set of algorithm meta-parameters), but works poorly while changing the meta-parameters. For example in the case of the K parameters, it will always pick the largest K, because increasing the K decreases the WSSSE.
A good idea would be to implement a gradient threshold based evaluator that will stop the evaluation when the WSSSE gradient gets smaller than a specified parameter.

The result of the cross validation is the "best found model", which will be added to the data transformation pipeline and serialised. This pipeline can be used directly for the predictions, both statically or in a streaming context.


## Run Samples and Options

### Command Line

Sample training command line for running the app in a Yarn cluster with HDFS
```
spark-submit  -v  \
--master yarn-cluster \
--class tupol.sparx.ml.pipelines.clustering.KddTrainXKMeans \
--driver-cores 3 \
--driver-memory 5G \
--num-executors 3 \
--executor-cores 1 \
--executor-memory 4G \
--conf spark.task.maxFailures=1 \
/tmp/mlx/sparx-ml-fat.jar \
app.kdd.training.file=\"hdfs:///tmp/sparx-ml/kdd/kddcup.data_01_percent\" \
app.kdd.path.output=\"hdfs:///tmp/sparx-ml/kdd/out\" \
app.kdd.path.wip=\"hdfs:///tmp/sparx-ml/kdd/wip\" \
app.kdd.prediction.file=\"hdfs:///tmp/sparx-ml/kdd/kddcup.newtestdata_01_percent_unlabeled\" 
```


Sample static prediction command line for running the app in a Yarn cluster with HDFS
```
spark-submit  -v  \
--master yarn-cluster \
--class tupol.sparx.ml.pipelines.clustering.KddPredictXKMeans \
--driver-cores 3 \
--driver-memory 5G \
--num-executors 3 \
--executor-cores 1 \
--executor-memory 4G \
--conf spark.task.maxFailures=1 \
/tmp/mlx/sparx-ml-fat.jar \
app.kdd.training.file=\"hdfs:///tmp/sparx-ml/kdd/kddcup.data_01_percent\" \
app.kdd.path.output=\"hdfs:///tmp/sparx-ml/kdd/out\" \
app.kdd.path.wip=\"hdfs:///tmp/sparx-ml/kdd/wip\" \
app.kdd.prediction.file=\"hdfs:///tmp/sparx-ml/kdd/kddcup.newtestdata_01_percent_unlabeled\" \
app.kdd.prediction.pipeline=\"hdfs:///tmp/sparx-ml/kdd/out/kdd_pipeline_.......plm\"
```


## Conclusions

The overall design of the Spark ML library is far superior in terms of code usability than the MLLIB.
The data transformers are very easy to chain and compose into serializable pipelines that can be in turn composed and chained.

The key concepts in ML are ***Transformer***, ***Estimator***, ***Pipeline*** and their corresponding ***Model*** classes.

For the moment the Spark ML API is far from being perfect, and writing customer transformers can be painful, since parts of the Spark ML API are package private. For now the workaround was to build some of the custom transformers into a ***shadow package***, but it is good to keep this lesson in mind.

However, as a word of caution, the interaction between Transformers is done through column names and though the API provides getters for column names, we should carefully avoid using plain strings.

The `KddPreProcessorBuilder` suggests that it should be relatively easy to provide a framework for the data scientist role to "describe" the transformations required to get to the point of usable data.

The `TrainXKMeans` suggests that it should be easy to pick the transformation pipeline provided at the previous step, configure the cross validator and get as a result the best model found, assembled in a ful pipeline, that can run from raw input to final result.


## Development Process Proposal

Given the current code, the main code that needs to be added for each input data is a corresponding `PreProcessorBuilder`.
The rest of the process can be configured without any code changes.

To be more precise, one needs to implement the following function:
```
def createPreProcessor(inputDataFrame: DataFrame,
                         rawDataColName: String, inputJsonSchema: String,
                         outputColName: String): PipelineModel
```

The key point of the `createPreProcessor` function is that it needs to perform all the required transformations on the raw input data to get it to a stage where it can be used for model training or prediction.

One can always think of it as a function from `String` to `Vector`, as actually both model training and prediction require as an input a column of `Vector`.
In the future we can add a pre-processing pipeline validation that check if this contract is met.

The inputJsonSchema parameter is a json description of the raw data after is being split/tokenized/extracted from the raw data.

The following classes/objects are just wiring in the corresponding application `PreProcessorBuilder` and the appropriate configuration.
 - `KddMain`
 - `NabMain`
 - `KddTrainXKMeans`
 - `NabTrainXKMeans`
 - `KddPredictXKMeans`
 - `NabPredictXKMeans`

As a result, the following classes and traits were moved to the `commons` package:
 - `DataImporter`
 - `TrainXKMeans`
 - `PredictXKMeans`
 - `Configuration`
 - `PreProcessorBuilder`
 

## TODOs

- [ ] Performance comparison between RDD transformations and DataFrame transformations.
- [ ] Add configuration validation for each SparkRunnable (see the corresponding validate() function)
- [ ] Code cleanup and refactoring (start with the existing TODOs from the code)
- [ ] Improve the main(args: Array[String]) transformation of args into a typesafe Configuration object
- [ ] Add more usability features at each stage (e.g. predictions might be also logged or printed)
- [ ] Design something better around the PreProcessorBuilder
- [ ] Design a better CrossValidator, that can make use of the gradient of the evaluation result, very useful when choosing the meta-parameters for an algorithm like KMeans, where the results of the evaluation take the shape of a descending exponential curve.
- [ ] Add a tool for inferring the schema of the raw input data and provide it as a starting point to the user.

