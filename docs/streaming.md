# Anomaly Detection on a Stream


## Configuration

**WARNING!**
The application is mainly a prototype, so there are no extensive input validations done on the configuration, so just make sure that the configuration is done right.
If the application does not work, it is usually a problem with the configuration parameters.

All the configuration parameters need to be prefixed by 'app.streaming'.

### General Required Parameters

| Parameter                 | Description                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| `prefix`                  | The prefix should be set for each app type (e.g 'kdd' for KDD and 'nab' for NAB) |
| `prediction.pipelines`    | Pipeline(s) used for predicting data; for multiple pipelines use a comma separated values approach; this should be the pipeline built by TranXKMeans runnable |

**Note** 
Please keep in mind, when running in a cluster to set the proper URLs for the HDFS files and paths.

**Note** 
If passing these arguments as application arguments, to escape quote the paths and property values containing standard JSON separators (like the `:` in the URL).
For example, an hdfs path parramter should be specified like this: 
`app.kdd.path.wip=\"hdfs:///my/path\"`

### File Stream Required Parameters

| Parameter                 | Description                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| `stream.file.directory`   | Directory that will be monitored by the Spark file stream for input files        |

**Note** 
Please keep in mind, when running in a cluster to set the proper URLs for the HDFS files and paths.

**Note** 
If passing these arguments as application arguments, to escape quote the paths and property values containing standard JSON separators (like the `:` in the URL).
For example, an hdfs path parramter should be specified like this: 
`stream.file.directory=\"hdfs:///my/path\"`

### Kafka Stream Required Parameters

| Parameter                 | Description                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| `stream.kafka.brokers`    | Kafka broker(s); for multiple pipelines use a comma separated values approach    |
| `stream.kafka.topics`     | Kafka topic(s); for multiple pipelines use a comma separated values approach     |

### ElasticSearch Consumer Required Parameters

| Parameter                 | Description                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| `es.index.root`           | Kafka broker(s); for multiple pipelines use a comma separated values approach    |

### Recommended Adjustable Parameters

| Parameter                            | Default    | Description                                                  |
| ------------------------------------ | ---------: | ------------------------------------------------------------ |
| `stream.batch.duration.seconds`      | 20         | Stream batch duration in seconds.                            |

### Other Properties

| Parameter                    | Description                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------- |
| `raw.input.col.name`         | Column name of the DataFrame column containing the raw data to be processed.     |
| `raw.timestamp.col.name`     | Column name of the DataFrame column containing the timestamp of the data input.  |

**WARNING!**
The parameters above should not be changed, unless there are external constraints (e.g. a third party application using the persisted column names can not work; for example Kibana does not like columns starting with '_', even though ElasticSearch has not problem with it). 


## Run Samples and Options

### Command Line

Sample static prediction command line for running the app in a Yarn cluster with HDFS using a Kafka input stream and storing the output in ElasticSearch
```
spark-submit  -v  \
--master yarn-cluster \
--class tupol.sparx.ml.streaming.KafkaToEsPredictor \
--driver-cores 3 \
--driver-memory 5G \
--num-executors 3 \
--executor-cores 1 \
--executor-memory 4G \
--conf spark.task.maxFailures=1 \
--conf es.nodes=ES_HOST_NAME:9200 \
--conf es.index.auto.create=true \
--conf es.nodes.discovery=false \
--conf es.net.http.auth.user=YOUR_USERNAME_HERE \
--conf es.net.http.auth.pass=YOUR_PASSWORD_HERE \
/tmp/mlx/sparx-ml-fat.jar \
app.streaming.stream.kafka.brokers=KAFKA_BROKER_HOST_NAME:9092 \
app.streaming.stream.kafka.topics=KAFKA_TOPIC \
app.streaming.prediction.pipelines=\"hdfs:///tmp/sparx-ml/kdd/out/kdd_pipeline_............plm\"
```


Sample static prediction command line for running the app in standalone spark using a Kafka input stream and storing the output in ElasticSearch
```
spark-submit  -v  \
--class tupol.sparx.ml.streaming.KafkaToEsPredictor \
--master local \
--conf spark.task.maxFailures=1 \
--conf es.nodes=localhost:9200 \
--conf es.index.auto.create=true \
--conf es.nodes.discovery=false \
/tmp/mlx/sparx-ml-fat.jar \
app.streaming.stream.kafka.brokers=KAFKA_BROKER_HOST_NAME:9092 \
app.streaming.stream.kafka.topics=KAFKA_TOPIC \
app.streaming.prediction.pipelines=\"/tmp/demo/kdd/out/kdd_pipeline_00e406338a0e.plm\"
```


### Ansible

TBD


## Conclusions

The current implementation of the StreamPredictor, though designed for KDD can easily work with any kind of data, provided that a proper pipeline is given. 
The current way to produce the ***aforementioned*** pipeline is to use the pattern provided in the `TrainXKMeans` class of the `demo-modgen-v2` project.


## TODOs

- [ ] Performance comparison between RDD transformations and DataFrame transformations.
- [ ] Implement Cassandra consumer (store the results to Cassandra)
- [ ] Implement Kafka consumer (store the results back to Kafka)

