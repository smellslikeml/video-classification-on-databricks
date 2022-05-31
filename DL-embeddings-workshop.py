# Databricks notebook source
# MAGIC %md
# MAGIC ## Deep Learning on Databricks: LSTMs and Image Embeddings using Tensorflow

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/cctvVideos/train_images/

# COMMAND ----------

import os, io
import uuid
import numpy as np
from PIL import Image
 
import tensorflow as tf
import mlflow

import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Initializing variables

# COMMAND ----------

# Load libraries
import shutil

# Set config for database name, file paths, and table names
database_name = 'dl_embedding_workshop'

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# Paths for various Delta tables
bronze_tbl_path = '/home/{}/dl_embedding_workshop/bronze/'.format(user)
silver_tbl_path = '/home/{}/dl_embedding_workshop/silver/'.format(user)

bronze_tbl_name = 'bronze'
silver_tbl_name = 'silver'

# Delete the old database and tables if needed (for demo purposes)
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_tbl_path, ignore_errors=True)

#To improve read performance when you load data back, Databricks recommends turning off compression when you save data loaded from binary files:
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Loading bronze data

# COMMAND ----------

bronze_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load("/databricks-datasets/cctvVideos/train_images/")
display(bronze_df)

# COMMAND ----------

# Create a Delta Lake table for bronze data
bronze_df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Processing silver data

# COMMAND ----------

@udf("integer")
def label_parser(path):
  # extract label from image path
  label = path.split("/")[-2].replace("label=","")
  return int(label)

@udf("string")
def video_source_parser(path):
  # extract video source from path
  source = path.split("/")[-1].split("frame")[0]
  return source

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType

schema = StructType([
    StructField("video_source_name", StringType(), True),
    StructField("image_array", ArrayType(BinaryType()), True)
])

@udf(ArrayType(IntegerType()))
def binary_to_image(sample):
  #convert list of binary strings to list of image arrays, flattened
  return np.array([np.array(Image.open(io.BytesIO(image)).resize((224,224))).flatten().tolist() for image in sample]).flatten().tolist()

@pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
def create_image_arrays(df):
  result = df.groupby(df.video_source_name)["content"].apply(list).reset_index(name="image_array")
  result["image_array"] = result["image_array"].apply(lambda x: x[:10]) #limit to 10 frames
  return result

# COMMAND ----------

silver_df = (
  bronze_df.withColumn("label", label_parser(F.col("path")))
           .withColumn("video_source_name", video_source_parser(F.col("path")))
           .select("video_source_name", "path", "content", "label")
           .sort("video_source_name", "path")
)

display(silver_df)

# COMMAND ----------

silver_image_arrays_df = silver_df.groupby("video_source_name").apply(create_image_arrays)
silver_df = silver_df.join(silver_image_arrays_df, on=["video_source_name"], how="inner")

display(silver_df)

# COMMAND ----------

silver_df.write.format('delta').mode('overwrite').save(silver_tbl_path)

# COMMAND ----------

# Create bronze table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_tbl_name,silver_tbl_path))

# COMMAND ----------

# actual dataset for training
dataset_df = (spark.table("dl_embedding_workshop.silver").select("image_array", "label")
                   .withColumn("sequences", binary_to_image(F.col("image_array")))
                   .drop("image_array").dropDuplicates()
             )

# COMMAND ----------

# Split data into two datasets for training and validation
train_df, val_df = dataset_df.randomSplit([0.9, 0.1], seed=12345)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Model

# COMMAND ----------

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM, Input

video_input = Input(shape=(10,224,224,3))
base_layer = TimeDistributed(ResNet50(include_top=False, weights='imagenet', pooling='avg'))(video_input)
lstm_layer = LSTM(128)(base_layer) 
dense_layer = Dense(128)(lstm_layer)
dropout_layer = Dropout(0.2)(dense_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

model = Model(inputs=video_input, outputs=output_layer, name="lstm_model")

for lyr in model.layers:
  if 'time_distributed' in lyr.name:
    lyr.trainable = False

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Petastorm for large scale data handling
# MAGIC 
# MAGIC Typical large datasets used for training may not fit in memory for a single machine. [Petastorm]() enables directly loading data stored in parquet format, meaning we can go from our silver Delta table to a distributed `tf.data.Dataset` without having to copy our table into a `Pandas` dataframe and wasting additional memory.

# COMMAND ----------

# Convert to tf.data.Dataset via Petastorm

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
train_df = train_df.repartition(4)
val_df = val_df.repartition(4)

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

dbutils.fs.rm("/tmp/distributed_dl_workshop/petastorm", recurse=True)
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/dl_embedding_workshop/petastorm")

converter_train = make_spark_converter(train_df)
converter_val = make_spark_converter(val_df)

train_size = len(converter_train)
val_size = len(converter_val)

# COMMAND ----------

# Reshape and apply resnet50v2 preprocessing function

def preprocess(sequence, label):
  sequence = tf.reshape(sequence, (10,224,224,3))
  sequence = tf.cast(sequence, tf.float32)
  sequence = tf.keras.applications.resnet_v2.preprocess_input(sequence)
  return sequence, label

# COMMAND ----------

batch_size = 8

with converter_train.make_tf_dataset(batch_size=batch_size) as train_ds,\
     converter_val.make_tf_dataset(batch_size=batch_size) as val_ds:
  
  # mlflow autologging
  mlflow.tensorflow.autolog()
  
  # Transforming datasets to map our preprocess function() and then batch()
  train_ds = train_ds.unbatch().map(lambda x: (x.sequences, x.label))
  val_ds = val_ds.unbatch().map(lambda x: (x.sequences, x.label))
  
  train_ds = train_ds.map(lambda sequences, label: preprocess(sequences, label)).batch(batch_size)
  val_ds = train_ds.map(lambda sequences, label: preprocess(sequences, label)).batch(batch_size)
  
  callbacks = [tf.keras.callbacks.TensorBoard('./logs')]
  
  epochs = 10
  steps_per_epoch = train_size // batch_size
  validation_steps = val_size // batch_size
  
  model.fit(train_ds,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            verbose = 1,
            validation_data = val_ds,
            validation_steps = validation_steps,
            callbacks = callbacks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring tools

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tensorboard
# MAGIC Tensorboard is a popular tool to visualize model progress. MLflow tensorfow autologging will automatically capture logs alongside model assets for later use. For convenience, we write logs locally for inline visualization. If the following two cells are executed before training, the stats will be updated as training occurs.

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC experiment_log_dir = "./logs"

# COMMAND ----------

# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ganglia Metrics
# MAGIC 
# MAGIC Found under the **Metrics** tab, [Ganglia](https://docs.databricks.com/clusters/clusters-manage.html#monitor-performance) live metrics and historical snapshots provide a view into how cluster resources are utilized at any point in time. 
# MAGIC 
# MAGIC ![ganglia metrics](https://docs.databricks.com/_images/metrics-tab.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Potential directions for further iteration
# MAGIC 
# MAGIC This workshop has only scratched the surface of possible experiments to try! Here are some suggestions on next steps.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refine data for image embedding extraction
# MAGIC The example we ran above is a very simple LSTM based architecture using ResNet50 to extract embeddings for every image sample. Although Resnet50 is a great starting point, we can further refine our embedding extraction techniques. Besides trying out other model architectures, one could consider passing the image data through an object detector to parse out specific regions of interest instead of passing the entire image data. 
# MAGIC 
# MAGIC Another knob would be to experiment with the sequence length of the video sample data. Depending on the action to classify, the duration of the sequence as well as the sampling method (ie the frames captured per second or every `x` seconds) could improve overall accuracy.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model Architectures
# MAGIC 
# MAGIC An LSTM based classifier is a great place to start and it's possible that this may be the best model architecture to implement that meets your performance requirements. However, other areas of exploration could be considering other model architectures altogether. For example, [Tf.hub](https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/3) hosts video-based model architectures pretrained on video datasets like the [Kinetics dataset](https://www.deepmind.com/open-source/kinetics) to classify video sequences. These model architectures encapsulate the embedding extraction procress which would reduce another area of manual optimization.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimizations
# MAGIC 
# MAGIC Once you've found a data processing method and model architecture that is up to your requirements, here are some additional techniques that can further optimize your training.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Early Stopping
# MAGIC 
# MAGIC Early stopping can prevent you from potentially overfitting and/or waste compute resources by monitoring a metric of interest. In this example we may measure `val_accuracy`, to end training early when the metric no longer shows improvement. You can dial the sensitivity of this setting by adjusting the `patience` parameter.
# MAGIC 
# MAGIC Add this as part of your `callbacks` parameter in the `fit()` method.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```python
# MAGIC 
# MAGIC # add Early Stopping as a list for the callbacks parameter
# MAGIC callbacks = [
# MAGIC   tf.keras.callbacks.EarlyStopping(
# MAGIC     monitor="val_loss",
# MAGIC     patience=3 # Number of epochs with no improvement after which training will be stopped
# MAGIC   )
# MAGIC ]
# MAGIC 
# MAGIC # ...
# MAGIC 
# MAGIC hist = model.fit(train_ds,
# MAGIC               steps_per_epoch = steps_per_epoch,
# MAGIC               epochs = epochs,
# MAGIC               verbose = 1,
# MAGIC               validation_data = val_ds,
# MAGIC               validation_steps = validation_steps,
# MAGIC               callbacks=callbacks)
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Larger Batch sizes
# MAGIC 
# MAGIC You can improve training speed and potentially [boost model performance](https://arxiv.org/abs/2012.08795) by increasing the batch size during training. Larger batch sizes can help stabilize training but may take more resources (memory + compute) to load and process. If you find you are underwhelming your cluster resources, this could be a good knob to turn to increase utilization and speed up training. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adapt Learning Rate
# MAGIC 
# MAGIC A larger learning rate may also help the model converge to a local minima faster by allowing the model weights to change more dramatically between epochs. Although it *may* find a local minima faster, this could lead to a suboptimal solution or even derail training all together where the model may not be able to converge. 
# MAGIC 
# MAGIC This parameter is a powerful one to experiment with and machine learning practitioners should be encouraged to do so! One can tune this by trial and error or by implementing a scheduler that can vary the parameter depending on duration of training or in response to some observed training progress.
# MAGIC 
# MAGIC More on the learning rate parameter can be found in this [reference](https://amzn.to/2NJW3gE).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distributed training with Horovod
# MAGIC In this workshop, we covered how to manage the training data in a distributed manner using Petastorm. If you find yourself needing additional resources during the model training itself, you can use Horovod to distribute the training logic across a cluster as well. This [workshop](https://github.com/smellslikeml/distributed-deep-learning-workshop) goes into depth on using Horovod with a GPU-enabled cluster for training a `tf.keras` classifier.

# COMMAND ----------


