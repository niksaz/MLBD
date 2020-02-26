#!/bin/bash

set -u
set -e
set -i

# Update Spark installation

sudo yum -y remove java
sudo yum install -y java-1.8.0-openjdk
export JAVA_HOME=/usr/lib/jvm/jre-1.8.0-openjdk.x86_64

SPARK_VERSION_NAME="spark-2.4.5-bin-hadoop2.6"

tar -xvf "${SPARK_VERSION_NAME}.tgz"
sudo mv "$SPARK_VERSION_NAME" /usr/local/spark


for FILE in /usr/bin/pyspark /usr/bin/spark-shell /usr/bin/spark-submit; do
  sed -i -e 's+/usr/lib/spark/+/usr/local/spark+g' "${FILE}"
done

export SPARK_HOME=/usr/local/spark
export PATH=$SPARK_HOME/bin:$PATH
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip

# Run Jupyter

jupyter notebook --no-browser --port 9999 --ip='*' --allow-root

