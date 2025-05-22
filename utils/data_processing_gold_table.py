import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(spark):
    
    gold_directory = "datamart/gold/feature_store"

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    silver_directory = "datamart/silver/features"

    #Prepare data for model (encode)
    csv_files = [f for f in os.listdir(silver_directory) if f.endswith(".csv")]
    
    for file in csv_files:
        file_path = os.path.join(silver_directory, file)
        base_name = os.path.splitext(file)[0]
        print(base_name)

        df = spark.read.csv(file_path, header=True, inferSchema=True)

    # save gold table - IRL connect to database to write
    partition_name = "SOMETHING_ELSE_FOR_NOW" + '_' + '.csv'
    output_file = os.path.join(gold_directory, partition_name)

    df.toPandas().to_csv(output_file, index=False)
    print('saved to:', output_file)
    
    return df