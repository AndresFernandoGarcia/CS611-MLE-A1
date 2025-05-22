import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import col, regexp_replace, regexp_extract, when, lit, coalesce, round as Fround

def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):

    csv_dir = os.path.join("data")
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    snapshot_path = os.path.join(bronze_lms_directory, snapshot_date.strftime("%Y-%m-%d"))

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    for file in csv_files:
        csv_file_path = os.path.join(csv_dir, file)
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

        print(snapshot_date_str + 'row count:', df.count())

        # save bronze table to datamart - IRL connect to database to write
        table_name = os.path.splitext(file)[0]
        partition_name = table_name + snapshot_date_str.replace('-', '_') + '.csv'
        output_file = os.path.join(snapshot_path, partition_name)

        df.toPandas().to_csv(output_file, index=False)
        print('saved to:', output_file)

    return False # Needs adjustment for scale but for now it is ok
