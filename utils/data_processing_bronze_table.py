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

def process_bronze_table(spark):

    csv_dir = os.path.join("data")
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    # create bronze datalake
    bronze_directory = "datamart/bronze/features/"

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)
    
    for file in csv_files:
        csv_file_path = os.path.join(csv_dir, file)
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

        # Label store handled by another file
        if(file == "lms_loan_daily.csv"):
            continue

        print('row count:', df.count())

        # save bronze table to datamart - IRL connect to database to write
        table_name = os.path.splitext(file)[0]
        partition_name = table_name +'.csv'
        output_file = os.path.join(bronze_directory, partition_name)

        df.toPandas().to_csv(output_file, index=False)
        print('saved to:', output_file)

    return False # Needs adjustment for scale but for now it is ok
