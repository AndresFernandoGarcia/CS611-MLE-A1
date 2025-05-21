import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark, csv_file_path):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    snapshot_path = os.path.join(bronze_lms_directory, snapshot_date.strftime("%Y-%m-%d"))

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # connect to source back end - IRL connect to back end source system
    # csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    df.toPandas().to_csv(snapshot_path, index=False)
    print('saved to:', filepath)

    return df
