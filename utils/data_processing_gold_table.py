import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer # for encoding

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

        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"  - Loaded {file} with {df.count()} rows")
        
        if(base_name == "features_attributes"):
            df_attr = feature_encode(df, ["Occupation"])

        elif(base_name == "features_financials"):
            df_finance = feature_encode(df, ["Credit Mix", "Payment_of_Min_Amount", "Payment_Behaviour"])

        elif(base_name == "feature_clickstream_last"):
            df_click = df
    # Join tables into one

    # save gold table - IRL connect to database to write
    partition_name = "SOMETHING_ELSE_FOR_NOW" + '_' + '.csv'
    output_file = os.path.join(gold_directory, partition_name)

    df.toPandas().to_csv(output_file, index=False)
    print('saved to:', output_file)
    
    return df

def feature_encode(df, column):
    for item in column: 
        outputc = item + "_Encoded"
        indexer = StringIndexer(
            inputCol= item,
            outputCol= outputc,
            handleInvalid="keep"
        )

        model = indexer.fit(df)
        df = model.transform(df)

        df = df.withColumn(item, col(outputc).cast(IntegerType()))
        df = df.drop(outputc) # Converted already so can be dropped
    return df