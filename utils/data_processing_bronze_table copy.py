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

        if(file == "features_financials.csv"):
            df = clean_financials(spark)

        elif(file == "features_attributes.csv"):
            df = clean_attr(spark)

        else: 
            df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

        print(snapshot_date_str + 'row count:', df.count())

        # save bronze table to datamart - IRL connect to database to write
        table_name = os.path.splitext(file)[0]
        partition_name = table_name + snapshot_date_str.replace('-', '_') + '.csv'
        output_file = os.path.join(snapshot_path, partition_name)

        df.toPandas().to_csv(output_file, index=False)
        print('saved to:', output_file)

    return False # Needs adjustment for scale but for now it is ok

def clean_financials(spark):
    finance_df = spark.read.csv("data/features_financials.csv", header=True, inferSchema=True)

    # === 1. Clean Annual_Income and Monthly_Inhand_Salary ===
    for income_col in ["Annual_Income", "Monthly_Inhand_Salary"]:
        finance_df = finance_df.withColumn(income_col, regexp_replace(col(income_col).cast("string"), r"\D+$", ""))
        finance_df = finance_df.withColumn(income_col, col(income_col).cast("float"))
        finance_df = finance_df.withColumn(income_col, Fround(col(income_col)).cast("int"))

    # === 2. Clean numeric columns with upper bounds ===
    col_upper_bounds = {
        "Num_of_Loan": 10,
        "Interest_Rate": 50,
        "Num_Credit_Card": 20,
        "Num_Bank_Accounts": 20,
        "Delay_from_due_date": 365,
        "Num_of_Delayed_Payment": 50,
        "Num_Credit_Inquiries": 50
    }

    for col_name, upper in col_upper_bounds.items():
        finance_df = finance_df.withColumn(col_name, regexp_replace(col(col_name).cast("string"), r"\D+$", ""))
        finance_df = finance_df.withColumn(col_name, col(col_name).cast("float"))
        finance_df = finance_df.withColumn(col_name, Fround(col(col_name)).cast("int"))
        finance_df = finance_df.filter(col(col_name).isNotNull() & (col(col_name) >= 0) & (col(col_name) <= upper))

    # === 3. Clean float columns ===
    float_cols = [
        "Changed_Credit_Limit", "Outstanding_Debt", "Total_EMI_per_month", 
        "Amount_invested_monthly", "Monthly_Balance", "Credit_Utilization_Ratio"
    ]

    for col_name in float_cols:
        finance_df = finance_df.withColumn(col_name, regexp_replace(col(col_name).cast("string"), r"\D+$", ""))
        finance_df = finance_df.withColumn(col_name, regexp_replace(col(col_name), r"^\D+", ""))
        finance_df = finance_df.withColumn(col_name, col(col_name).cast("float"))
        finance_df = finance_df.filter(col(col_name).isNotNull() & (col(col_name) >= 0))
        finance_df = finance_df.withColumn(col_name, Fround(col(col_name), 2))

    # === 4. Extract Credit_History_Age into numeric ===
    finance_df = finance_df.withColumn("year_val", regexp_extract(col("Credit_History_Age"), r"(?i)(\d+)\s*year", 1).cast("float"))
    finance_df = finance_df.withColumn("month_val", regexp_extract(col("Credit_History_Age"), r"(?i)(\d+)\s*month", 1).cast("float"))
    finance_df = finance_df.withColumn("Credit_History_Age_Num",
        (coalesce(col("year_val"), lit(0)) * 365 + coalesce(col("month_val"), lit(0)) * 30).cast("int")
    )

    # === 5. Clean Credit_Mix ===
    finance_df = finance_df.filter(col("Credit_Mix").isNotNull())
    finance_df = finance_df.withColumn("Credit_Mix", when(~col("Credit_Mix").rlike(r"^[A-Za-z]"), "Unknown").otherwise(col("Credit_Mix")))

    # === 6. Clean Payment_Behaviour ===
    finance_df = finance_df.filter(col("Payment_Behaviour").isNotNull())
    finance_df = finance_df.withColumn("Payment_Behaviour", when(~col("Payment_Behaviour").rlike(r"^[A-Za-z]"), "Unknown").otherwise(col("Payment_Behaviour")))

    # === 7. Drop unused columns ===
    finance_df = finance_df.drop("Type_of_Loan", "Credit_History_Age", "year_val", "month_val")

    return finance_df


def clean_attr(spark):
    # 2. Read CSV
    attr_df = spark.read.csv("data/features_attributes.csv", header=True, inferSchema=True)

    # 1. Remove nulls in Age
    attr_df = attr_df.filter(col("Age").isNotNull())

    # 2. Remove non-digit characters (underscores, letters, dots, etc.) from Age
    attr_df = attr_df.withColumn("Age", regexp_replace(col("Age").cast("string"), r"\D", ""))

    # 3. Convert Age to int
    attr_df = attr_df.withColumn("Age", col("Age").cast("int"))

    # 4. Filter Age between 0 and 130
    attr_df = attr_df.filter((col("Age") >= 0) & (col("Age") <= 130))

    # 5. Replace non-alphabetic starting values in Occupation with "Unknown"
    attr_df = attr_df.withColumn("Occupation", when(
        col("Occupation").isNull() | ~col("Occupation").rlike(r"^[A-Za-z]"),
        "Unknown"
    ).otherwise(col("Occupation")))

    # 6. Drop Name and SSN columns
    attr_df = attr_df.drop("Name", "SSN")

    return attr_df