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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table():

    # Features_Attributes
    # Replace Age values that are between 0 and 130
    # And All the values that have an _ at the end of their number are removed (or a decimal place)
    
    attr_df = attr_df[attr_df["Age"].notnull()]
    attr_df["Age"] = attr_df["Age"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    attr_df = attr_df[(attr_df["Age"] >= 0) & (attr_df["Age"] <= 130)]
    attr_df["Age"]
    
    # Replace non occupations with unknown
    attr_df["Occupation"] = attr_df["Occupation"].apply(lambda x: "Unknown" if pd.isnull(x) or not str(x)[0].isalpha() else x)
    
    attr_df = attr_df.drop(columns=["Name", "SSN"])

    finance_df["Annual_Income"] = finance_df["Annual_Income"].astype(str).str.replace(r"\D$", "", regex=True).astype(float)
    finance_df["Annual_Income"] = finance_df["Annual_Income"].round().astype(int)

    finance_df["Monthly_Inhand_Salary"] = finance_df["Monthly_Inhand_Salary"].astype(str).str.replace(r"\D$", "", regex=True).astype(float)
    finance_df["Monthly_Inhand_Salary"] = finance_df["Monthly_Inhand_Salary"].round().astype(int)

    float_cols = [
        "Changed_Credit_Limit", "Outstanding_Debt", "Total_EMI_per_month", 
        "Amount_invested_monthly", "Monthly_Balance", "Credit_Utilization_Ratio"]

    col_upper_bounds = {
        "Num_of_Loan": 10,
        "Interest_Rate": 50,
        "Num_Credit_Card": 20,
        "Num_Bank_Accounts": 20,
        "Delay_from_due_date": 365,
        "Num_of_Delayed_Payment": 50,
        "Num_Credit_Inquiries": 50
    }

    for col, upper_bound in col_upper_bounds.items():
        # Remove nulls and empty strings
        finance_df = finance_df[finance_df[col].notnull() & (finance_df[col].astype(str).str.strip() != "")]

        # Remove trailing non-digit characters
        finance_df[col] = finance_df[col].astype(str).str.replace(r"\D+$", "", regex=True)

        # Convert to float and round to 0 decimals
        finance_df[col] = pd.to_numeric(finance_df[col], errors="coerce").round(0)

        # Drop rows where conversion failed
        finance_df = finance_df[finance_df[col].notnull()]

        # Convert to int
        finance_df[col] = finance_df[col].astype(int)

        # Filter valid ranges
        finance_df = finance_df[(finance_df[col] >= 0) & (finance_df[col] <= upper_bound)]

    for col in float_cols:
        # Remove trailing _
        finance_df[col] = finance_df[col].astype(str).str.replace(r"\D+$", "", regex=True)
        finance_df[col] = finance_df[col].astype(str).str.replace(r"^\D+", "", regex=True)

        # Dropping null or empty values
        finance_df = finance_df[finance_df[col].notnull() & (finance_df[col].str.strip() != "")]

        finance_df[col] = finance_df[col].astype(float)

        # Round to 2 sf
        finance_df[col] = finance_df[col].round(2)

        # Ensure that the values are >= 0
        finance_df = finance_df[finance_df[col] >= 0]

        
        numeric = pd.to_numeric(finance_df["Credit_History_Age"], errors='coerce')
        
        # Extract numeric components from text
        y = finance_df["Credit_History_Age"].str.extract(r'(?i)(\d+)\s*year', expand=False).astype(float).fillna(0)
        m = finance_df["Credit_History_Age"].str.extract(r'(?i)(\d+)\s* month', expand=False).astype(float).fillna(0)
        
        # Add numeric to df
        finance_df["Credit_History_Age_Num"] = y.mul(365).add(m.mul(30)).fillna(numeric).astype(int)
        
        # Dropping null values and replacing empty cells with Unknown
        finance_df = finance_df[finance_df["Credit_Mix"].notnull()]
        finance_df["Credit_Mix"] = finance_df["Credit_Mix"].astype(str).str.replace(r"^[^A-Za-z]+", "Unknown", regex=True)
        
        # Removing non values
        finance_df = finance_df[finance_df["Payment_Behaviour"].notnull()]
        finance_df["Payment_Behaviour"] = finance_df["Payment_Behaviour"].astype(str).str.replace(r"^[^A-Za-z]+", "Unknown", regex=True)
        
        # Dropping unneeded columns
        finance_df = finance_df.drop(columns=["Type_of_Loan", "Credit_History_Age"])
