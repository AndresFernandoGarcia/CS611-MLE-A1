import os
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum as Fsum
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import col, regexp_replace, regexp_extract, when, lit, coalesce, round as Fround, row_number, desc

def process_silver_table(spark):
    silver_directory = "datamart/silver/features"

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    bronze_directory = "datamart/bronze/features"
    csv_files = [f for f in os.listdir(bronze_directory) if f.endswith(".csv")]

    for file in csv_files:
        file_path = os.path.join(bronze_directory, file)
        base_name = os.path.splitext(file)[0]

        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"  - Loaded {file} with {df.count()} rows")

        if(base_name == "features_attributes"):
            df = clean_attr(df)

        elif(base_name == "features_financials"):
            df = clean_financials(df)

        elif(base_name == "feature_clickstream"):
            df = clean_clickstream(df)

            # Cleaning clickstream to condense one to many relationship
            df_t = clickstream_get_latest_snap(df)

            # Saving last snapshot
            partition_name = "feature_clickstream_last" + '.csv'
            output_file = os.path.join(silver_directory, partition_name)
            df_t.toPandas().to_csv(output_file, index=False)
            print('saved to:', output_file)

        # save silver table to datamart - IRL connect to database to write
        partition_name = base_name + '.csv'
        output_file = os.path.join(silver_directory, partition_name)

        df.toPandas().to_csv(output_file, index=False)
        print('saved to:', output_file)
    return True

def clean_financials(df):

    # Creating dictionary to enforce types
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": IntegerType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age_Num": IntegerType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # Clean Annual_Income and Monthly_Inhand_Salary
    for income_col in ["Annual_Income", "Monthly_Inhand_Salary"]:
        df = df.withColumn(income_col, regexp_replace(col(income_col).cast("string"), r"\D+$", ""))
        df = df.withColumn(income_col, col(income_col).cast("float"))
        df = df.withColumn(income_col, Fround(col(income_col)).cast("int"))

    # Clean numeric columns with upper bounds
    col_upper_bounds = {
        "Num_of_Loan": 10,
        "Interest_Rate": 50,
        "Num_Credit_Card": 20,
        "Num_Bank_Accounts": 20,
        "Delay_from_due_date": 365,
        "Num_of_Delayed_Payment": 50,
        "Num_Credit_Inquiries": 50
    }

    # Clean integer columns - (columns with integer values) - require same processing
    for col_name, upper in col_upper_bounds.items():
        df = df.withColumn(col_name, regexp_replace(col(col_name).cast("string"), r"[^\d.]+", ""))
        df = df.withColumn(col_name, col(col_name).cast("float"))
        df = df.withColumn(col_name, Fround(col(col_name)).cast("int"))
        df = df.filter(col(col_name).isNotNull() & (col(col_name) >= 0) & (col(col_name) <= upper))

    # Clean float columns - (columns with float values)
    float_cols = [
        "Changed_Credit_Limit", "Outstanding_Debt", "Total_EMI_per_month", 
        "Amount_invested_monthly", "Monthly_Balance", "Credit_Utilization_Ratio"
    ]

    for col_name in float_cols:
        df = df.withColumn(col_name, regexp_replace(col(col_name).cast("string"), r"\D+$", ""))
        df = df.withColumn(col_name, regexp_replace(col(col_name), r"^\D+", ""))
        df = df.withColumn(col_name, col(col_name).cast("float"))
        df = df.filter(col(col_name).isNotNull() & (col(col_name) >= 0))
        df = df.withColumn(col_name, Fround(col(col_name), 2))

    # Extract Credit_History_Age into numeric
    df = df.withColumn("year_val", regexp_extract(col("Credit_History_Age"), r"(?i)(\d+)\s*year", 1).cast("float"))
    df = df.withColumn("month_val", regexp_extract(col("Credit_History_Age"), r"(?i)(\d+)\s*month", 1).cast("float"))
    df = df.withColumn("Credit_History_Age_Num",
        (coalesce(col("year_val"), lit(0)) * 365 + coalesce(col("month_val"), lit(0)) * 30).cast("int")
    )

    # Clean Credit_Mix 
    df = df.filter(col("Credit_Mix").isNotNull())
    df = df.withColumn("Credit_Mix", when(~col("Credit_Mix").rlike(r"^[A-Za-z]"), "Unknown").otherwise(col("Credit_Mix")))

    # Clean Payment_Behaviour
    df = df.filter(col("Payment_Behaviour").isNotNull())
    df = df.withColumn("Payment_Behaviour", when(~col("Payment_Behaviour").rlike(r"^[A-Za-z]"), "Unknown").otherwise(col("Payment_Behaviour")))

    # Drop unused columns
    df = df.drop("Type_of_Loan", "Credit_History_Age", "year_val", "month_val")

    # Enforce schema / data type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    return df

def clean_attr(df):
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    # Remove nulls in Age
    df = df.filter(col("Age").isNotNull())

    # Remove non-digit characters (underscores, letters, dots, etc.) from Age
    df = df.withColumn("Age", regexp_replace(col("Age").cast("string"), r"\D", ""))

    # Filter Age between 0 and 130
    df = df.filter((col("Age") >= 0) & (col("Age") <= 130))

    # Replace non-alphabetic starting values in Occupation with "Unknown"
    df = df.withColumn("Occupation", when(
        col("Occupation").isNull() | ~col("Occupation").rlike(r"^[A-Za-z]"),
        "Unknown"
    ).otherwise(col("Occupation")))

    # Drop Name and SSN columns
    df = df.drop("Name", "SSN")

    # Enforce schema / data type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    return df

def clean_clickstream(df):
    column_type_map = {
    "fe_1": IntegerType(),
    "fe_2": IntegerType(),
    "fe_3": IntegerType(),
    "fe_4": IntegerType(),
    "fe_5": IntegerType(),
    "fe_6": IntegerType(),
    "fe_7": IntegerType(),
    "fe_8": IntegerType(),
    "fe_9": IntegerType(),
    "fe_10": IntegerType(),
    "fe_11": IntegerType(),
    "fe_12": IntegerType(),
    "fe_13": IntegerType(),
    "fe_14": IntegerType(),
    "fe_15": IntegerType(),
    "fe_16": IntegerType(),
    "fe_17": IntegerType(),
    "fe_18": IntegerType(),
    "fe_19": IntegerType(),
    "fe_20": IntegerType(),
    "Customer_ID": StringType(),
    "snapshot_date": DateType(),
}
    
    # Enforce schema / data type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    return df

def clickstream_get_latest_snap(df):
    # Ensure snapshot_date is a DateType (to avoid errors in future)
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # Partition by Customer_ID, order by snapshot_date descending
    w = Window.partitionBy("Customer_ID").orderBy(desc("snapshot_date"))

    # Ranking records
    df = df.withColumn("row_num", row_number().over(w))

    # Get the number 1 record (which should be latest)
    df_latest = df.filter(col("row_num") == 1).drop("row_num")

    return df_latest

# Aggregate values of each Customer_ID in clickstream for a better representation of values
# Doesn't work... more of a proof of concept (could be implemented at a later stage)
def clickstream_feature_aggregation(df):
    # Cast snapshot_date to DateType (as a precaution)
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # Get all columns that are integers
    int_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, IntegerType) 
        and f.name not in ["Customer_ID", "snapshot_date"]
    ]

    # Group by Customer_ID and sum all numeric fields
    agg_exprs = [Fsum(col(c)).alias(c) for c in int_cols]
    df_agg = df.groupBy("Customer_ID").agg(*agg_exprs)

    return df_agg


# def clean_loan_daily(df):
#     # clean data: enforce schema / data type
#     # Dictionary specifying columns and their desired datatypes
#     column_type_map = {
#         "loan_id": StringType(),
#         "Customer_ID": StringType(),
#         "loan_start_date": DateType(),
#         "tenure": IntegerType(),
#         "installment_num": IntegerType(),
#         "loan_amt": FloatType(),
#         "due_amt": FloatType(),
#         "paid_amt": FloatType(),
#         "overdue_amt": FloatType(),
#         "balance": FloatType(),
#         "snapshot_date": DateType(),
#     }

#     for column, new_type in column_type_map.items():
#         df = df.withColumn(column, col(column).cast(new_type))

#     # augment data: add month on book
#     df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

#     # augment data: add days past due
#     df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")))
#     df = df.withColumn("installments_missed", coalesce(col("installments_missed"), lit(0)).cast(IntegerType()))
#     df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
#     df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

#     return df