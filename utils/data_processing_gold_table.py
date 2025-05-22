import os
from pyspark.ml.feature import StringIndexer # for encoding
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

def process_labels_gold_table(spark):
    
    gold_directory = "datamart/gold/feature_store"

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    silver_directory = "datamart/silver/features"

    #Prepare data for model (encode)
    csv_files = [f for f in os.listdir(silver_directory) if f.endswith(".csv")]
    
    for file in csv_files:
        file_path = os.path.join(silver_directory, file)
        base_name = os.path.splitext(file)[0]
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"  - Loaded {file} with {df.count()} rows")
        
        print("Feature Store currently working with file: ", base_name)
        if(base_name == "features_attributes"):
            df_attr = feature_encode(df, ["Occupation"])
            df_attr = df_attr.drop("snapshot_date")

        elif(base_name == "features_financials"):
            df_finance = feature_encode(df, ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"])

        elif(base_name == "feature_clickstream_last"):
            df_click = df
            df_click = df.drop("snapshot_date")
    # Join tables into one

    df_combined = df_finance.join(df_attr, on="Customer_ID", how="outer")
    df_combined = df_combined.join(df_click, on="Customer_ID", how="outer")

    # Inserting snapshot_date after Customer_ID
    cols = df_combined.columns
    cols.remove("snapshot_date")

    # Insert After Customer_ID
    cols.insert(cols.index("Customer_ID") + 1, "snapshot_date")

    # Reorder the DataFrame 
    df_combined = df_combined.select(cols)
    
    # Save gold table - IRL connect to database to write
    partition_name = "feature_store" + '.csv'
    output_file = os.path.join(gold_directory, partition_name)
    df_combined.toPandas().to_csv(output_file, index=False)
    print('saved to:', output_file)
    return df

# numerical encoding values
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