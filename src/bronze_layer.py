from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
import os 

#paths

raw_dir = 'data/Raw/'
bronze_dir = 'data/Bronze'
silver_dir = 'data/Silver'
gold_dir = 'data/Gold'

spark = SparkSession.builder.appName("TakeHomeExamBronzeLayer").getOrCreate()


file_names = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]


bronze_dfs = []

for file_name in file_names:
    raw_file_path = os.path.join(raw_dir,file_name)
    bronze_file_path = os.path.join(bronze_dir,file_name)

    bronze_df = spark.read.csv(raw_file_path,header=True,inferSchema=True)

    bronze_df.write.option("header", "true").mode("overwrite").csv(bronze_file_path)
    
    bronze_dfs.append(bronze_df)


print("Schema of the last CSV Dataframe read:")
bronze_df.printSchema()
