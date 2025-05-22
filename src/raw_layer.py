from pyspark.sql import SparkSession
import os

spark = SparkSession.builder \
    .appName("TakeHomeExamRawLayer") \
    .getOrCreate()


df_address = spark.read.csv("./data/Raw/address.csv", header=True, inferSchema=True)

df_address.printSchema()
df_address.show(5)
print(f"Aantal rijen: {df_address.count()}")

# Zorg dat de Raw-folder bestaat
raw_base_path = "/app/data/Raw"
os.makedirs(raw_base_path, exist_ok=True)
df_address.write.mode("overwrite").parquet("/app/data/Raw/address")


df_check = spark.read.parquet("data/Raw/address")
df_check.show(3)