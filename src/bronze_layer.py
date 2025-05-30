from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType
import os
import logging
from datetime import datetime

# --- Configuration ---
# Define paths for your data layers
raw_dir = 'data/Raw/'
bronze_dir = 'data/Bronze'
silver_dir = 'data/Silver' # Defined for completeness, not used in Bronze layer logic yet
gold_dir = 'data/Gold'     # Defined for completeness, not used in Bronze layer logic yet

# Define path for the error log file
error_log_dir = 'logs'
error_log_file_path = os.path.join(error_log_dir, 'bronze_layer_errors.log')

# Ensure directories exist
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(bronze_dir, exist_ok=True)
os.makedirs(error_log_dir, exist_ok=True) # Ensure log directory exists

# --- Logging Setup ---
# Clear existing handlers to prevent duplicate logs if run multiple times in a notebook
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(error_log_file_path), # Log to the specified file
        logging.StreamHandler()                    # Also print to console
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting PySpark Data Layer Processing (Bronze Layer)...")

# --- Spark Session Initialization ---
try:
    spark = SparkSession.builder.appName("TakeHomeExamBronzeLayer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    logger.info("SparkSession created successfully.")
except Exception as e:
    logger.error(f"Error initializing SparkSession: {e}")
    exit(1) # Exit if SparkSession cannot be created

# --- Data Processing: Raw to Bronze Layer ---

# Placeholder for a 'bad records' path if any malformed records are found during CSV read
# This path will be within the 'logs' directory
bad_records_path = os.path.join(error_log_dir, 'bad_records_from_bronze')
os.makedirs(bad_records_path, exist_ok=True)

file_names = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]

if not file_names:
    logger.warning(f"No CSV files found in the raw directory: {raw_dir}. Please ensure files are present.")

bronze_dfs = {} # Using a dictionary to store DFs by file name for easier access later

for file_name in file_names:
    raw_file_path = os.path.join(raw_dir, file_name)
    # Remove .csv extension for the bronze folder name
    bronze_output_path = os.path.join(bronze_dir, file_name.replace('.csv', ''))

    logger.info(f"Processing file: {file_name}")
    
    try:
        # Read raw CSV with options for error handling and schema inference
        bronze_df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("mode", "PERMISSIVE") \
            .option("columnNameOfCorruptRecord", "_corrupt_record") \
            .option("badRecordsPath", bad_records_path) \
            .option("sep", "\t") \
            .csv(raw_file_path)
        
        # Add ingestion timestamp
        bronze_df = bronze_df.withColumn("ingestion_timestamp", current_timestamp())

        # Check for corrupt records and log if any are found
        if "_corrupt_record" in bronze_df.columns:
            corrupt_count = bronze_df.filter(bronze_df["_corrupt_record"].isNotNull()).count()
            if corrupt_count > 0:
                logger.warning(f"Found {corrupt_count} corrupt records in {file_name}. Details in {bad_records_path}.")
            bronze_df = bronze_df.drop("_corrupt_record") # Drop the corrupt record column after logging

        # Write to Bronze layer in Parquet format
        bronze_df.write.mode("overwrite").parquet(bronze_output_path)
        logger.info(f"Successfully processed and wrote {file_name} to Bronze layer: {bronze_output_path}")
        
        bronze_dfs[file_name.replace('.csv', '')] = bronze_df # Store DF by its new folder name

    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")

# --- Schema Display (example for one DataFrame) ---
if bronze_dfs:
    # Get the key (folder name) of the last processed DataFrame
    last_df_key = list(bronze_dfs.keys())[-1] 
    logger.info(f"Schema of the last processed Bronze DataFrame ({last_df_key}):")
    bronze_dfs[last_df_key].printSchema()
else:
    logger.warning("No Bronze DataFrames were created.")

logger.info("Bronze Layer processing finished.")

# --- Stop Spark Session ---
spark.stop()
logger.info("SparkSession stopped.")