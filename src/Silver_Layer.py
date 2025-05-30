from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    current_timestamp, lit, col, when, isnan, isnull, trim, upper, lower,
    regexp_replace, to_date, to_timestamp, coalesce, length, substring,
    round as spark_round, abs as spark_abs, split, size, concat, initcap
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType, BooleanType
import os
import logging
from datetime import datetime

# --- Configuration ---
# Define paths for your data layers
bronze_dir = 'data/Bronze'
silver_dir = 'data/Silver'
gold_dir = 'data/Gold'     # Defined for completeness, not used in Silver layer logic yet

# Define path for the error log file
error_log_dir = 'logs'
error_log_file_path = os.path.join(error_log_dir, 'silver_layer_errors.log')

# Ensure directories exist
os.makedirs(silver_dir, exist_ok=True)
os.makedirs(error_log_dir, exist_ok=True)

# --- Logging Setup ---
# Clear existing handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(error_log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting PySpark Data Layer Processing (Silver Layer)...")

# --- Spark Session Initialization ---
try:
    spark = SparkSession.builder.appName("TakeHomeExamSilverLayer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    logger.info("SparkSession created successfully.")
except Exception as e:
    logger.error(f"Error initializing SparkSession: {e}")
    exit(1)

# --- Data Quality Functions ---
def clean_string_column(df, column_name):
    """Clean string columns by trimming whitespace and handling nulls"""
    if column_name in df.columns:
        return df.withColumn(column_name, 
            when(col(column_name).isNull() | (trim(col(column_name)) == ""), None)
            .otherwise(trim(col(column_name)))
        )
    return df

def standardize_boolean_flags(df, column_name):
    """Standardize boolean flags to consistent format"""
    if column_name in df.columns:
        # Cast the column to StringType to handle mixed types gracefully
        # Then convert to boolean based on various string representations
        return df.withColumn(column_name,
            when(lower(col(column_name).cast(StringType())).isin(["1", "true", "t", "y", "yes"]), True)
            .when(lower(col(column_name).cast(StringType())).isin(["0", "false", "f", "n", "no"]), False)
            .otherwise(None)
        ).withColumn(column_name, col(column_name).cast(BooleanType())) # Ensure final type is Boolean
    return df

def validate_dates(df, column_name):
    """Validate and clean date columns"""
    if column_name in df.columns:
        return df.withColumn(column_name,
            when(col(column_name).isNull(), None)
            .otherwise(to_date(col(column_name)))
        )
    return df

def validate_numeric_positive(df, column_name):
    """Ensure numeric columns are positive where applicable"""
    if column_name in df.columns:
        return df.withColumn(column_name,
            when(col(column_name) < 0, None)
            .otherwise(col(column_name))
        )
    return df

def add_data_quality_flags(df, table_name):
    """Add data quality flags for monitoring"""
    df = df.withColumn("data_quality_score", lit(1.0))
    df = df.withColumn("source_table", lit(table_name))
    df = df.withColumn("processing_timestamp", current_timestamp())
    return df

# --- Silver Layer Processing Functions ---

def process_address(bronze_df):
    """Process address table with data quality checks"""
    logger.info("Processing address table...")
    
    df = bronze_df
    
    # Clean string columns
    string_cols = ['AddressLine1', 'AddressLine2', 'City', 'PostalCode']
    for col_name in string_cols:
        df = clean_string_column(df, col_name)
    
    # Validate postal codes (basic format check)
    if 'PostalCode' in df.columns:
        df = df.withColumn('PostalCode',
            when(length(col('PostalCode')) < 3, None)
            .otherwise(col('PostalCode'))
        )
    
    # Validate dates
    df = validate_dates(df, 'ModifiedDate')
    
    # Add quality flags
    df = add_data_quality_flags(df, 'address')
    
    return df

def process_customer(bronze_df):
    """Process customer table"""
    logger.info("Processing customer table...")
    
    df = bronze_df
    
    # Clean account number
    df = clean_string_column(df, 'AccountNumber')
    
    # Validate IDs are positive
    id_cols = ['CustomerID', 'PersonID', 'StoreID', 'TerritoryID']
    for col_name in id_cols:
        df = validate_numeric_positive(df, col_name)
    
    # Validate dates
    df = validate_dates(df, 'ModifiedDate')
    
    df = add_data_quality_flags(df, 'customer')
    return df

def process_person(bronze_df):
    """Process person table with name standardization"""
    logger.info("Processing person table...")

    df = bronze_df

    # Clean and standardize name fields
    name_cols = ['FirstName', 'MiddleName', 'LastName', 'Title', 'Suffix']
    for col_name in name_cols:
        df = clean_string_column(df, col_name)
        if col_name in df.columns:
            # Capitalize first letter of each word using initcap()
            df = df.withColumn(col_name,
                when(col(col_name).isNotNull(), initcap(col(col_name)))
                .otherwise(None)
            )

    # Standardize PersonType
    df = clean_string_column(df, 'PersonType')

    # Validate EmailPromotion values
    if 'EmailPromotion' in df.columns:
        df = df.withColumn('EmailPromotion',
            when(col('EmailPromotion').isin([0, 1, 2]), col('EmailPromotion'))
            .otherwise(0)
        )

    # Validate and clean date fields
    df = validate_dates(df, 'ModifiedDate')

    # Add DQ flags
    df = add_data_quality_flags(df, 'person')

    return df


def process_product(bronze_df):
    """Process product table with comprehensive validation"""
    logger.info("Processing product table...")
    
    df = bronze_df
    
    # Clean string columns
    string_cols = ['Name', 'ProductNumber', 'Color', 'Size', 'ProductLine', 'Class', 'Style']
    for col_name in string_cols:
        df = clean_string_column(df, col_name)
    
    # Standardize boolean flags
    bool_cols = ['MakeFlag', 'FinishedGoodsFlag']
    for col_name in bool_cols:
        df = standardize_boolean_flags(df, col_name)
    
    # Validate numeric columns
    numeric_cols = ['SafetyStockLevel', 'ReorderPoint', 'StandardCost', 'ListPrice', 'Weight', 'DaysToManufacture']
    for col_name in numeric_cols:
        df = validate_numeric_positive(df, col_name)
    
    # Validate price consistency (ListPrice >= StandardCost)
    if 'ListPrice' in df.columns and 'StandardCost' in df.columns:
        df = df.withColumn('price_validation_flag',
            when((col('ListPrice').isNotNull() & col('StandardCost').isNotNull()),
                col('ListPrice') >= col('StandardCost')
            ).otherwise(True)
        )
    
    # Validate dates
    date_cols = ['SellStartDate', 'SellEndDate', 'DiscontinuedDate', 'ModifiedDate']
    for col_name in date_cols:
        df = validate_dates(df, col_name)
    
    df = add_data_quality_flags(df, 'product')
    return df

def process_sales_order_header(bronze_df):
    """Process sales order header with business logic validation"""
    logger.info("Processing salesorderheader table...")
    
    df = bronze_df
    
    # Clean string columns
    string_cols = ['SalesOrderNumber', 'PurchaseOrderNumber', 'AccountNumber', 'Comment']
    for col_name in string_cols:
        df = clean_string_column(df, col_name)
    
    # Standardize boolean flags
    df = standardize_boolean_flags(df, 'OnlineOrderFlag')
    
    # Validate dates and date logic
    date_cols = ['OrderDate', 'DueDate', 'ShipDate', 'ModifiedDate']
    for col_name in date_cols:
        df = validate_dates(df, col_name)
    
    # Validate date sequence (OrderDate <= DueDate, OrderDate <= ShipDate)
    df = df.withColumn('date_sequence_valid',
        when(
            (col('OrderDate').isNotNull() & col('DueDate').isNotNull()),
            col('OrderDate') <= col('DueDate')
        ).otherwise(True) &
        when(
            (col('OrderDate').isNotNull() & col('ShipDate').isNotNull()),
            col('OrderDate') <= col('ShipDate')
        ).otherwise(True)
    )
    
    # Validate monetary amounts
    money_cols = ['SubTotal', 'TaxAmt', 'Freight', 'TotalDue']
    for col_name in money_cols:
        df = validate_numeric_positive(df, col_name)
    
    # Validate total calculation (TotalDue = SubTotal + TaxAmt + Freight)
    if all(col_name in df.columns for col_name in ['SubTotal', 'TaxAmt', 'Freight', 'TotalDue']):
        df = df.withColumn('total_calculation_valid',
            when(
                col('SubTotal').isNotNull() & 
                col('TaxAmt').isNotNull() & 
                col('Freight').isNotNull() &
                col('TotalDue').isNotNull(),
                spark_abs(col('TotalDue') - (col('SubTotal') + col('TaxAmt') + col('Freight'))) < 0.01
            ).otherwise(True)
        )
    
    df = add_data_quality_flags(df, 'salesorderheader')
    return df

def process_sales_order_detail(bronze_df):
    """Process sales order detail with calculation validation"""
    logger.info("Processing salesorderdetail table...")
    
    df = bronze_df
    
    # Clean string columns
    df = clean_string_column(df, 'CarrierTrackingNumber')
    
    # Validate numeric columns
    numeric_cols = ['OrderQty', 'UnitPrice', 'UnitPriceDiscount', 'LineTotal']
    for col_name in numeric_cols:
        df = validate_numeric_positive(df, col_name)
    
    # Validate line total calculation
    if all(col_name in df.columns for col_name in ['OrderQty', 'UnitPrice', 'UnitPriceDiscount', 'LineTotal']):
        df = df.withColumn('line_total_valid',
            when(
                col('OrderQty').isNotNull() & 
                col('UnitPrice').isNotNull() & 
                col('UnitPriceDiscount').isNotNull() &
                col('LineTotal').isNotNull(),
                spark_abs(col('LineTotal') - (col('OrderQty') * col('UnitPrice') * (1 - col('UnitPriceDiscount')))) < 0.01
            ).otherwise(True)
        )
    
    df = validate_dates(df, 'ModifiedDate')
    df = add_data_quality_flags(df, 'salesorderdetail')
    return df

def process_creditcard(bronze_df):
    """Process credit card with sensitive data handling"""
    logger.info("Processing creditcard table...")
    
    df = bronze_df
    
    # Clean card type
    df = clean_string_column(df, 'CardType')
    
    # Mask credit card number (keep last 4 digits)
    if 'CardNumber' in df.columns:
        df = df.withColumn('CardNumber',
            when(col('CardNumber').isNotNull() & (length(col('CardNumber')) >= 4),
                concat(lit("****-****-****-"), substring(col('CardNumber'), -4, 4))
            ).otherwise(lit("****-****-****-****"))
        )
    
    # Validate expiration dates
    if 'ExpMonth' in df.columns:
        df = df.withColumn('ExpMonth',
            when((col('ExpMonth') >= 1) & (col('ExpMonth') <= 12), col('ExpMonth'))
            .otherwise(None)
        )
    
    if 'ExpYear' in df.columns:
        df = df.withColumn('ExpYear',
            when((col('ExpYear') >= 2020) & (col('ExpYear') <= 2050), col('ExpYear'))
            .otherwise(None)
        )
    
    df = validate_dates(df, 'ModifiedDate')
    df = add_data_quality_flags(df, 'creditcard')
    return df

def process_generic_table(bronze_df, table_name):
    """Generic processing for simpler tables"""
    logger.info(f"Processing {table_name} table...")
    
    df = bronze_df
    
    # Clean all string columns
    for col_name, col_type in df.dtypes:
        if col_type == 'string':
            df = clean_string_column(df, col_name)
    
    # Validate ModifiedDate if exists
    df = validate_dates(df, 'ModifiedDate')
    
    # Add quality flags
    df = add_data_quality_flags(df, table_name)
    return df

# --- Main Processing Logic ---

# Dictionary mapping table names to their specific processing functions
table_processors = {
    'address': process_address,
    'customer': process_customer,
    'person': process_person,
    'product': process_product,
    'salesorderheader': process_sales_order_header,
    'salesorderdetail': process_sales_order_detail,
    'creditcard': process_creditcard
}

# Get list of Bronze layer tables
bronze_tables = [d for d in os.listdir(bronze_dir) if os.path.isdir(os.path.join(bronze_dir, d))]

if not bronze_tables:
    logger.warning(f"No Bronze tables found in directory: {bronze_dir}")
    exit(1)

silver_dfs = {}
processing_stats = {}

for table_name in bronze_tables:
    bronze_table_path = os.path.join(bronze_dir, table_name)
    silver_table_path = os.path.join(silver_dir, table_name)
    
    try:
        logger.info(f"Processing Bronze table: {table_name}")
        
        # Read Bronze layer data
        bronze_df = spark.read.parquet(bronze_table_path)
        
        # Get initial record count
        initial_count = bronze_df.count()
        
        # Apply specific processing function or generic processing
        if table_name in table_processors:
            silver_df = table_processors[table_name](bronze_df)
        else:
            silver_df = process_generic_table(bronze_df, table_name)
        
        # Get final record count
        final_count = silver_df.count()
        
        # Calculate null percentages for key columns
        total_cols = len(silver_df.columns)
        null_counts = {}
        for col_name in silver_df.columns:
            if col_name not in ['ingestion_timestamp', 'processing_timestamp', 'data_quality_score', 'source_table']:
                null_count = silver_df.filter(col(col_name).isNull()).count()
                null_percentage = (null_count / final_count * 100) if final_count > 0 else 0
                null_counts[col_name] = null_percentage
        
        # Store processing statistics
        processing_stats[table_name] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'records_dropped': initial_count - final_count,
            'null_percentages': null_counts
        }
        
        # Write to Silver layer
        silver_df.write.mode("overwrite").parquet(silver_table_path)
        logger.info(f"Successfully processed {table_name}: {initial_count} -> {final_count} records")
        
        # Store DataFrame for potential further use
        silver_dfs[table_name] = silver_df
        
    except Exception as e:
        logger.error(f"Error processing {table_name}: {e}")
        processing_stats[table_name] = {'error': str(e)}

# --- Data Quality Report ---
logger.info("=== SILVER LAYER DATA QUALITY REPORT ===")
for table_name, stats in processing_stats.items():
    if 'error' in stats:
        logger.error(f"{table_name}: Processing failed - {stats['error']}")
    else:
        logger.info(f"{table_name}:")
        logger.info(f"  Records: {stats['initial_count']} -> {stats['final_count']}")
        if stats['records_dropped'] > 0:
            logger.warning(f"  Dropped records: {stats['records_dropped']}")
        
        # Report columns with high null percentages
        high_null_cols = {k: v for k, v in stats['null_percentages'].items() if v > 20}
        if high_null_cols:
            logger.warning(f"  High null percentage columns: {high_null_cols}")

# --- Schema Display (example for one DataFrame) ---
if silver_dfs:
    sample_table = list(silver_dfs.keys())[0]
    logger.info(f"Sample Silver DataFrame schema ({sample_table}):")
    silver_dfs[sample_table].printSchema()

logger.info("Silver Layer processing finished.")

# --- Stop Spark Session ---
spark.stop()
logger.info("SparkSession stopped.")