from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    current_timestamp, lit, col, when, isnan, isnull, trim, upper, lower,
    regexp_replace, to_date, to_timestamp, coalesce, length, substring,
    round as spark_round, abs as spark_abs, split, size, concat, sum as spark_sum,
    count, max as spark_max, min as spark_min, avg, year, month, dayofmonth,
    dayofweek, quarter, weekofyear, date_format, row_number, rank, dense_rank,
    lag, lead, first, last, collect_list, collect_set, explode, array_contains,
    struct, desc, asc, monotonically_increasing_id, hash, md5, sha1,
    regexp_extract, split as spark_split, slice as spark_slice,
    date_add, date_sub, datediff, months_between, next_day, last_day,
    from_unixtime, unix_timestamp, date_trunc, concat_ws
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType, BooleanType, LongType
from pyspark.sql.window import Window
import os
import logging
from datetime import datetime, date, timedelta

# --- Configuration ---
silver_dir = 'data/Silver'
gold_dir = 'data/Gold'

# Define path for the error log file
error_log_dir = 'logs'
error_log_file_path = os.path.join(error_log_dir, 'gold_layer_errors.log')

# Ensure directories exist
os.makedirs(gold_dir, exist_ok=True)
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

logger.info("Starting PySpark Data Layer Processing (Gold Layer with SCD and Upsert)...")

# --- Spark Session Initialization ---
try:
    spark = SparkSession.builder.appName("TakeHomeExamGoldLayer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    logger.info("SparkSession created successfully.")
except Exception as e:
    logger.error(f"Error initializing SparkSession: {e}")
    exit(1)

def generate_surrogate_key(df, columns):
    """Generate surrogate key based on multiple columns"""
    concat_cols = [coalesce(col(c).cast(StringType()), lit("NULL")) for c in columns]
    return df.withColumn("surrogate_key",
                         spark_abs(hash(concat(*concat_cols))))

def safe_table_read(table_path, table_name):
    """Safely read a table with error handling"""
    try:
        return spark.read.parquet(table_path)
    except Exception as e:
        logger.warning(f"Could not read {table_name} from {table_path}: {e}")
        return None

def add_scd_columns(df):
    """Add SCD Type 2 columns to dimension tables"""
    return df.withColumn("effective_date", current_timestamp()) \
             .withColumn("end_date", lit(None).cast(TimestampType())) \
             .withColumn("is_current", lit(True)) \
             .withColumn("version", lit(1))

def upsert_dimension_table(new_data, table_path, table_name, business_key_columns):
    """
    Perform SCD Type 2 upsert on dimension table
    """
    logger.info(f"Performing SCD Type 2 upsert for {table_name}...")
    
    # Check if table exists
    if os.path.exists(table_path):
        try:
            existing_data = spark.read.parquet(table_path)
            logger.info(f"Found existing {table_name} with {existing_data.count()} records")
            
            # Get current records (is_current = True)
            current_records = existing_data.filter(col("is_current") == True)
            
            # Create a comparison key for detecting changes
            comparison_cols = [c for c in new_data.columns if c not in ['created_date', 'modified_date', 'effective_date', 'end_date', 'is_current', 'version', 'surrogate_key']]
            
            # Join to find changes
            joined_df = new_data.alias("new").join(
                current_records.alias("existing"),
                [col(f"new.{bk}") == col(f"existing.{bk}") for bk in business_key_columns],
                "left"
            )
            
            # Identify new records (no match in existing)
            new_records = joined_df.filter(col("existing.surrogate_key").isNull()) \
                                   .select("new.*")
            
            # Identify changed records
            changed_records = joined_df.filter(col("existing.surrogate_key").isNotNull())
            
            # Check for actual changes in data
            change_conditions = []
            for col_name in comparison_cols:
                if col_name in business_key_columns:
                    continue
                change_conditions.append(
                    coalesce(col(f"new.{col_name}"), lit("NULL")) != 
                    coalesce(col(f"existing.{col_name}"), lit("NULL"))
                )
            
            if change_conditions:
                changed_filter = change_conditions[0]
                for condition in change_conditions[1:]:
                    changed_filter = changed_filter | condition
                
                actually_changed = changed_records.filter(changed_filter)
                unchanged = changed_records.filter(~changed_filter)
            else:
                actually_changed = spark.createDataFrame([], new_data.schema)
                unchanged = changed_records
            
            # Handle unchanged records - keep existing
            unchanged_existing = unchanged.select("existing.*")
            
            # Handle new records - add with SCD columns
            final_new_records = add_scd_columns(new_records)
            
            # Handle changed records
            if actually_changed.count() > 0:
                # Close existing records
                closed_records = actually_changed.select("existing.*") \
                    .withColumn("end_date", current_timestamp()) \
                    .withColumn("is_current", lit(False))
                
                # Create new versions of changed records
                new_versions = actually_changed.select("new.*") \
                    .join(actually_changed.select("existing.version", *[f"existing.{bk}" for bk in business_key_columns]),
                          [col(f"new.{bk}") == col(f"existing.{bk}") for bk in business_key_columns]) \
                    .withColumn("effective_date", current_timestamp()) \
                    .withColumn("end_date", lit(None).cast(TimestampType())) \
                    .withColumn("is_current", lit(True)) \
                    .withColumn("version", col("existing.version") + 1) \
                    .drop(*[f"existing.{bk}" for bk in business_key_columns]) \
                    .drop("existing.version")
                
                # Combine all records
                historical_records = existing_data.filter(col("is_current") == False)
                result_df = historical_records.union(unchanged_existing) \
                                             .union(closed_records) \
                                             .union(final_new_records) \
                                             .union(new_versions)
            else:
                # No changes, just add new records
                historical_records = existing_data.filter(col("is_current") == False)
                result_df = historical_records.union(unchanged_existing) \
                                             .union(final_new_records)
                
            logger.info(f"SCD upsert completed: {result_df.count()} total records")
            return result_df
            
        except Exception as e:
            logger.error(f"Error during SCD upsert for {table_name}: {e}")
            # Fall back to initial load
            return add_scd_columns(new_data)
    else:
        # Initial load - add SCD columns
        logger.info(f"Initial load for {table_name}")
        return add_scd_columns(new_data)

def upsert_fact_table(new_data, table_path, table_name, business_key_columns):
    """
    Perform upsert on fact table (Type 1 - overwrite existing records)
    """
    logger.info(f"Performing fact table upsert for {table_name}...")
    
    if os.path.exists(table_path):
        try:
            existing_data = spark.read.parquet(table_path)
            logger.info(f"Found existing {table_name} with {existing_data.count()} records")
            
            # For fact tables, we typically do a merge (upsert)
            # Remove existing records that match business keys
            existing_to_keep = existing_data.alias("existing").join(
                new_data.alias("new"),
                [col(f"existing.{bk}") == col(f"new.{bk}") for bk in business_key_columns],
                "left_anti"
            )
            
            # Combine with new data
            result_df = existing_to_keep.union(new_data)
            
            logger.info(f"Fact upsert completed: {result_df.count()} total records")
            return result_df
            
        except Exception as e:
            logger.error(f"Error during fact upsert for {table_name}: {e}")
            return new_data
    else:
        logger.info(f"Initial load for fact table {table_name}")
        return new_data

def create_date_dimension():
    """Create comprehensive date dimension table"""
    logger.info("Creating date dimension...")

    # Create date range from 2010 to 2030
    start_date = date(2010, 1, 1)
    end_date = date(2030, 12, 31)

    # Generate date range
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append((current_date,))
        current_date += timedelta(days=1)

    schema = StructType([StructField("date", DateType(), True)])
    date_df = spark.createDataFrame(date_list, schema)

    # Add date dimension attributes
    date_dim = date_df.select(
        col("date").alias("date_key"),
        col("date"),
        year("date").alias("year"),
        month("date").alias("month"),
        dayofmonth("date").alias("day"),
        dayofweek("date").alias("day_of_week"),
        quarter("date").alias("quarter"),
        weekofyear("date").alias("week_of_year"),
        date_format("date", "MMMM").alias("month_name"),
        date_format("date", "EEEE").alias("day_name"),
        when(dayofweek("date").isin([1, 7]), True).otherwise(False).alias("is_weekend"),
        date_format("date", "yyyy-MM").alias("year_month"),
        concat_ws("Q", year("date").cast("string"), quarter("date").cast("string")).alias("year_quarter"),
        current_timestamp().alias("created_date"),
        current_timestamp().alias("modified_date")
    )

    return date_dim

def create_customer_dimension():
    """Create customer dimension with person info"""
    logger.info("Creating customer dimension...")

    # Read Silver layer data
    customer_df = safe_table_read(os.path.join(silver_dir, 'customer'), 'customer')
    person_df = safe_table_read(os.path.join(silver_dir, 'person'), 'person')
    
    if customer_df is None:
        logger.error("Customer table not found in Silver layer")
        return None
    
    if person_df is None:
        logger.warning("Person table not found, creating customer dimension without person details")
        customer_dim = customer_df.select(
            col("CustomerID").alias("customer_key"),
            col("CustomerID").alias("customer_id"),
            col("AccountNumber").alias("account_number"),
            lit("Unknown").alias("first_name"),
            lit("Unknown").alias("middle_name"),
            lit("Unknown").alias("last_name"),
            lit("Unknown").alias("full_name"),
            lit("Unknown").alias("person_type"),
            col("TerritoryID").alias("territory_id"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )
    else:
        # Join customer with person data
        customer_dim = customer_df.join(
            person_df,
            customer_df.PersonID == person_df.BusinessEntityID,
            "left"
        ).select(
            customer_df.CustomerID.alias("customer_key"),
            customer_df.CustomerID.alias("customer_id"),
            customer_df.AccountNumber.alias("account_number"),
            coalesce(person_df.FirstName, lit("Unknown")).alias("first_name"),
            coalesce(person_df.MiddleName, lit("")).alias("middle_name"),
            coalesce(person_df.LastName, lit("Unknown")).alias("last_name"),
            concat(
                coalesce(person_df.FirstName, lit("Unknown")), 
                lit(" "),
                coalesce(person_df.LastName, lit("Unknown"))
            ).alias("full_name"),
            coalesce(person_df.PersonType, lit("Unknown")).alias("person_type"),
            customer_df.TerritoryID.alias("territory_id"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )

    # Add surrogate key
    customer_dim = generate_surrogate_key(customer_dim, ["customer_id"])
    
    return customer_dim

def create_product_dimension():
    """Create product dimension with category information"""
    logger.info("Creating product dimension...")

    product_df = safe_table_read(os.path.join(silver_dir, 'product'), 'product')
    
    if product_df is None:
        logger.error("Product table not found in Silver layer")
        return None

    # Try to read additional tables for product categories
    product_category_df = safe_table_read(os.path.join(silver_dir, 'productcategory'), 'productcategory')
    product_subcategory_df = safe_table_read(os.path.join(silver_dir, 'productsubcategory'), 'productsubcategory')

    if product_category_df is not None and product_subcategory_df is not None:
        # Join product with category information
        product_dim = product_df.alias("p") \
            .join(product_subcategory_df.alias("psc"),
                  col("p.ProductSubcategoryID") == col("psc.ProductSubcategoryID"), "left") \
            .join(product_category_df.alias("pc"),
                  col("psc.ProductCategoryID") == col("pc.ProductCategoryID"), "left")

        product_dim = product_dim.select(
            col("p.ProductID").alias("product_key"),
            col("p.ProductID").alias("product_id"),
            col("p.Name").alias("product_name"),
            col("p.ProductNumber").alias("product_number"),
            coalesce(col("p.Color"), lit("Unknown")).alias("color"),
            coalesce(col("p.Size"), lit("Unknown")).alias("size"),
            coalesce(col("p.Weight"), lit(0.0)).alias("weight"),
            coalesce(col("p.ListPrice"), lit(0.0)).alias("list_price"),
            coalesce(col("p.StandardCost"), lit(0.0)).alias("standard_cost"),
            coalesce(col("psc.Name"), lit("Unknown")).alias("subcategory_name"),
            coalesce(col("pc.Name"), lit("Unknown")).alias("category_name"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )
    else:
        # Create dimension without category information
        product_dim = product_df.select(
            col("ProductID").alias("product_key"),
            col("ProductID").alias("product_id"),
            col("Name").alias("product_name"),
            col("ProductNumber").alias("product_number"),
            coalesce(col("Color"), lit("Unknown")).alias("color"),
            coalesce(col("Size"), lit("Unknown")).alias("size"),
            coalesce(col("Weight"), lit(0.0)).alias("weight"),
            coalesce(col("ListPrice"), lit(0.0)).alias("list_price"),
            coalesce(col("StandardCost"), lit(0.0)).alias("standard_cost"),
            lit("Unknown").alias("subcategory_name"),
            lit("Unknown").alias("category_name"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )

    # Add surrogate key
    product_dim = generate_surrogate_key(product_dim, ["product_id"])

    return product_dim

def create_geography_dimension():
    """Create geography dimension"""
    logger.info("Creating geography dimension...")

    # Read address data
    address_df = safe_table_read(os.path.join(silver_dir, 'address'), 'address')
    
    if address_df is None:
        logger.error("Address table not found in Silver layer")
        return None

    # Try to read geography reference tables
    territory_df = safe_table_read(os.path.join(silver_dir, 'salesterritory'), 'salesterritory')
    state_df = safe_table_read(os.path.join(silver_dir, 'stateprovince'), 'stateprovince')
    country_df = safe_table_read(os.path.join(silver_dir, 'countryregion'), 'countryregion')

    if all([territory_df is not None, state_df is not None, country_df is not None]):
        geography_dim = address_df.alias("a") \
            .join(state_df.alias("s"), col("a.StateProvinceID") == col("s.StateProvinceID"), "left") \
            .join(country_df.alias("c"), col("s.CountryRegionCode") == col("c.CountryRegionCode"), "left") \
            .join(territory_df.alias("t"), col("s.TerritoryID") == col("t.TerritoryID"), "left")

        geography_dim = geography_dim.select(
            col("a.AddressID").alias("geography_key"),
            col("a.AddressID").alias("address_id"),
            coalesce(col("a.AddressLine1"), lit("Unknown")).alias("address_line1"),
            coalesce(col("a.AddressLine2"), lit("")).alias("address_line2"),
            coalesce(col("a.City"), lit("Unknown")).alias("city"),
            coalesce(col("a.PostalCode"), lit("Unknown")).alias("postal_code"),
            coalesce(col("s.Name"), lit("Unknown")).alias("state_name"),
            coalesce(col("s.StateProvinceCode"), lit("UN")).alias("state_code"),
            coalesce(col("c.Name"), lit("Unknown")).alias("country_name"),
            coalesce(col("c.CountryRegionCode"), lit("UN")).alias("country_code"),
            coalesce(col("t.Name"), lit("Unknown")).alias("territory_name"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )
    else:
        # Create simplified geography dimension
        geography_dim = address_df.select(
            col("AddressID").alias("geography_key"),
            col("AddressID").alias("address_id"),
            coalesce(col("AddressLine1"), lit("Unknown")).alias("address_line1"),
            coalesce(col("AddressLine2"), lit("")).alias("address_line2"),
            coalesce(col("City"), lit("Unknown")).alias("city"),
            coalesce(col("PostalCode"), lit("Unknown")).alias("postal_code"),
            lit("Unknown").alias("state_name"),
            lit("UN").alias("state_code"),
            lit("Unknown").alias("country_name"),
            lit("UN").alias("country_code"),
            lit("Unknown").alias("territory_name"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )

    # Add surrogate key
    geography_dim = generate_surrogate_key(geography_dim, ["address_id"])

    return geography_dim

def create_sales_fact_table():
    """Create main sales fact table"""
    logger.info("Creating sales fact table...")

    # Read Silver layer data
    order_header_df = safe_table_read(os.path.join(silver_dir, 'salesorderheader'), 'salesorderheader')
    order_detail_df = safe_table_read(os.path.join(silver_dir, 'salesorderdetail'), 'salesorderdetail')

    if order_header_df is None or order_detail_df is None:
        logger.error("Required sales tables not found in Silver layer")
        return None

    # Join order header and detail
    sales_fact = order_detail_df.alias("od").join(
        order_header_df.alias("oh"),
        col("od.SalesOrderID") == col("oh.SalesOrderID")
    )

    # Create fact table with measures and foreign keys
    sales_fact = sales_fact.select(
        # Surrogate key
        monotonically_increasing_id().alias("sales_fact_key"),

        # Business keys
        col("oh.SalesOrderID").alias("sales_order_id"),
        col("od.SalesOrderDetailID").alias("sales_order_detail_id"),

        # Foreign keys
        col("oh.CustomerID").alias("customer_key"),
        col("od.ProductID").alias("product_key"),
        coalesce(col("oh.BillToAddressID"), col("oh.ShipToAddressID")).alias("bill_to_geography_key"),
        coalesce(col("oh.ShipToAddressID"), col("oh.BillToAddressID")).alias("ship_to_geography_key"),
        col("oh.OrderDate").alias("order_date_key"),
        coalesce(col("oh.DueDate"), col("oh.OrderDate")).alias("due_date_key"),
        coalesce(col("oh.ShipDate"), col("oh.OrderDate")).alias("ship_date_key"),

        # Measures
        coalesce(col("od.OrderQty"), lit(0)).alias("order_quantity"),
        coalesce(col("od.UnitPrice"), lit(0.0)).alias("unit_price"),
        coalesce(col("od.UnitPriceDiscount"), lit(0.0)).alias("unit_price_discount"),
        coalesce(col("od.LineTotal"), lit(0.0)).alias("line_total"),
        coalesce(col("oh.SubTotal"), lit(0.0)).alias("order_subtotal"),
        coalesce(col("oh.TaxAmt"), lit(0.0)).alias("tax_amount"),
        coalesce(col("oh.Freight"), lit(0.0)).alias("freight"),
        coalesce(col("oh.TotalDue"), lit(0.0)).alias("total_due"),

        # Calculated measures
        (coalesce(col("od.OrderQty"), lit(0)) * coalesce(col("od.UnitPrice"), lit(0.0))).alias("gross_revenue"),
        (coalesce(col("od.LineTotal"), lit(0.0))).alias("net_revenue"),

        # Attributes
        coalesce(col("oh.Status"), lit(0)).alias("order_status"),
        coalesce(col("oh.OnlineOrderFlag"), lit(False)).alias("online_order_flag"),

        # Audit columns
        current_timestamp().alias("created_date"),
        current_timestamp().alias("modified_date")
    )

    return sales_fact

def create_comprehensive_revenue_table():
    """Create comprehensive table with all revenue analysis info merged"""
    logger.info("Creating comprehensive revenue analysis table...")

    # Check if all required tables exist
    required_tables = ['fact_sales', 'dim_customer', 'dim_product', 'dim_geography', 'dim_date']
    missing_tables = []
    
    for table in required_tables:
        table_path = os.path.join(gold_dir, table)
        if not os.path.exists(table_path):
            missing_tables.append(table)
            logger.error(f"Required table {table} not found at {table_path}")
    
    if missing_tables:
        logger.error(f"Cannot create comprehensive revenue table. Missing tables: {missing_tables}")
        return None

    try:
        # Read fact and dimension tables - get current records only for dimensions
        sales_fact = spark.read.parquet(os.path.join(gold_dir, 'fact_sales'))
        logger.info(f"Successfully read sales fact table with {sales_fact.count()} records")
        
        # For SCD tables, only get current records for the comprehensive view
        # Date dimension doesn't have SCD columns, so read it directly
        customer_dim = spark.read.parquet(os.path.join(gold_dir, 'dim_customer')).filter(col("is_current") == True)
        logger.info(f"Successfully read customer dimension with {customer_dim.count()} current records")
        
        product_dim = spark.read.parquet(os.path.join(gold_dir, 'dim_product')).filter(col("is_current") == True)
        logger.info(f"Successfully read product dimension with {product_dim.count()} current records")
        
        geography_dim = spark.read.parquet(os.path.join(gold_dir, 'dim_geography')).filter(col("is_current") == True)
        logger.info(f"Successfully read geography dimension with {geography_dim.count()} current records")
        
        date_dim = spark.read.parquet(os.path.join(gold_dir, 'dim_date'))  # No SCD filter for date dimension
        logger.info(f"Successfully read date dimension with {date_dim.count()} records")
        
    except Exception as e:
        logger.error(f"Error reading dimension tables: {e}")
        return None

    # Create comprehensive table with all dimensions joined
    comprehensive_df = sales_fact.alias("sf") \
        .join(customer_dim.alias("cd"), col("sf.customer_key") == col("cd.customer_key"), "left") \
        .join(product_dim.alias("pd"), col("sf.product_key") == col("pd.product_key"), "left") \
        .join(geography_dim.alias("gd"), col("sf.bill_to_geography_key") == col("gd.geography_key"), "left") \
        .join(date_dim.alias("dd"), col("sf.order_date_key") == col("dd.date_key"), "left")

    # Select all relevant columns for revenue analysis
    revenue_analysis_table = comprehensive_df.select(
        # Sales metrics
        col("sf.sales_fact_key"),
        col("sf.sales_order_id"),
        col("sf.sales_order_detail_id"),
        col("sf.order_quantity"),
        col("sf.unit_price"),
        col("sf.unit_price_discount"),
        col("sf.line_total"),
        col("sf.net_revenue"),
        col("sf.gross_revenue"),
        col("sf.order_subtotal"),
        col("sf.tax_amount"),
        col("sf.freight"),
        col("sf.total_due"),
        col("sf.order_status"),
        
        # Order status description
        when(col("sf.order_status") == 1, "In Process")
        .when(col("sf.order_status") == 2, "Approved")
        .when(col("sf.order_status") == 3, "Backordered")
        .when(col("sf.order_status") == 4, "Rejected")
        .when(col("sf.order_status") == 5, "Shipped")
        .when(col("sf.order_status") == 6, "Cancelled")
        .otherwise("Unknown").alias("order_status_desc"),
        
        col("sf.online_order_flag"),

        # Customer information
        col("cd.customer_id"),
        col("cd.account_number"),
        col("cd.first_name"),
        col("cd.middle_name"),
        col("cd.last_name"),
        col("cd.full_name"),
        col("cd.person_type"),

        # Product information
        col("pd.product_id"),
        col("pd.product_name"),
        col("pd.product_number"),
        col("pd.color"),
        col("pd.size"),
        col("pd.weight"),
        col("pd.list_price"),
        col("pd.standard_cost"),
        col("pd.subcategory_name"),
        col("pd.category_name"),

        # Geography information
        col("gd.address_id"),
        col("gd.address_line1"),
        col("gd.address_line2"),
        col("gd.city"),
        col("gd.postal_code"),
        col("gd.state_name"),
        col("gd.state_code"),
        col("gd.country_name"),
        col("gd.country_code"),
        col("gd.territory_name"),

        # Date information
        col("sf.order_date_key").alias("order_date"),
        col("dd.year").alias("order_year"),
        col("dd.month").alias("order_month"),
        col("dd.quarter").alias("order_quarter"),
        col("dd.month_name"),
        col("dd.day_name"),
        col("dd.year_month"),
        col("dd.year_quarter"),

        # Calculated fields for analysis
        (col("sf.net_revenue") * col("sf.order_quantity")).alias("total_line_revenue"),
        (col("pd.list_price") - col("pd.standard_cost")).alias("profit_margin"),
        current_timestamp().alias("created_date")
    )

    return revenue_analysis_table

def write_table_with_logging(df, table_path, table_name, is_dimension=False, business_keys=None):
    """Write table with logging, error handling, and upsert support"""
    try:
        if df is not None:
            if is_dimension and business_keys:
                # Use SCD Type 2 upsert for dimensions
                final_df = upsert_dimension_table(df, table_path, table_name, business_keys)
            elif not is_dimension and business_keys:
                # Use Type 1 upsert for facts
                final_df = upsert_fact_table(df, table_path, table_name, business_keys)
            else:
                # Default to overwrite for tables without business keys
                final_df = df
            
            record_count = final_df.count()
            final_df.write.mode("overwrite").parquet(table_path)
            logger.info(f"{table_name} written successfully with {record_count} records")
        else:
            logger.error(f"Failed to write {table_name} - DataFrame is None")
    except Exception as e:
        logger.error(f"Error writing {table_name}: {e}")

def main():
    """Main processing function with SCD and upsert capabilities"""
    logger.info("Starting Gold layer processing with SCD and upsert...")

    try:
        # Create Date Dimension (no SCD needed for date dimension typically)
        logger.info("Creating Date Dimension...")
        date_dim = create_date_dimension()
        write_table_with_logging(date_dim, os.path.join(gold_dir, 'dim_date'), 'Date Dimension')

        # Create Customer Dimension with SCD Type 2
        logger.info("Creating Customer Dimension...")
        customer_dim = create_customer_dimension()
        if customer_dim is not None:
            write_table_with_logging(customer_dim, os.path.join(gold_dir, 'dim_customer'), 'Customer Dimension', 
                                   is_dimension=True, business_keys=['customer_id'])
        else:
            logger.error("Failed to create Customer Dimension")

        # Create Product Dimension with SCD Type 2
        logger.info("Creating Product Dimension...")
        product_dim = create_product_dimension()
        if product_dim is not None:
            write_table_with_logging(product_dim, os.path.join(gold_dir, 'dim_product'), 'Product Dimension',
                                   is_dimension=True, business_keys=['product_id'])
        else:
            logger.error("Failed to create Product Dimension")

        # Create Geography Dimension with SCD Type 2
        logger.info("Creating Geography Dimension...")
        geography_dim = create_geography_dimension()
        if geography_dim is not None:
            write_table_with_logging(geography_dim, os.path.join(gold_dir, 'dim_geography'), 'Geography Dimension',
                                   is_dimension=True, business_keys=['address_id'])
        else:
            logger.error("Failed to create Geography Dimension")

        # Create Sales Fact Table with upsert
        logger.info("Creating Sales Fact Table...")
        sales_fact = create_sales_fact_table()
        if sales_fact is not None:
            write_table_with_logging(sales_fact, os.path.join(gold_dir, 'fact_sales'), 'Sales Fact Table',
                                   is_dimension=False, business_keys=['sales_order_id', 'sales_order_detail_id'])
        else:
            logger.error("Failed to create Sales Fact Table")

        # Create Comprehensive Revenue Analysis Table
        logger.info("Checking if all tables exist for comprehensive revenue analysis...")
        required_tables = ['fact_sales', 'dim_customer', 'dim_product', 'dim_geography', 'dim_date']
        tables_exist = all(os.path.exists(os.path.join(gold_dir, table)) for table in required_tables)
        
        if tables_exist:
            logger.info("All required tables exist, creating comprehensive revenue analysis table...")
            revenue_analysis_table = create_comprehensive_revenue_table()
            if revenue_analysis_table is not None:
                write_table_with_logging(revenue_analysis_table, os.path.join(gold_dir, 'revenue_analysis_comprehensive'), 
                                       'Comprehensive Revenue Analysis Table')
            else:
                logger.warning("Failed to create comprehensive revenue analysis table")
        else:
            missing_tables = [table for table in required_tables if not os.path.exists(os.path.join(gold_dir, table))]
            logger.warning(f"Not all required tables exist, skipping comprehensive revenue table creation. Missing: {missing_tables}")

        logger.info("Gold layer processing with SCD and upsert completed successfully!")

        # Display summary statistics
        logger.info("=== GOLD LAYER SUMMARY ===")
        gold_tables = [d for d in os.listdir(gold_dir) if os.path.isdir(os.path.join(gold_dir, d))]
        for table in gold_tables:
            try:
                df = spark.read.parquet(os.path.join(gold_dir, table))
                
                # Show SCD info for dimension tables
                if table.startswith('dim_') and 'is_current' in df.columns:
                    current_count = df.filter(col("is_current") == True).count()
                    total_count = df.count()
                    logger.info(f"{table}: {total_count} total records ({current_count} current, {total_count - current_count} historical), {len(df.columns)} columns")
                else:
                    logger.info(f"{table}: {df.count()} records, {len(df.columns)} columns")
                    
            except Exception as e:
                logger.error(f"Error reading {table}: {e}")

    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("🔥 Unhandled exception occurred in main()", exc_info=True)
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        spark.stop()
        logger.info("SparkSession stopped.")