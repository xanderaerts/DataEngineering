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
from datetime import datetime, date
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
silver_dir = 'data/Silver'
gold_dir = 'data/Gold'
plots_dir = 'plots'

# Define path for the error log file
error_log_dir = 'logs'
error_log_file_path = os.path.join(error_log_dir, 'gold_layer_errors.log')

# Ensure directories exist
os.makedirs(gold_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
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

logger.info("Starting PySpark Data Layer Processing (Gold Layer)...")

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
        current_date = date(current_date.year + (1 if current_date.month == 12 else 0),
                          (current_date.month % 12) + 1 if current_date.month != 12 else 1,
                          1) if current_date.day == 1 else date(current_date.year, current_date.month, current_date.day + 1)
    
    # Create DataFrame with date range
    from datetime import timedelta
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

def implement_scd_type2(existing_df, new_df, business_keys, compare_columns):
    """Implement SCD Type 2 for slowly changing dimensions"""
    logger.info("Implementing SCD Type 2...")
    
    if existing_df is None:
        # First load - add SCD columns
        return new_df.withColumn("effective_start_date", current_timestamp()) \
                    .withColumn("effective_end_date", lit(None).cast(TimestampType())) \
                    .withColumn("is_current", lit(True)) \
                    .withColumn("version", lit(1))
    
    # Join existing and new data on business keys
    join_condition = [col(f"existing.{key}") == col(f"new.{key}") for key in business_keys]
    joined_df = existing_df.alias("existing").join(new_df.alias("new"), 
                                                  join_condition, "full_outer")
    
    # Identify changes
    change_conditions = []
    for col_name in compare_columns:
        change_conditions.append(
            col(f"existing.{col_name}") != col(f"new.{col_name}")
        )
    
    # Create result DataFrame with SCD logic
    # This is a simplified version - in production, you'd want more sophisticated logic
    result_df = joined_df.select(
        *[coalesce(col(f"new.{c}"), col(f"existing.{c}")).alias(c) for c in new_df.columns],
        when(col("existing.is_current").isNull(), current_timestamp())
        .otherwise(col("existing.effective_start_date")).alias("effective_start_date"),
        lit(None).cast(TimestampType()).alias("effective_end_date"),
        lit(True).alias("is_current"),
        when(col("existing.version").isNull(), lit(1))
        .otherwise(col("existing.version") + 1).alias("version")
    )
    
    return result_df

def upsert_to_gold(df, table_path, business_keys):
    """Implement upsert functionality for Gold layer tables"""
    logger.info(f"Upserting to Gold table: {table_path}")
    
    try:
        # Check if table exists
        if os.path.exists(table_path):
            existing_df = spark.read.parquet(table_path)
            
            # Perform merge logic (simplified upsert)
            # In production, you'd use Delta Lake for proper MERGE operations
            join_condition = [col(f"existing.{key}") == col(f"new.{key}") for key in business_keys]
            
            # Get records that don't exist in target
            new_records = df.alias("new").join(
                existing_df.alias("existing").select(*business_keys), 
                [col(f"new.{key}") == col(f"existing.{key}") for key in business_keys], 
                "left_anti"
            )
            
            # Union existing records with new records
            result_df = existing_df.unionByName(new_records, allowMissingColumns=True)
        else:
            result_df = df
        
        # Write to Gold layer
        result_df.write.mode("overwrite").parquet(table_path)
        logger.info(f"Successfully upserted to {table_path}")
        
    except Exception as e:
        logger.error(f"Error during upsert to {table_path}: {e}")
        # Fallback to simple overwrite
        df.write.mode("overwrite").parquet(table_path)

# --- Gold Layer Dimension and Fact Table Creation ---

def create_customer_dimension():
    """Create customer dimension with SCD Type 2"""
    logger.info("Creating customer dimension...")
    
    # Read Silver layer data
    customer_df = spark.read.parquet(os.path.join(silver_dir, 'customer'))
    person_df = spark.read.parquet(os.path.join(silver_dir, 'person'))
    
    # Join customer with person data
    customer_dim = customer_df.join(
        person_df, 
        customer_df.PersonID == person_df.BusinessEntityID, 
        "left"
    ).select(
        customer_df.CustomerID.alias("customer_key"),
        customer_df.CustomerID.alias("customer_id"),
        customer_df.AccountNumber.alias("account_number"),
        person_df.FirstName.alias("first_name"),
        person_df.MiddleName.alias("middle_name"),
        person_df.LastName.alias("last_name"),
        concat(coalesce(person_df.FirstName, lit("")), lit(" "), 
               coalesce(person_df.LastName, lit(""))).alias("full_name"),
        person_df.PersonType.alias("person_type"),
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
    
    product_df = spark.read.parquet(os.path.join(silver_dir, 'product'))
    
    # Read additional tables if they exist for product categories
    try:
        product_category_df = spark.read.parquet(os.path.join(silver_dir, 'productcategory'))
        product_subcategory_df = spark.read.parquet(os.path.join(silver_dir, 'productsubcategory'))
        
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
            col("p.Color").alias("color"),
            col("p.Size").alias("size"),
            col("p.Weight").alias("weight"),
            col("p.ListPrice").alias("list_price"),
            col("p.StandardCost").alias("standard_cost"),
            col("psc.Name").alias("subcategory_name"),
            col("pc.Name").alias("category_name"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )
        
    except Exception as e:
        logger.warning(f"Could not find product category tables: {e}")
        # Create dimension without category information
        product_dim = product_df.select(
            col("ProductID").alias("product_key"),
            col("ProductID").alias("product_id"),
            col("Name").alias("product_name"),
            col("ProductNumber").alias("product_number"),
            col("Color").alias("color"),
            col("Size").alias("size"),
            col("Weight").alias("weight"),
            col("ListPrice").alias("list_price"),
            col("StandardCost").alias("standard_cost"),
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
    
    # Read address and territory data
    address_df = spark.read.parquet(os.path.join(silver_dir, 'address'))
    
    try:
        # Try to read territory and country/state data if available
        territory_df = spark.read.parquet(os.path.join(silver_dir, 'salesterritory'))
        state_df = spark.read.parquet(os.path.join(silver_dir, 'stateprovince'))
        country_df = spark.read.parquet(os.path.join(silver_dir, 'countryregion'))
        
        geography_dim = address_df.alias("a") \
            .join(state_df.alias("s"), col("a.StateProvinceID") == col("s.StateProvinceID"), "left") \
            .join(country_df.alias("c"), col("s.CountryRegionCode") == col("c.CountryRegionCode"), "left") \
            .join(territory_df.alias("t"), col("s.TerritoryID") == col("t.TerritoryID"), "left")
        
        geography_dim = geography_dim.select(
            col("a.AddressID").alias("geography_key"),
            col("a.AddressID").alias("address_id"),
            col("a.AddressLine1").alias("address_line1"),
            col("a.AddressLine2").alias("address_line2"),
            col("a.City").alias("city"),
            col("a.PostalCode").alias("postal_code"),
            col("s.Name").alias("state_name"),
            col("s.StateProvinceCode").alias("state_code"),
            col("c.Name").alias("country_name"),
            col("c.CountryRegionCode").alias("country_code"),
            col("t.Name").alias("territory_name"),
            current_timestamp().alias("created_date"),
            current_timestamp().alias("modified_date")
        )
        
    except Exception as e:
        logger.warning(f"Could not find geography reference tables: {e}")
        # Create simplified geography dimension
        geography_dim = address_df.select(
            col("AddressID").alias("geography_key"),
            col("AddressID").alias("address_id"),
            col("AddressLine1").alias("address_line1"),
            col("AddressLine2").alias("address_line2"),
            col("City").alias("city"),
            col("PostalCode").alias("postal_code"),
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
    order_header_df = spark.read.parquet(os.path.join(silver_dir, 'salesorderheader'))
    order_detail_df = spark.read.parquet(os.path.join(silver_dir, 'salesorderdetail'))
    
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
        
        # Foreign keys (will be updated with dimension surrogate keys)
        col("oh.CustomerID").alias("customer_key"),
        col("od.ProductID").alias("product_key"),
        col("oh.BillToAddressID").alias("bill_to_geography_key"),
        col("oh.ShipToAddressID").alias("ship_to_geography_key"),
        col("oh.OrderDate").alias("order_date_key"),
        col("oh.DueDate").alias("due_date_key"),
        col("oh.ShipDate").alias("ship_date_key"),
        
        # Measures
        col("od.OrderQty").alias("order_quantity"),
        col("od.UnitPrice").alias("unit_price"),
        col("od.UnitPriceDiscount").alias("unit_price_discount"),
        col("od.LineTotal").alias("line_total"),
        col("oh.SubTotal").alias("order_subtotal"),
        col("oh.TaxAmt").alias("tax_amount"),
        col("oh.Freight").alias("freight"),
        col("oh.TotalDue").alias("total_due"),
        
        # Attributes
        col("oh.Status").alias("order_status"),
        col("oh.OnlineOrderFlag").alias("online_order_flag"),
        
        # Audit columns
        current_timestamp().alias("created_date"),
        current_timestamp().alias("modified_date")
    )
    
    # Filter for current year (2014 based on typical AdventureWorks data)
    current_year = 2014  # Adjust based on your data
    sales_fact = sales_fact.filter(year("order_date_key") == current_year)
    
    return sales_fact

# --- Main Processing Logic ---

def main():
    """Main processing function"""
    logger.info("Starting Gold layer processing...")
    
    # Create Date Dimension
    date_dim = create_date_dimension()
    upsert_to_gold(date_dim, os.path.join(gold_dir, 'dim_date'), ["date_key"])
    logger.info(f"Date dimension created with {date_dim.count()} records")
    
    # Create Customer Dimension
    customer_dim = create_customer_dimension()
    upsert_to_gold(customer_dim, os.path.join(gold_dir, 'dim_customer'), ["customer_id"])
    logger.info(f"Customer dimension created with {customer_dim.count()} records")
    
    # Create Product Dimension
    product_dim = create_product_dimension()
    upsert_to_gold(product_dim, os.path.join(gold_dir, 'dim_product'), ["product_id"])
    logger.info(f"Product dimension created with {product_dim.count()} records")
    
    # Create Geography Dimension
    geography_dim = create_geography_dimension()
    upsert_to_gold(geography_dim, os.path.join(gold_dir, 'dim_geography'), ["address_id"])
    logger.info(f"Geography dimension created with {geography_dim.count()} records")
    
    # Create Sales Fact Table
    sales_fact = create_sales_fact_table()
    upsert_to_gold(sales_fact, os.path.join(gold_dir, 'fact_sales'), ["sales_order_id", "sales_order_detail_id"])
    logger.info(f"Sales fact table created with {sales_fact.count()} records")
    
    # Generate required plots
    generate_plots(sales_fact, product_dim, customer_dim, geography_dim)
    
    logger.info("Gold layer processing completed successfully!")

def generate_plots(sales_fact, product_dim, customer_dim, geography_dim):
    """Generate all required plots for the assignment"""
    logger.info("Generating plots...")
    
    # 1. Revenue by Category (all)
    try:
        category_revenue = sales_fact.join(product_dim, "product_key") \
            .groupBy("category_name") \
            .agg(spark_sum("line_total").alias("revenue")) \
            .orderBy(desc("revenue"))
        
        category_revenue_pd = category_revenue.toPandas()
        
        plt.figure(figsize=(12, 8))
        plt.bar(category_revenue_pd['category_name'], category_revenue_pd['revenue'])
        plt.title('Revenue by Category')
        plt.xlabel('Category')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'revenue_by_category.png'))
        plt.close()
        
        logger.info("Generated: Revenue by Category plot")
    except Exception as e:
        logger.error(f"Error generating category revenue plot: {e}")
    
    # 2. Top-10 Subcategories
    try:
        subcategory_revenue = sales_fact.join(product_dim, "product_key") \
            .groupBy("subcategory_name") \
            .agg(spark_sum("line_total").alias("revenue")) \
            .orderBy(desc("revenue")) \
            .limit(10)
        
        subcategory_revenue_pd = subcategory_revenue.toPandas()
        
        plt.figure(figsize=(12, 8))
        plt.bar(subcategory_revenue_pd['subcategory_name'], subcategory_revenue_pd['revenue'])
        plt.title('Top 10 Subcategories by Revenue')
        plt.xlabel('Subcategory')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top10_subcategories.png'))
        plt.close()
        
        logger.info("Generated: Top 10 Subcategories plot")
    except Exception as e:
        logger.error(f"Error generating subcategory revenue plot: {e}")
    
    # 3. Top-10 Customers
    try:
        customer_revenue = sales_fact.join(customer_dim, "customer_key") \
            .groupBy("customer_id", "full_name") \
            .agg(spark_sum("line_total").alias("revenue")) \
            .orderBy(desc("revenue")) \
            .limit(10)
        
        customer_revenue_pd = customer_revenue.toPandas()
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(customer_revenue_pd)), customer_revenue_pd['revenue'])
        plt.title('Top 10 Customers by Revenue')
        plt.xlabel('Customer Rank')
        plt.ylabel('Revenue')
        plt.xticks(range(len(customer_revenue_pd)), [f"Customer {i+1}" for i in range(len(customer_revenue_pd))])
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top10_customers.png'))
        plt.close()
        
        logger.info("Generated: Top 10 Customers plot")
    except Exception as e:
        logger.error(f"Error generating customer revenue plot: {e}")
    
    # 4. Revenue by Order Status (all)
    try:
        status_revenue = sales_fact.groupBy("order_status") \
            .agg(spark_sum("line_total").alias("revenue")) \
            .orderBy(desc("revenue"))
        
        status_revenue_pd = status_revenue.toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.pie(status_revenue_pd['revenue'], labels=status_revenue_pd['order_status'], autopct='%1.1f%%')
        plt.title('Revenue by Order Status')
        plt.savefig(os.path.join(plots_dir, 'revenue_by_order_status.png'))
        plt.close()
        
        logger.info("Generated: Revenue by Order Status plot")
    except Exception as e:
        logger.error(f"Error generating order status revenue plot: {e}")
    
    # 5. Top-10 Countries revenue
    try:
        country_revenue = sales_fact.join(geography_dim, 
                                        sales_fact.bill_to_geography_key == geography_dim.geography_key) \
            .groupBy("country_name") \
            .agg(spark_sum("line_total").alias("revenue")) \
            .orderBy(desc("revenue")) \
            .limit(10)
        
        country_revenue_pd = country_revenue.toPandas()
        
        plt.figure(figsize=(12, 8))
        plt.bar(country_revenue_pd['country_name'], country_revenue_pd['revenue'])
        plt.title('Top 10 Countries by Revenue')
        plt.xlabel('Country')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top10_countries.png'))
        plt.close()
        
        logger.info("Generated: Top 10 Countries plot")
    except Exception as e:
        logger.error(f"Error generating country revenue plot: {e}")
    
    # 6. Top-10 States revenue
    try:
        state_revenue = sales_fact.join(geography_dim, 
                                      sales_fact.bill_to_geography_key == geography_dim.geography_key) \
            .groupBy("state_name") \
            .agg(spark_sum("line_total").alias("revenue")) \
            .orderBy(desc("revenue")) \
            .limit(10)
        
        state_revenue_pd = state_revenue.toPandas()
        
        plt.figure(figsize=(12, 8))
        plt.bar(state_revenue_pd['state_name'], state_revenue_pd['revenue'])
        plt.title('Top 10 States by Revenue')
        plt.xlabel('State')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top10_states.png'))
        plt.close()
        
        logger.info("Generated: Top 10 States plot")
    except Exception as e:
        logger.error(f"Error generating state revenue plot: {e}")
    
    # 7. Top-10 Cities revenue
    try:
        city_revenue = sales_fact.join(geography_dim, 
                                     sales_fact.bill_to_geography_key == geography_dim.geography_key) \
            .groupBy("city") \
            .agg(spark_sum("line_total").alias("revenue")) \
            .orderBy(desc("revenue")) \
            .limit(10)
        
        city_revenue_pd = city_revenue.toPandas()
        
        plt.figure(figsize=(12, 8))
        plt.bar(city_revenue_pd['city'], city_revenue_pd['revenue'])
        plt.title('Top 10 Cities by Revenue')
        plt.xlabel('City')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top10_cities.png'))
        plt.close()
        
        logger.info("Generated: Top 10 Cities plot")
    except Exception as e:
        logger.error(f"Error generating city revenue plot: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("🔥 Unhandled exception occurred in main()", exc_info=True)
        # Optional: print traceback to console too
        import traceback
        traceback.print_exc()
        # Exit with error code so you still know something broke
        exit(1)
    finally:
        # Make sure Spark gets stopped no matter what
        spark.stop()
        logger.info("SparkSession stopped.")


