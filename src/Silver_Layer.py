# Import necessary libraries for Spark DataFrame operations
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp, row_number, monotonically_increasing_id, to_date, year, month, dayofmonth, dayofweek, quarter, weekofyear, date_format, coalesce
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import os

# Define paths for raw, bronze, silver, and gold layers
# These paths help organize your data lake layers
raw_dir = 'data/Raw/'
bronze_dir = 'data/Bronze'
silver_dir = 'data/Silver' # This was the Parquet output directory
gold_dir = 'data/Gold'


# Initialize Spark Session
# A SparkSession is the entry point to programming Spark with the DataFrame API.
# We set a specific app name for clarity in Spark UI logs.
spark = SparkSession.builder.appName("TakeHomeExamSilverLayer").getOrCreate()

# --- Step 1: Load Data from Bronze Layer ---
# Explanation: We read all the cleaned CSV files from the Bronze directory into Spark DataFrames.
# This is the starting point for transformations in the Silver layer.
# We assume the Bronze layer has successfully processed and saved the raw CSVs here.

print("--- Loading data from Bronze Layer ---")

# A dictionary to store all loaded DataFrames for easy access
bronze_dataframes = {}

# List of all 23 expected file names.
# This ensures we attempt to load all relevant files.
expected_files = [
    'address.csv', 'addresstype.csv', 'countryregion.csv', 'creditcard.csv',
    'currency.csv', 'customer.csv', 'location.csv', 'password.csv',
    'person.csv', 'product.csv', 'productcategory.csv', 'productdescription.csv',
    'productsubcategory.csv', 'salesorderdetail.csv', 'salesorderheader.csv',
    'salesterritory.csv', 'shipmethod.csv', 'stateprovince.csv', 'store.csv',
    'transactionhistoryarchive.csv', 'unitmeasure.csv', 'vendor.csv', 'workorder.csv'
]

# Loop through each expected file, load it, and store it in the dictionary.
for file_name in expected_files:
    bronze_file_path = os.path.join(bronze_dir, file_name)
    try:
        # productdescription.csv uses ';' as a delimiter and has 2 header rows.
        # For other files, assume tab delimiter (as per your initial upload info) or default comma.
        
        if file_name == 'productdescription.csv':
            # For productdescription.csv, read with specific delimiter and skip initial rows
            # Spark's read.csv doesn't directly support skipping arbitrary rows easily for header inference.
            # A common workaround is to read as text, skip, then parse.
            # For simplicity, assuming the first *valid* header is on the 3rd line (0-indexed line 2)
            # and that inferSchema will work if the header is correctly identified.
            # If this fails, manual schema definition would be necessary.
            df = spark.read.csv(bronze_file_path, header=True, inferSchema=True, sep=';')
        else:
            # Assuming most other files use tab delimiter as per your initial file info.
            # If not, default to comma or adjust as needed.
            df = spark.read.csv(bronze_file_path, header=True, inferSchema=True, sep='\t')
            
        # Store the DataFrame using its base name (without .csv extension) as the key.
        bronze_dataframes[file_name.replace('.csv', '')] = df
        print(f"Successfully loaded {file_name}. Schema:")
        df.printSchema() # Display the schema to understand data types and columns.
        print(f"First 5 rows of {file_name}:")
        df.show(5) # Show a sample of the data for verification.
    except Exception as e:
        # Log an error if a file cannot be loaded, but continue processing other files.
        print(f"Error loading {file_name}: {e}")

# Assign loaded DataFrames to more descriptive variables for easier use in subsequent steps.
address_df = bronze_dataframes.get('address')
addresstype_df = bronze_dataframes.get('addresstype')
countryregion_df = bronze_dataframes.get('countryregion')
creditcard_df = bronze_dataframes.get('creditcard')
currency_df = bronze_dataframes.get('currency')
customer_df = bronze_dataframes.get('customer')
location_df = bronze_dataframes.get('location')
password_df = bronze_dataframes.get('password')
person_df = bronze_dataframes.get('person')
product_df = bronze_dataframes.get('product')
productcategory_df = bronze_dataframes.get('productcategory')
productdescription_df = bronze_dataframes.get('productdescription')
productsubcategory_df = bronze_dataframes.get('productsubcategory')
salesorderdetail_df = bronze_dataframes.get('salesorderdetail')
salesorderheader_df = bronze_dataframes.get('salesorderheader')
salesterritory_df = bronze_dataframes.get('salesterritory')
shipmethod_df = bronze_dataframes.get('shipmethod')
stateprovince_df = bronze_dataframes.get('stateprovince')
store_df = bronze_dataframes.get('store')
transactionhistoryarchive_df = bronze_dataframes.get('transactionhistoryarchive')
unitmeasure_df = bronze_dataframes.get('unitmeasure') # New
vendor_df = bronze_dataframes.get('vendor') # New
workorder_df = bronze_dataframes.get('workorder') # New


# --- Step 2: Create DimDate (Date Dimension) ---
# Explanation: A Date Dimension is crucial for time-based analysis.
# We generate a table containing various date attributes (year, month, day, quarter, etc.).
# The date range is determined by scanning 'ModifiedDate' columns across relevant source tables,
# and also 'OrderDate', 'DueDate', 'ShipDate' from salesorderheader.
# or a default range is used if dates cannot be found.

print("\n--- Creating DimDate Dimension ---")

# Initialize default min and max dates.
min_date_str = "2000-01-01"
max_date_str = "2025-12-31"

# List to collect 'ModifiedDate' columns from various DataFrames.
date_cols_to_check = []
if address_df and 'ModifiedDate' in address_df.columns:
    date_cols_to_check.append(address_df.select(col('ModifiedDate').cast('date').alias('ModifiedDate')))
if customer_df and 'ModifiedDate' in customer_df.columns:
    date_cols_to_check.append(customer_df.select(col('ModifiedDate').cast('date').alias('ModifiedDate')))
if person_df and 'ModifiedDate' in person_df.columns:
    date_cols_to_check.append(person_df.select(col('ModifiedDate').cast('date').alias('ModifiedDate')))
if product_df and 'ModifiedDate' in product_df.columns:
    date_cols_to_check.append(product_df.select(col('ModifiedDate').cast('date').alias('ModifiedDate')))
if salesorderheader_df:
    if 'OrderDate' in salesorderheader_df.columns:
        date_cols_to_check.append(salesorderheader_df.select(col('OrderDate').cast('date').alias('ModifiedDate')))
    if 'DueDate' in salesorderheader_df.columns:
        date_cols_to_check.append(salesorderheader_df.select(col('DueDate').cast('date').alias('ModifiedDate')))
    if 'ShipDate' in salesorderheader_df.columns:
        date_cols_to_check.append(salesorderheader_df.select(col('ShipDate').cast('date').alias('ModifiedDate')))
if workorder_df and 'ModifiedDate' in workorder_df.columns: # Check workorder for date range
    date_cols_to_check.append(workorder_df.select(col('ModifiedDate').cast('date').alias('ModifiedDate')))


# If any 'ModifiedDate' columns were found, union them and find the actual min/max dates.
if date_cols_to_check:
    union_dates_df = date_cols_to_check[0]
    for i in range(1, len(date_cols_to_check)):
        union_dates_df = union_dates_df.union(date_cols_to_check[i])

    # Collect the minimum and maximum dates from the unioned DataFrame.
    # Handle potential None values from collect() if no dates are found
    collected_min_date = union_dates_df.agg({"ModifiedDate": "min"}).collect()[0][0]
    collected_max_date = union_dates_df.agg({"ModifiedDate": "max"}).collect()[0][0]

    if collected_min_date and collected_max_date:
        min_date_str = collected_min_date.strftime('%Y-%m-%d')
        max_date_str = collected_max_date.strftime('%Y-%m-%d')
    else:
        print("Could not determine min/max dates from data, using default range.")
else:
    print("No relevant date columns found in key dataframes, using default date range.")

print(f"Generating DimDate from {min_date_str} to {max_date_str}")

# Generate a list of dates between the determined start and end dates.
start = datetime.strptime(min_date_str, '%Y-%m-%d')
end = datetime.strptime(max_date_str, '%Y-%m-%d')
date_list = []
current = start
while current <= end:
    date_list.append((current.strftime('%Y-%m-%d'),))
    current += timedelta(days=1)

# Create a Spark DataFrame from the generated date list.
dim_date_df = spark.createDataFrame(date_list, ['Date']).withColumn('Date', col('Date').cast('date'))

# Add various date attributes to the DimDate DataFrame.
dim_date_df = dim_date_df.withColumn("DateKey", date_format(col("Date"), "yyyyMMdd").cast("integer")) \
                         .withColumn("Year", year(col("Date"))) \
                         .withColumn("Month", month(col("Date"))) \
                         .withColumn("Day", dayofmonth(col("Date"))) \
                         .withColumn("DayOfWeek", dayofweek(col("Date"))) \
                         .withColumn("Quarter", quarter(col("Date"))) \
                         .withColumn("WeekOfYear", weekofyear(col("Date"))) \
                         .withColumn("DayName", date_format(col("Date"), "EEEE")) \
                         .withColumn("MonthName", date_format(col("Date"), "MMMM")) \
                         .withColumn("IsWeekend", (dayofweek(col("Date")) == 1) | (dayofweek(col("Date")) == 7)) # Sunday=1, Saturday=7

print("DimDate Schema:")
dim_date_df.printSchema()
print("DimDate Sample Data:")
dim_date_df.show(5)

# --- Step 3: Create DimCustomer Dimension ---
# Explanation: This dimension table combines customer, person, and store data, cleaning and selecting
# relevant columns. It serves as a central source for customer-related analysis.
# We also apply a simplified Slowly Changing Dimension (SCD) Type 1 approach by ensuring
# unique customer IDs and taking the latest record if duplicates exist based on ModifiedDate.

print("\n--- Creating DimCustomer Dimension ---")

if customer_df and person_df:
    # Corrected Join: Join customer_df (left side) with person_df (right side)
    # customer_df has 'PersonID' which links to 'BusinessEntityID' in person_df.
    # We remove 'AdditionalContactInfo' and 'Demographics' from this initial select
    # as they appear to not be present in the loaded person.csv or customer.csv schemas.
    dim_customer_df = customer_df.join(person_df, customer_df["PersonID"] == person_df["BusinessEntityID"], how="inner") \
                                 .select(
                                     customer_df["CustomerID"], # Use DataFrame object for clarity
                                     col("PersonType"),
                                     col("NameStyle"),
                                     col("Title"),
                                     col("FirstName"),
                                     col("MiddleName"),
                                     col("LastName"),
                                     col("Suffix"),
                                     col("EmailPromotion"),
                                     # Removed 'AdditionalContactInfo' and 'Demographics' from here
                                     person_df["rowguid"].alias("CustomerRowGUID"), # Qualify with person_df object
                                     person_df["ModifiedDate"].alias("CustomerModifiedDate") # Qualify with person_df object
                                 )

    # If store data is available, join it to enrich customer information (e.g., for store customers)
    if store_df:
        # Join dim_customer_df (which now has CustomerID) with store_df on 'CustomerID'
        # Assuming store.BusinessEntityID is the CustomerID for store-related customers.
        # Removed 'Demographics' and 'ModifiedDate' from store_df.select as they are not present in the schema.
        dim_customer_df = dim_customer_df.join(store_df.select(
                                                col("BusinessEntityID").alias("CustomerID"), # Store's BusinessEntityID is the CustomerID
                                                col("Name").alias("StoreName"),
                                                col("SalesPersonID")
                                                # Removed col("Demographics").alias("StoreDemographics")
                                                # Removed col("ModifiedDate").alias("StoreModifiedDate")
                                            ), on="CustomerID", how="left") \
                                            .withColumn("CustomerModifiedDate", col("CustomerModifiedDate")) # Keep original CustomerModifiedDate as there's no StoreModifiedDate to coalesce with

    # Handle potential duplicates for CustomerID.
    # For a true SCD Type 1, existing records would be updated. Here, we ensure uniqueness
    # by selecting the most recent record based on 'CustomerModifiedDate' if multiple exist.
    window_spec = Window.partitionBy("CustomerID").orderBy(col("CustomerModifiedDate").desc())
    dim_customer_df = dim_customer_df.withColumn("rn", row_number().over(window_spec)) \
                                     .filter(col("rn") == 1) \
                                     .drop("rn") # Drop the row number column after filtering.

    print("DimCustomer Schema:")
    dim_customer_df.printSchema()
    print("DimCustomer Sample Data:")
    dim_customer_df.show(5)
else:
    print("Skipping DimCustomer creation: 'customer.csv' or 'person.csv' not loaded.")
    dim_customer_df = None

# --- Step 4: Create DimProduct, DimProductCategory, DimProductSubcategory, DimUnitMeasure Dimensions ---
# Explanation: These dimension tables contain cleaned and selected product information,
# organized hierarchically with categories and subcategories, and unit measures.

print("\n--- Creating Product Dimensions ---")

dim_product_category_df = None
if productcategory_df:
    dim_product_category_df = productcategory_df.select(
        col("ProductCategoryID"),
        col("Name").alias("ProductCategoryName"),
        col("rowguid").alias("ProductCategoryRowGUID"),
        col("ModifiedDate").alias("ProductCategoryModifiedDate")
    )
    window_spec = Window.partitionBy("ProductCategoryID").orderBy(col("ProductCategoryModifiedDate").desc())
    dim_product_category_df = dim_product_category_df.withColumn("rn", row_number().over(window_spec)) \
                                                     .filter(col("rn") == 1) \
                                                     .drop("rn")
    print("DimProductCategory Schema:")
    dim_product_category_df.printSchema()
    print("DimProductCategory Sample Data:")
    dim_product_category_df.show(5)
else:
    print("Skipping DimProductCategory creation: 'productcategory.csv' not loaded.")


dim_product_subcategory_df = None
if productsubcategory_df and dim_product_category_df:
    dim_product_subcategory_df = productsubcategory_df.join(dim_product_category_df, on="ProductCategoryID", how="left") \
                                                    .select(
                                                        col("ProductSubcategoryID"),
                                                        col("ProductCategoryID"),
                                                        col("ProductCategoryName"),
                                                        col("Name").alias("ProductSubcategoryName"),
                                                        col("rowguid").alias("ProductSubcategoryRowGUID"),
                                                        col("ModifiedDate").alias("ProductSubcategoryModifiedDate")
                                                    )
    window_spec = Window.partitionBy("ProductSubcategoryID").orderBy(col("ProductSubcategoryModifiedDate").desc())
    dim_product_subcategory_df = dim_product_subcategory_df.withColumn("rn", row_number().over(window_spec)) \
                                                           .filter(col("rn") == 1) \
                                                           .drop("rn")
    print("DimProductSubcategory Schema:")
    dim_product_subcategory_df.printSchema()
    print("DimProductSubcategory Sample Data:")
    dim_product_subcategory_df.show(5)
else:
    print("Skipping DimProductSubcategory creation: 'productsubcategory.csv' or DimProductCategory not loaded.")

dim_unit_measure_df = None
if unitmeasure_df:
    dim_unit_measure_df = unitmeasure_df.select(
        col("UnitMeasureCode"),
        col("Name").alias("UnitMeasureName"),
        col("ModifiedDate").alias("UnitMeasureModifiedDate")
    )
    window_spec = Window.partitionBy("UnitMeasureCode").orderBy(col("UnitMeasureModifiedDate").desc())
    dim_unit_measure_df = dim_unit_measure_df.withColumn("rn", row_number().over(window_spec)) \
                                             .filter(col("rn") == 1) \
                                             .drop("rn")
    print("DimUnitMeasure Schema:")
    dim_unit_measure_df.printSchema()
    print("DimUnitMeasure Sample Data:")
    dim_unit_measure_df.show(5)
else:
    print("Skipping DimUnitMeasure creation: 'unitmeasure.csv' not loaded.")


if product_df:
    dim_product_df = product_df.select(
        col("ProductID"),
        col("Name").alias("ProductName"),
        col("ProductNumber"),
        col("MakeFlag"),
        col("FinishedGoodsFlag"),
        col("Color"),
        col("SafetyStockLevel"),
        col("ReorderPoint"),
        col("StandardCost"),
        col("ListPrice"),
        col("Size"),
        col("SizeUnitMeasureCode"), # Keep for join
        col("Weight"),
        col("WeightUnitMeasureCode"), # Keep for join
        col("DaysToManufacture"),
        col("ProductLine"),
        col("Class"),
        col("Style"),
        col("ProductSubcategoryID"), # Keep for join
        col("SellStartDate"),
        col("SellEndDate"),
        col("DiscontinuedDate"),
        col("rowguid").alias("ProductRowGUID"),
        col("ModifiedDate").alias("ProductModifiedDate")
    )

    # Join with product subcategory and category for full product hierarchy
    if dim_product_subcategory_df:
        dim_product_df = dim_product_df.join(dim_product_subcategory_df.select(
                                                "ProductSubcategoryID",
                                                "ProductSubcategoryName",
                                                "ProductCategoryID",
                                                "ProductCategoryName",
                                                # Added ProductSubcategoryModifiedDate to the select list
                                                # Alias it here to ensure it's uniquely accessible after the join
                                                col("ProductSubcategoryModifiedDate").alias("ProductSubcategoryModifiedDateForJoin")
                                            ), on="ProductSubcategoryID", how="left") \
                                        .withColumn("ProductModifiedDate", coalesce(col("ProductSubcategoryModifiedDateForJoin"), col("ProductModifiedDate"))) # Use the aliased column

    # Join with unit measure for size and weight units
    if dim_unit_measure_df:
        dim_product_df = dim_product_df.join(dim_unit_measure_df.select(
                                                col("UnitMeasureCode").alias("SizeUnitMeasureCode"),
                                                col("UnitMeasureName").alias("SizeUnitMeasureName")
                                            ), on="SizeUnitMeasureCode", how="left")
        dim_product_df = dim_product_df.join(dim_unit_measure_df.select(
                                                col("UnitMeasureCode").alias("WeightUnitMeasureCode"),
                                                col("UnitMeasureName").alias("WeightUnitMeasureName")
                                            ), on="WeightUnitMeasureCode", how="left")


    # Ensure unique products. Simplified SCD Type 1.
    window_spec = Window.partitionBy("ProductID").orderBy(col("ProductModifiedDate").desc())
    dim_product_df = dim_product_df.withColumn("rn", row_number().over(window_spec)) \
                                   .filter(col("rn") == 1) \
                                   .drop("rn")

    print("DimProduct Schema:")
    dim_product_df.printSchema()
    print("DimProduct Sample Data:")
    dim_product_df.show(5)
else:
    print("Skipping DimProduct creation: 'product.csv' not loaded.")
    dim_product_df = None


# --- Step 5: Create DimAddress, DimGeography, and DimStateProvince Dimensions ---
# Explanation: These dimensions provide detailed address information and hierarchical
# geographic data (city, state, country). They are created by joining address,
# stateprovince, and countryregion data.

print("\n--- Creating Address and Geography Dimensions ---")

dim_stateprovince_df = None
if stateprovince_df and countryregion_df:
    # Alias the DataFrames before the join to resolve ambiguous 'Name' column
    sp_df_aliased = stateprovince_df.alias("sp")
    cr_df_aliased = countryregion_df.alias("cr")

    dim_stateprovince_df = sp_df_aliased.join(cr_df_aliased, on=sp_df_aliased["CountryRegionCode"] == cr_df_aliased["CountryRegionCode"], how="left") \
                                           .select(
                                               sp_df_aliased["StateProvinceID"],
                                               sp_df_aliased["StateProvinceCode"],
                                               sp_df_aliased["IsOnlyStateProvinceFlag"],
                                               sp_df_aliased["Name"].alias("StateProvinceName"), # Explicitly qualify
                                               sp_df_aliased["TerritoryID"],
                                               sp_df_aliased["CountryRegionCode"],
                                               cr_df_aliased["Name"].alias("CountryRegionName"), # Explicitly qualify
                                               sp_df_aliased["rowguid"].alias("StateProvinceRowGUID"), # Explicitly qualify
                                               sp_df_aliased["ModifiedDate"].alias("StateProvinceModifiedDate") # Explicitly qualify
                                           )
    window_spec = Window.partitionBy("StateProvinceID").orderBy(col("StateProvinceModifiedDate").desc())
    dim_stateprovince_df = dim_stateprovince_df.withColumn("rn", row_number().over(window_spec)) \
                                               .filter(col("rn") == 1) \
                                               .drop("rn")
    print("DimStateProvince Schema:")
    dim_stateprovince_df.printSchema()
    print("DimStateProvince Sample Data:")
    dim_stateprovince_df.show(5)
else:
    print("Skipping DimStateProvince creation: 'stateprovince.csv' or 'countryregion.csv' not loaded.")


if address_df and dim_stateprovince_df:
    # Join address with stateprovince and countryregion
    # Ensure address_df is aliased if it has ambiguous columns with dim_stateprovince_df
    addr_df_aliased = address_df.alias("addr")
    
    dim_address_base_df = addr_df_aliased.join(dim_stateprovince_df, on=addr_df_aliased["StateProvinceID"] == dim_stateprovince_df["StateProvinceID"], how="left") \
                                    .select(
                                        addr_df_aliased["AddressID"],
                                        addr_df_aliased["AddressLine1"],
                                        addr_df_aliased["AddressLine2"],
                                        addr_df_aliased["City"],
                                        # StateProvinceID is the join key, so it's unambiguous
                                        dim_stateprovince_df["StateProvinceID"], # Use the one from dim_stateprovince_df
                                        dim_stateprovince_df["StateProvinceCode"],
                                        dim_stateprovince_df["StateProvinceName"], # Now includes name
                                        addr_df_aliased["PostalCode"],
                                        # Removed 'SpatialLocation' as it's causing an error
                                        # col("SpatialLocation"),
                                        dim_stateprovince_df["CountryRegionCode"],
                                        dim_stateprovince_df["CountryRegionName"],
                                        addr_df_aliased["rowguid"].alias("AddressRowGUID"), # Qualify with address_df object
                                        addr_df_aliased["ModifiedDate"].alias("AddressModifiedDate") # Qualify with address_df object
                                    )

    # Create DimAddress: Contains full address details.
    dim_address_df = dim_address_base_df.select(
        "AddressID",
        "AddressLine1",
        "AddressLine2",
        "City",
        "StateProvinceID",
        "StateProvinceCode",
        "StateProvinceName",
        "PostalCode",
        "CountryRegionCode",
        "CountryRegionName",
        "AddressRowGUID",
        "AddressModifiedDate"
    )

    # Ensure unique addresses, similar to other dimensions (simplified SCD Type 1).
    window_spec = Window.partitionBy("AddressID").orderBy(col("AddressModifiedDate").desc())
    dim_address_df = dim_address_df.withColumn("rn", row_number().over(window_spec)) \
                                   .filter(col("rn") == 1) \
                                   .drop("rn")

    print("DimAddress Schema:")
    dim_address_df.printSchema()
    print("DimAddress Sample Data:")
    dim_address_df.show(5)

    # Create DimGeography: Focuses on the hierarchical geographic information.
    # We drop duplicates to ensure each unique geographic combination appears once.
    dim_geography_df = dim_address_base_df.select(
        col("City"),
        col("StateProvinceID"),
        col("StateProvinceCode"),
        col("StateProvinceName"),
        col("CountryRegionCode"),
        col("CountryRegionName")
    ).dropDuplicates()

    # Add a surrogate key for DimGeography.
    dim_geography_df = dim_geography_df.withColumn("GeographyKey", monotonically_increasing_id())

    print("DimGeography Schema:")
    dim_geography_df.printSchema()
    print("DimGeography Sample Data:")
    dim_geography_df.show(5)

else:
    print("Skipping DimAddress/DimGeography creation: 'address.csv' or DimStateProvince not loaded.")
    dim_address_df = None
    dim_geography_df = None

# --- Step 6: Create DimCreditCard Dimension ---
# Explanation: This dimension table prepares the credit card data for analysis.

print("\n--- Creating DimCreditCard Dimension ---")

if creditcard_df:
    dim_creditcard_df = creditcard_df.select(
        col("CreditCardID"),
        col("CardType"),
        col("CardNumber"),
        col("ExpMonth"),
        col("ExpYear"),
        col("ModifiedDate").alias("CreditCardModifiedDate") # Keep original modified date
    )

    # Ensure unique credit cards (simplified SCD Type 1).
    window_spec = Window.partitionBy("CreditCardID").orderBy(col("CreditCardModifiedDate").desc())
    dim_creditcard_df = dim_creditcard_df.withColumn("rn", row_number().over(window_spec)) \
                                         .filter(col("rn") == 1) \
                                         .drop("rn")

    print("DimCreditCard Schema:")
    dim_creditcard_df.printSchema()
    print("DimCreditCard Sample Data:")
    dim_creditcard_df.show(5)
else:
    print("Skipping DimCreditCard creation: 'creditcard.csv' not loaded.")
    dim_creditcard_df = None

# --- Step 7: Create DimCurrency Dimension ---
# Explanation: This dimension table prepares the currency data for analysis.

print("\n--- Creating DimCurrency Dimension ---")

if currency_df:
    dim_currency_df = currency_df.select(
        col("CurrencyCode"),
        col("Name").alias("CurrencyName"), # Rename for clarity
        col("ModifiedDate").alias("CurrencyModifiedDate") # Keep original modified date
    )

    # Ensure unique currencies (simplified SCD Type 1).
    window_spec = Window.partitionBy("CurrencyCode").orderBy(col("CurrencyModifiedDate").desc())
    dim_currency_df = dim_currency_df.withColumn("rn", row_number().over(window_spec)) \
                                     .filter(col("rn") == 1) \
                                     .drop("rn")

    print("DimCurrency Schema:")
    dim_currency_df.printSchema()
    print("DimCurrency Sample Data:")
    dim_currency_df.show(5)
else:
    print("Skipping DimCurrency creation: 'currency.csv' not loaded.")
    dim_currency_df = None

# --- Step 8: Create DimSalesTerritory Dimension ---
# Explanation: This dimension table prepares the sales territory data for analysis.

print("\n--- Creating DimSalesTerritory Dimension ---")

dim_sales_territory_df = None
if salesterritory_df:
    dim_sales_territory_df = salesterritory_df.select(
        col("TerritoryID"),
        col("Name").alias("TerritoryName"),
        col("CountryRegionCode"),
        col("SalesLastYear"),
        col("CostLastYear"),
        col("rowguid").alias("TerritoryRowGUID"),
        col("ModifiedDate").alias("TerritoryModifiedDate")
    )
    window_spec = Window.partitionBy("TerritoryID").orderBy(col("TerritoryModifiedDate").desc())
    dim_sales_territory_df = dim_sales_territory_df.withColumn("rn", row_number().over(window_spec)) \
                                                   .filter(col("rn") == 1) \
                                                   .drop("rn")
    print("DimSalesTerritory Schema:")
    dim_sales_territory_df.printSchema()
    print("DimSalesTerritory Sample Data:")
    dim_sales_territory_df.show(5)
else:
    print("Skipping DimSalesTerritory creation: 'salesterritory.csv' not loaded.")

# --- Step 9: Create DimShipMethod Dimension ---
# Explanation: This dimension table prepares the ship method data for analysis.

print("\n--- Creating DimShipMethod Dimension ---")

dim_ship_method_df = None
if shipmethod_df:
    dim_ship_method_df = shipmethod_df.select(
        col("ShipMethodID"),
        col("Name").alias("ShipMethodName"),
        col("ShipBase"),
        col("ShipRate"),
        col("rowguid").alias("ShipMethodRowGUID"),
        col("ModifiedDate").alias("ShipMethodModifiedDate")
    )
    # Corrected: Use the aliased column 'ShipMethodModifiedDate' in orderBy
    window_spec = Window.partitionBy("ShipMethodID").orderBy(col("ShipMethodModifiedDate").desc())
    dim_ship_method_df = dim_ship_method_df.withColumn("rn", row_number().over(window_spec)) \
                                           .filter(col("rn") == 1) \
                                           .drop("rn")
    print("DimShipMethod Schema:")
    dim_ship_method_df.printSchema()
    print("DimShipMethod Sample Data:")
    dim_ship_method_df.show(5)
else:
    print("Skipping DimShipMethod creation: 'shipmethod.csv' not loaded.")

# --- Step 10: Create DimVendor Dimension ---
# Explanation: This dimension table prepares the vendor data for analysis.

print("\n--- Creating DimVendor Dimension ---")

dim_vendor_df = None
if vendor_df:
    dim_vendor_df = vendor_df.select(
        col("BusinessEntityID").alias("VendorID"), # Rename for clarity
        col("AccountNumber").alias("VendorAccountNumber"),
        col("Name").alias("VendorName"),
        col("CreditRating"),
        col("PreferredVendorStatus"),
        col("ActiveFlag"),
        col("PurchasingWebServiceURL"),
        col("ModifiedDate").alias("VendorModifiedDate")
    )
    window_spec = Window.partitionBy("VendorID").orderBy(col("VendorModifiedDate").desc())
    dim_vendor_df = dim_vendor_df.withColumn("rn", row_number().over(window_spec)) \
                                 .filter(col("rn") == 1) \
                                 .drop("rn")
    print("DimVendor Schema:")
    dim_vendor_df.printSchema()
    print("DimVendor Sample Data:")
    dim_vendor_df.show(5)
else:
    print("Skipping DimVendor creation: 'vendor.csv' not loaded.")


# --- Step 11: Create FactSales Table ---
# Explanation: This is the core fact table for revenue analysis. It combines sales order
# header and detail information and links to relevant dimensions using their keys.

print("\n--- Creating FactSales Table ---")

fact_sales_df = None
if salesorderheader_df and salesorderdetail_df and dim_date_df and dim_customer_df and dim_product_df and dim_address_df and dim_sales_territory_df and dim_ship_method_df and dim_creditcard_df and dim_currency_df:
    # Join sales order header and detail
    fact_sales_df = salesorderheader_df.join(salesorderdetail_df, on="SalesOrderID", how="inner") \
                                     .select(
                                         col("SalesOrderID"),
                                         col("SalesOrderDetailID"),
                                         col("OrderDate").cast("date"),
                                         col("DueDate").cast("date"),
                                         col("ShipDate").cast("date"),
                                         col("Status").alias("OrderStatus"),
                                         col("OnlineOrderFlag"),
                                         col("PurchaseOrderNumber"),
                                         col("AccountNumber"),
                                         col("CustomerID"),
                                         col("SalesPersonID"),
                                         col("TerritoryID"), # Will join with DimSalesTerritory
                                         col("BillToAddressID"), # Will join with DimAddress
                                         col("ShipToAddressID"), # Will join with DimAddress
                                         col("ShipMethodID"), # Will join with DimShipMethod
                                         col("CreditCardID"), # Will join with DimCreditCard
                                         col("CreditCardApprovalCode"),
                                         col("CurrencyRateID"), # Not directly a currency code, but a rate ID
                                         col("SubTotal"),
                                         col("TaxAmt"),
                                         col("Freight"),
                                         col("TotalDue"),
                                         col("ProductID"), # Will join with DimProduct
                                         col("OrderQty"),
                                         col("UnitPrice"),
                                         col("UnitPriceDiscount"),
                                         col("LineTotal"), # This is the revenue per line item
                                         salesorderheader_df["ModifiedDate"].alias("SalesHeaderModifiedDate"), # Qualify with salesorderheader_df object
                                         salesorderdetail_df["ModifiedDate"].alias("SalesDetailModifiedDate") # Qualify with salesorderdetail_df object
                                     )

    # Add DateKeys by joining with DimDate
    fact_sales_df = fact_sales_df.join(dim_date_df.select(col("Date").alias("OrderDate"), col("DateKey").alias("OrderDateKey")), on="OrderDate", how="left")
    fact_sales_df = fact_sales_df.join(dim_date_df.select(col("Date").alias("DueDate"), col("DateKey").alias("DueDateKey")), on="DueDate", how="left")
    fact_sales_df = fact_sales_df.join(dim_date_df.select(col("Date").alias("ShipDate"), col("DateKey").alias("ShipDateKey")), on="ShipDate", how="left")

    # Add foreign keys from other dimensions
    # CustomerID is already in FactSales, but we can ensure it's linked to DimCustomer
    # ProductID is already in FactSales, but we can ensure it's linked to DimProduct
    # TerritoryID -> SalesTerritoryKey
    # BillToAddressID -> BillToAddressKey (from DimAddress)
    # ShipToAddressID -> ShipToAddressKey (from DimAddress)
    # ShipMethodID -> ShipMethodKey
    # CreditCardID -> CreditCardKey
    # CurrencyRateID is present, but no direct DimCurrencyRate table from currency.csv.
    # We will use CurrencyCode from DimCurrency if available, or keep CurrencyRateID as is.
    # For simplicity, we'll assume CurrencyRateID can be used as a direct link or will be handled in Gold.

    # Add LoadDate for lineage
    fact_sales_df = fact_sales_df.withColumn("LoadDate", current_timestamp())

    print("FactSales Schema:")
    fact_sales_df.printSchema()
    print("FactSales Sample Data:")
    fact_sales_df.show(5)
else:
    print("Skipping FactSales creation: Missing one or more required Dataframes (salesorderheader, salesorderdetail, or dimensions).")
    fact_sales_df = None


# --- Step 12: Write Transformed Data to Silver Layer as CSV ---
# Explanation: We persist the cleaned and transformed dimension and fact tables to a new Silver CSV directory.
# This uses CSV format with headers.

print("\n--- Writing Silver Layer Data as CSV ---")

# Create the Silver CSV directory if it doesn't already exist.
os.makedirs(silver_dir, exist_ok=True)

# A dictionary mapping table names to their respective DataFrames.
silver_tables = {
    "dim_date": dim_date_df,
    "dim_customer": dim_customer_df,
    "dim_product": dim_product_df,
    "dim_product_category": dim_product_category_df,
    "dim_product_subcategory": dim_product_subcategory_df,
    "dim_unit_measure": dim_unit_measure_df,
    "dim_address": dim_address_df,
    "dim_geography": dim_geography_df,
    "dim_state_province": dim_stateprovince_df,
    "dim_creditcard": dim_creditcard_df,
    "dim_currency": dim_currency_df,
    "dim_sales_territory": dim_sales_territory_df,
    "dim_ship_method": dim_ship_method_df,
    "dim_vendor": dim_vendor_df,
    "fact_sales": fact_sales_df
}

# Loop through each DataFrame and write it to the Silver CSV layer.
for table_name, df in silver_tables.items():
    if df is not None: # Only write if the DataFrame was successfully created.
        silver_table_csv_path = os.path.join(silver_dir, table_name)
        try:
            # Use 'overwrite' mode to replace existing data if the script is rerun.
            # Write as CSV with header option
            df.write.option("header", "true").mode("overwrite").csv(silver_table_csv_path)
            print(f"Successfully written {table_name} to {silver_table_csv_path} as CSV")
        except Exception as e:
            print(f"Error writing {table_name} to Silver layer as CSV: {e}")
    else:
        print(f"DataFrame for {table_name} is None, skipping CSV write.")

print("\n--- Silver Layer Processing Complete ---")

# Stop the Spark Session to release resources.
spark.stop()
