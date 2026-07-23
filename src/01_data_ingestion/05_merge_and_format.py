from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def create_spark_session():
    return SparkSession.builder \
        .appName("MergeAndFormat") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()

def merge_datasets(spark, ebird_path, inat_path, output_path):
    print("Loading interim datasets...")
    ebird_df = spark.read.parquet(ebird_path)
    inat_df = spark.read.parquet(inat_path)
    
    print("Formatting eBird data...")
    # Cast coordinates, handle 'X' by replacing with 1, cast count to integer, drop null species
    ebird_clean = ebird_df.dropna(subset=["scientific_name"]) \
        .withColumn("latitude", F.col("latitude").cast("double")) \
        .withColumn("longitude", F.col("longitude").cast("double")) \
        .withColumn("observation_count", F.when(F.col("observation_count") == 'X', 1)
                                          .otherwise(F.col("observation_count")).cast("integer")) \
        .withColumn("source", F.lit("ebird")) \
        .withColumnRenamed("observation_date", "date") \
        .select("source", "date", "latitude", "longitude", "scientific_name", "observation_count")

    print("Formatting iNaturalist data...")
    # Fix dates, extract binomial name, set count to 1 (presence-only), align column names
    inat_clean = inat_df.withColumn("date", F.substring(F.col("eventDate"), 1, 10)) \
        .withColumn("latitude", F.col("decimalLatitude").cast("double")) \
        .withColumn("longitude", F.col("decimalLongitude").cast("double")) \
        .withColumn("scientific_name", F.substring_index(F.col("scientificName"), " ", 2)) \
        .withColumn("observation_count", F.lit(1).cast("integer")) \
        .withColumn("source", F.lit("inaturalist")) \
        .select("source", "date", "latitude", "longitude", "scientific_name", "observation_count")

    print("Executing Union...")
    # Stack the two datasets on top of each other
    final_df = ebird_clean.unionByName(inat_clean)
    
    print("Writing final processed dataset...")
    final_df.write.mode("overwrite").parquet(output_path)

if __name__ == "__main__":
    spark = create_spark_session()
    
    EBIRD_PATH = "data/interim/ebird_zero_filled.parquet"
    INAT_PATH = "data/interim/inaturalist_filtered.parquet"
    OUT_PATH = "data/processed/combined_species_data.parquet"
    
    merge_datasets(spark, EBIRD_PATH, INAT_PATH, OUT_PATH)
    spark.stop()