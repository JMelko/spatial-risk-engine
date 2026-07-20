import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def create_spark_session():
    """Initializes Spark, suppressing unnecessary logs and Hadoop errors."""
    spark = SparkSession.builder \
        .appName("Bird_Conflict_Risk_Pipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def process_gbif_inaturalist(spark, raw_path, interim_path):
    """Processes iNaturalist GBIF data."""
    print(f"Reading iNaturalist data from {raw_path}...")
    df_inat = spark.read.csv(raw_path, header=True, sep='\t', inferSchema=False)
    
    df_filtered = df_inat.select(
        F.col("eventDate"), 
        F.col("decimalLatitude").cast("double"), 
        F.col("decimalLongitude").cast("double"), 
        F.col("coordinateUncertaintyInMeters").cast("double"),
        F.col("scientificName")
    ).dropna(subset=["decimalLatitude", "decimalLongitude", "eventDate"])
    
    df_filtered = df_filtered.filter(
        (F.col("coordinateUncertaintyInMeters").isNull()) | 
        (F.col("coordinateUncertaintyInMeters") <= 5000)
    )
    
    print(f"Filtered iNaturalist row count: {df_filtered.count()}")
    df_filtered.write.mode("overwrite").parquet(interim_path)
    print("iNaturalist processing complete.\n")

def process_ebird(spark, raw_dir, interim_path):
    """Processes eBird data using wildcards to combine AZ and NM."""
    print("Reading eBird data...")
    # Using wildcards to load both AZ and NM files simultaneously
    smp_path = os.path.join(raw_dir, "ebd_US-*_smp_relMay-2026_sampling.txt.gz")
    obs_path = os.path.join(raw_dir, "ebd_US-*_smp_relMay-2026.txt.gz")
    
    df_smp = spark.read.csv(smp_path, header=True, sep='\t')
    df_obs = spark.read.csv(obs_path, header=True, sep='\t')
    
    print(f"EBird Checklists loaded: {df_smp.count()}")
    print(f"EBird Observations loaded: {df_obs.count()}")
    
    # We will expand this join logic in the next step!
    print("EBird pipeline initialized.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
    
    INAT_RAW_FILE = os.path.join(RAW_DIR, "gbif_occurrence.txt.gz")
    INAT_PARQUET = os.path.join(INTERIM_DIR, "inaturalist_filtered.parquet")
    
    spark = create_spark_session()
    
    try:
        process_gbif_inaturalist(spark, INAT_RAW_FILE, INAT_PARQUET)
        process_ebird(spark, RAW_DIR, os.path.join(INTERIM_DIR, "ebird_processed.parquet"))
    finally:
        print("Pipeline step 1 complete. Shutting down Spark...")
        spark.stop()