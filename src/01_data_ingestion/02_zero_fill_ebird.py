from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def create_spark_session():
    return SparkSession.builder \
        .appName("EBirdZeroFill") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()

def process_ebird_zero_fill(spark, sampling_path, obs_path, output_path):
    # eBird uses tab-separated files
    print("Loading Sampling Events...")
    sampling_df = spark.read.option("header", "true").option("sep", "\t").csv(sampling_path)
    
    print("Loading Observations...")
    obs_df = spark.read.option("header", "true").option("sep", "\t").csv(obs_path)
    
    # Selecting using the exact uppercase/spaced names found in the raw files
    sampling_df = sampling_df.select(
        F.col("SAMPLING EVENT IDENTIFIER").alias("sampling_event_identifier"),
        F.col("LATITUDE").alias("latitude"),
        F.col("LONGITUDE").alias("longitude"),
        F.col("OBSERVATION DATE").alias("observation_date")
    )
    
    obs_df = obs_df.select(
        F.col("SAMPLING EVENT IDENTIFIER").alias("sampling_event_identifier"),
        F.col("SCIENTIFIC NAME").alias("scientific_name"),
        F.col("OBSERVATION COUNT").alias("observation_count")
    )
    
    # Perform the join
    print("Joining and zero-filling...")
    joined_df = sampling_df.join(obs_df, "sampling_event_identifier", "left")
    
    # Fill nulls with 0
    final_df = joined_df.fillna({"observation_count": 0})
    
    print("Writing result...")
    final_df.write.mode("overwrite").parquet(output_path)

if __name__ == "__main__":
    spark = create_spark_session()
    
    SAMP_PATH = "data/raw/ebd_US-*_smp_relMay-2026_sampling.txt.gz"
    OBS_PATH = "data/raw/ebd_US-*_smp_relMay-2026.txt.gz"
    OUT_PATH = "data/interim/ebird_zero_filled.parquet"
    
    process_ebird_zero_fill(spark, SAMP_PATH, OBS_PATH, OUT_PATH)
    spark.stop()