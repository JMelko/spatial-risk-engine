import duckdb
import time

def build_species_database():
    print("1. Initializing DuckDB Engine...")
    conn = duckdb.connect('data/processed/spatial_risk.db')
    
    print("2. Installing and Loading Spatial Extensions...")
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")
    
    print("3. Executing SQL Extraction and Merge...")
    start_time = time.time()
    
    # Target species by Scientific Name
    sql_species = "'Calypte anna', 'Parabuteo unicinctus', 'Aphelocoma wollweberi', 'Coccyzus americanus'"

    query = f"""
    CREATE OR REPLACE TABLE species_observations AS
    
    WITH combined_data AS (
        -- Extract from eBird: Explicitly CAST strings to DOUBLE for coordinates
        SELECT 
            scientific_name,
            CAST(latitude AS DOUBLE) AS latitude,
            CAST(longitude AS DOUBLE) AS longitude,
            observation_date,
            'eBird' AS source
        FROM read_parquet('data/interim/ebird_zero_filled.parquet')
        WHERE scientific_name IN ({sql_species})
        
        UNION ALL
        
        -- Extract from iNaturalist: Alias columns to match the eBird schema
        SELECT 
            scientificName AS scientific_name,
            decimalLatitude AS latitude,
            decimalLongitude AS longitude,
            eventDate AS observation_date,
            'iNaturalist' AS source
        FROM read_parquet('data/interim/inaturalist_filtered.parquet')
        WHERE scientificName IN ({sql_species})
    )
    SELECT 
        scientific_name,
        -- Generate the common names dynamically for Streamlit
        CASE 
            WHEN scientific_name = 'Calypte anna' THEN 'Anna''s Hummingbird'
            WHEN scientific_name = 'Parabuteo unicinctus' THEN 'Harris''s Hawk'
            WHEN scientific_name = 'Aphelocoma wollweberi' THEN 'Mexican Jay'
            WHEN scientific_name = 'Coccyzus americanus' THEN 'Yellow-billed Cuckoo'
        END AS common_name,
        latitude,
        longitude,
        observation_date,
        source
    FROM combined_data;
    """
    
    conn.execute(query)
    
    print("4. Exporting Model-Ready Parquet...")
    export_query = """
    COPY species_observations 
    TO 'data/processed/target_species_filtered.parquet' 
    (FORMAT PARQUET);
    """
    conn.execute(export_query)
    
    # Check the results and grouping by the dynamically created common_name
    result = conn.execute("SELECT common_name, COUNT(*) as count FROM species_observations GROUP BY common_name;").df()
    
    end_time = time.time()
    print(f"\nExtraction complete in {round(end_time - start_time, 2)} seconds.")
    print("\nObservation Counts by Species:")
    print(result.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    build_species_database()