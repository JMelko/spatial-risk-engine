import duckdb

def run_duckdb_analytics(parquet_path):
    print("Initializing DuckDB and loading spatial extension...")
    # Connect to an in-memory database
    con = duckdb.connect(database=':memory:')
    
    # Install and load the spatial extension (only needs to be downloaded once)
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    print("\nExecuting query directly against Parquet...")
    # We query the parquet file directly as if it were a table
    query = f"""
        SELECT 
            scientific_name,
            COUNT(*) as total_sightings,
            SUM(observation_count) as total_individuals
        FROM '{parquet_path}/*.parquet'
        GROUP BY scientific_name
        ORDER BY total_sightings DESC
        LIMIT 10;
    """
    
    # Fetch the results as a Pandas DataFrame for easy viewing
    results_df = con.execute(query).df()
    
    print("\n--- Top 10 Most Sighted Species ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    # Pointing to the directory containing our processed data
    PROCESSED_DIR = "data/processed/combined_species_data.parquet"
    run_duckdb_analytics(PROCESSED_DIR)