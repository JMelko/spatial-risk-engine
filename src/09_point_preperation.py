import duckdb

def prepare_exact_points(input_path, output_path):
    print("Initializing DuckDB and loading spatial extension...")
    con = duckdb.connect(database=':memory:')
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    print("Projecting all 1.8 million exact points to Albers Equal Area (EPSG:5070)...")
    print("This keeps every record independent for micro-habitat intersection.")
    
    # We are NOT using GROUP BY here. We want every single row preserved, 
    # just adding the new meter-based X and Y columns.
    query = f"""
        COPY (
            SELECT 
                source,
                date,
                scientific_name,
                observation_count,
                ST_X(ST_Transform(ST_Point(longitude, latitude), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS x_meters,
                ST_Y(ST_Transform(ST_Point(longitude, latitude), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS y_meters
            FROM '{input_path}/*.parquet'
        ) TO '{output_path}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    print(f"Point preparation complete! Saved to {output_path}")
    
    print("\n--- Sample of Exact Projected Points ---")
    sample_query = f"SELECT * FROM '{output_path}' LIMIT 5;"
    print(con.execute(sample_query).df().to_string(index=False))

if __name__ == "__main__":
    INPUT_DIR = "data/processed/combined_species_data.parquet"
    OUTPUT_FILE = "data/processed/exact_points_5070.parquet"
    
    prepare_exact_points(INPUT_DIR, OUTPUT_FILE)