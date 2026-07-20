import duckdb

def create_projected_spatial_grid(input_path, output_path):
    print("Initializing DuckDB and loading spatial extension...")
    con = duckdb.connect(database=':memory:')
    
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    # 150 meter spacing for the regional grid
    grid_spacing = 150 
    
    print(f"Projecting to Albers Equal Area and gridding to exactly {grid_spacing}x{grid_spacing} meters...")
    
    # Added always_xy := true to force DuckDB to read the point correctly
    query = f"""
        COPY (
            WITH projected_data AS (
                SELECT 
                    scientific_name,
                    observation_count,
                    ST_X(ST_Transform(ST_Point(longitude, latitude), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS x_meters,
                    ST_Y(ST_Transform(ST_Point(longitude, latitude), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS y_meters
                FROM '{input_path}/*.parquet'
            )
            SELECT 
                ROUND(x_meters / {grid_spacing}) * {grid_spacing} AS grid_x_5070,
                ROUND(y_meters / {grid_spacing}) * {grid_spacing} AS grid_y_5070,
                scientific_name,
                COUNT(*) as total_checklists,
                SUM(observation_count) as total_individuals
            FROM projected_data
            GROUP BY 
                grid_x_5070, 
                grid_y_5070, 
                scientific_name
        ) TO '{output_path}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    print(f"Grid aggregation complete! Saved to {output_path}")
    
    print("\n--- Sample of True 150m Gridded Data (Albers Equal Area) ---")
    sample_query = f"SELECT * FROM '{output_path}' LIMIT 5;"
    print(con.execute(sample_query).df().to_string(index=False))

if __name__ == "__main__":
    INPUT_DIR = "data/processed/combined_species_data.parquet"
    OUTPUT_FILE = "data/processed/spatial_grid_summary.parquet"
    
    create_projected_spatial_grid(INPUT_DIR, OUTPUT_FILE)