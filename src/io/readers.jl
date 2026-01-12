# HydroForge IO Readers
# Functions for reading input data files

# Placeholder for ArchGDAL-based readers
# Will be implemented in Phase 4

"""
    read_geotiff(path::String)

Read a GeoTIFF file and return (data, metadata).

# Returns
- `data::Matrix{Float64}`: Raster data
- `metadata::Dict`: Contains crs, transform, nodata_value
"""
function read_geotiff(path::String)
    # Placeholder - will use ArchGDAL
    error("GeoTIFF reading not yet implemented. Requires ArchGDAL.")
end

"""
    read_rainfall_csv(path::String)

Read rainfall time series from CSV file.

Expected format:
```
time,intensity
0,0
300,10.5
600,25.0
...
```

Where time is in seconds and intensity in mm/hr.

# Returns
- `RainfallEvent`: Parsed rainfall event
"""
function read_rainfall_csv(path::String)
    # Placeholder - will be implemented
    error("CSV reading not yet implemented.")
end

"""
    read_points_geojson(path::String)

Read output points from GeoJSON file.

# Returns
- `Vector{Tuple{Float64,Float64}}`: List of (x, y) coordinates
"""
function read_points_geojson(path::String)
    # Placeholder
    error("GeoJSON reading not yet implemented.")
end
