# HydroForge IO Writers
# Functions for writing output data files

using DelimitedFiles
using JSON3
using Dates

"""
    write_array(path::String, data::Matrix)

Write a matrix to a simple text format (space-separated values).
"""
function write_array(path::String, data::Matrix)
    open(path, "w") do io
        writedlm(io, data, ' ')
    end
    path
end

"""
    write_geotiff(path::String, data::Matrix, grid::Grid)

Write raster data to GeoTIFF format.

Note: Full GeoTIFF support requires ArchGDAL.jl.
Falls back to text format if path ends with .txt or .asc.
"""
function write_geotiff(path::String, data::Matrix, grid::Grid)
    if endswith(path, ".tif") || endswith(path, ".tiff")
        error("GeoTIFF writing requires ArchGDAL.jl. Use .txt extension for text output.")
    elseif endswith(path, ".txt") || endswith(path, ".asc")
        write_array(path, data)
    else
        # Default to text format
        write_array(path * ".txt", data)
    end
end

"""
    write_hydrograph_csv(path::String, hydrograph::Vector{NTuple{4,T}}) where T

Write point hydrograph to CSV file.

# Arguments
- `path`: Output file path
- `hydrograph`: Vector of (time, depth, qx, qy) tuples
"""
function write_hydrograph_csv(path::String, hydrograph::Vector{NTuple{4,T}}) where T
    open(path, "w") do io
        println(io, "time,depth,qx,qy")
        for (t, h, qx, qy) in hydrograph
            println(io, "$t,$h,$qx,$qy")
        end
    end
    path
end

"""
    write_hydrograph_csv(path::String, hydrograph::Vector{Tuple{T,T}}) where T

Write point hydrograph to CSV file (2-tuple variant for backwards compatibility).

# Arguments
- `path`: Output file path
- `hydrograph`: Vector of (time, depth) tuples
"""
function write_hydrograph_csv(path::String, hydrograph::Vector{Tuple{T,T}}) where T
    open(path, "w") do io
        println(io, "time,depth")
        for (t, h) in hydrograph
            println(io, "$t,$h")
        end
    end
    path
end

"""
    write_hydrograph_csv(path::String, times::Vector, depths::Vector)

Write point hydrograph to CSV file from separate vectors.
"""
function write_hydrograph_csv(path::String, times::Vector, depths::Vector)
    length(times) == length(depths) || throw(DimensionMismatch("times and depths must have same length"))
    open(path, "w") do io
        println(io, "time,depth")
        for i in eachindex(times)
            println(io, "$(times[i]),$(depths[i])")
        end
    end
    path
end

"""
    write_results_json(path::String, metadata::Dict)

Write run metadata to JSON file.
"""
function write_results_json(path::String, metadata::Dict)
    open(path, "w") do io
        JSON3.pretty(io, metadata)
    end
    path
end

"""
    ResultsPackage{T}

Container for all simulation outputs.

# Fields
- `max_depth`: Maximum water depth reached at each cell (m)
- `arrival_time`: Time when water first arrived at each cell (s)
- `max_velocity`: Maximum velocity magnitude at each cell (m/s)
- `point_hydrographs`: Time series at specified output points - (time, depth, qx, qy) tuples
- `metadata`: Run metadata dictionary
"""
struct ResultsPackage{T<:AbstractFloat}
    max_depth::Matrix{T}
    arrival_time::Matrix{T}
    max_velocity::Matrix{T}
    point_hydrographs::Dict{Tuple{Int,Int}, Vector{NTuple{4,T}}}
    metadata::Dict{String,Any}
end

# Note: ResultsPackage constructor from ResultsAccumulator is in models/simulation.jl
# to avoid circular dependency issues

"""
    write_results(output_dir::String, results::ResultsPackage, grid::Grid)

Write all results to output directory.

Creates the following files:
- max_depth.txt: Maximum depth raster
- arrival_time.txt: First arrival time raster
- max_velocity.txt: Maximum velocity raster
- hydrograph_i_j.csv: Time series for each output point
- metadata.json: Run metadata
"""
function write_results(output_dir::String, results::ResultsPackage{T}, grid::Grid{T}) where T
    # Create output directory if needed
    mkpath(output_dir)

    # Write rasters
    write_array(joinpath(output_dir, "max_depth.txt"), results.max_depth)
    write_array(joinpath(output_dir, "arrival_time.txt"), results.arrival_time)
    write_array(joinpath(output_dir, "max_velocity.txt"), results.max_velocity)

    # Write hydrographs
    for ((i, j), hydrograph) in results.point_hydrographs
        filename = "hydrograph_$(i)_$(j).csv"
        write_hydrograph_csv(joinpath(output_dir, filename), hydrograph)
    end

    # Write metadata
    write_results_json(joinpath(output_dir, "metadata.json"), results.metadata)

    @info "Results written to $output_dir"
    output_dir
end

# Note: write_results(output_dir, results::ResultsAccumulator, scenario, run_id)
# is defined in models/simulation.jl to avoid circular dependency
