# HydroForge Scenario Type
# Bundles all simulation inputs

"""
    RainfallEvent{T<:AbstractFloat}

Represents a rainfall time series for the simulation.

# Fields
- `times::Vector{T}`: Time points (s)
- `intensities::Vector{T}`: Rainfall intensity at each time point (mm/hr)
"""
struct RainfallEvent{T<:AbstractFloat}
    times::Vector{T}
    intensities::Vector{T}

    function RainfallEvent{T}(times, intensities) where T
        length(times) == length(intensities) ||
            throw(DimensionMismatch("times and intensities must have same length"))
        length(times) >= 2 ||
            throw(ArgumentError("Need at least 2 time points"))
        issorted(times) ||
            throw(ArgumentError("times must be sorted"))
        all(i -> i >= 0, intensities) ||
            throw(ArgumentError("intensities must be non-negative"))
        new{T}(times, intensities)
    end
end

function RainfallEvent(times::Vector{T}, intensities::Vector{T}) where T<:AbstractFloat
    RainfallEvent{T}(times, intensities)
end

"""
    rainfall_rate(event::RainfallEvent, t)

Get rainfall intensity at time t (mm/hr) via linear interpolation.
Returns 0 outside the defined time range.
"""
function rainfall_rate(event::RainfallEvent{T}, t::Real) where T
    if t <= event.times[1]
        return event.intensities[1]
    elseif t >= event.times[end]
        return event.intensities[end]
    end

    # Find bracketing indices
    idx = searchsortedlast(event.times, t)
    t1, t2 = event.times[idx], event.times[idx+1]
    i1, i2 = event.intensities[idx], event.intensities[idx+1]

    # Linear interpolation
    T(i1 + (i2 - i1) * (t - t1) / (t2 - t1))
end

"""
    rainfall_rate_ms(event::RainfallEvent, t)

Get rainfall intensity at time t in m/s (for numerical computation).
"""
function rainfall_rate_ms(event::RainfallEvent{T}, t::Real) where T
    # Convert mm/hr to m/s: mm/hr * (1m/1000mm) * (1hr/3600s)
    rainfall_rate(event, t) / (1000.0 * 3600.0)
end

"""
    total_rainfall(event::RainfallEvent)

Calculate total rainfall depth over the event (mm).
"""
function total_rainfall(event::RainfallEvent{T}) where T
    total = zero(T)
    for i in 1:length(event.times)-1
        dt_hr = (event.times[i+1] - event.times[i]) / 3600.0  # Convert to hours
        avg_intensity = (event.intensities[i] + event.intensities[i+1]) / 2
        total += avg_intensity * dt_hr
    end
    total
end

"""
    duration(event::RainfallEvent)

Return the total duration of the rainfall event (s).
"""
duration(event::RainfallEvent) = event.times[end] - event.times[1]

"""
    peak_intensity(event::RainfallEvent)

Return the peak rainfall intensity (mm/hr).
"""
peak_intensity(event::RainfallEvent) = maximum(event.intensities)

# Show method
function Base.show(io::IO, event::RainfallEvent{T}) where T
    dur_hr = duration(event) / 3600
    print(io, "RainfallEvent{$T}($(round(dur_hr, digits=1))hr, peak=$(round(peak_intensity(event), digits=1))mm/hr)")
end


"""
    Scenario{T<:AbstractFloat}

Complete simulation scenario bundling all inputs.

# Fields
- `name::String`: Scenario identifier
- `grid::Grid{T}`: Computational grid
- `topography::Topography{T}`: Terrain and surface data
- `parameters::SimulationParameters{T}`: Runtime parameters
- `rainfall::RainfallEvent{T}`: Rainfall input
- `infiltration::Union{InfiltrationParameters{T}, Nothing}`: Infiltration parameters (optional)
- `output_points::Vector{Tuple{Int,Int}}`: Points for time series output
- `output_dir::String`: Directory for output files
"""
struct Scenario{T<:AbstractFloat}
    name::String
    grid::Grid{T}
    topography::Topography{T}
    parameters::SimulationParameters{T}
    rainfall::RainfallEvent{T}
    infiltration::Union{InfiltrationParameters{T}, Nothing}
    output_points::Vector{Tuple{Int,Int}}
    output_dir::String

    function Scenario{T}(name, grid, topography, parameters, rainfall, infiltration,
                         output_points, output_dir) where T
        # Validate dimensions match
        (grid.nx, grid.ny) == size(topography.elevation) ||
            throw(DimensionMismatch("Grid and topography dimensions must match"))

        # Validate output points are within grid
        for (i, j) in output_points
            (1 <= i <= grid.nx && 1 <= j <= grid.ny) ||
                throw(ArgumentError("Output point ($i, $j) outside grid bounds"))
        end

        new{T}(name, grid, topography, parameters, rainfall, infiltration, output_points, output_dir)
    end
end

# Full constructor
function Scenario(name::String, grid::Grid{T}, topography::Topography{T},
                  parameters::SimulationParameters{T}, rainfall::RainfallEvent{T},
                  infiltration::Union{InfiltrationParameters{T}, Nothing},
                  output_points::Vector{Tuple{Int,Int}}, output_dir::String) where T
    Scenario{T}(name, grid, topography, parameters, rainfall, infiltration, output_points, output_dir)
end

# Backward-compatible constructor without infiltration
function Scenario(name::String, grid::Grid{T}, topography::Topography{T},
                  parameters::SimulationParameters{T}, rainfall::RainfallEvent{T},
                  output_points::Vector{Tuple{Int,Int}}, output_dir::String) where T
    Scenario{T}(name, grid, topography, parameters, rainfall, nothing, output_points, output_dir)
end

"""
    validate(scenario::Scenario)

Validate entire scenario and return all warnings.
"""
function validate(scenario::Scenario)
    warnings = String[]

    # Validate components
    append!(warnings, validate(scenario.parameters))
    append!(warnings, validate(scenario.topography))

    # Check rainfall duration vs simulation time
    if duration(scenario.rainfall) < scenario.parameters.t_end
        push!(warnings, "Rainfall duration shorter than simulation time")
    end

    # Check for output directory
    if !isdir(scenario.output_dir) && scenario.output_dir != ""
        push!(warnings, "Output directory does not exist: $(scenario.output_dir)")
    end

    warnings
end

# Show method
function Base.show(io::IO, scenario::Scenario{T}) where T
    print(io, "Scenario{$T}(\"$(scenario.name)\", $(scenario.grid.nx)x$(scenario.grid.ny))")
end
