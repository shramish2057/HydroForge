# HydroForge Rainfall Module
# Rainfall source term application with spatial and temporal support

# Note: RainfallEvent type is defined in types/scenario.jl

# =============================================================================
# Spatial Rainfall Types
# =============================================================================

"""
    SpatialRainfallEvent{T<:AbstractFloat}

Spatially and temporally varying rainfall event.

# Fields
- `times::Vector{T}`: Time points (s)
- `fields::Vector{Matrix{T}}`: Rainfall intensity fields at each time (mm/hr)
- `interpolation::Symbol`: Interpolation method (:nearest, :linear)
"""
struct SpatialRainfallEvent{T<:AbstractFloat}
    times::Vector{T}
    fields::Vector{Matrix{T}}  # Each matrix is intensity in mm/hr
    interpolation::Symbol
end

"""
    SpatialRainfallEvent(times, fields; interpolation=:linear)

Create a spatial rainfall event from time series of rainfall fields.

# Arguments
- `times`: Vector of time points (seconds)
- `fields`: Vector of 2D matrices (rainfall intensity in mm/hr)
- `interpolation`: :nearest or :linear (default)
"""
function SpatialRainfallEvent(times::Vector{T}, fields::Vector{Matrix{T}};
                               interpolation::Symbol=:linear) where T<:AbstractFloat
    @assert length(times) == length(fields) "Number of times must match number of fields"
    @assert length(times) >= 1 "Need at least one time point"
    @assert issorted(times) "Times must be in ascending order"

    # Check all fields have same size
    nx, ny = size(fields[1])
    for (i, f) in enumerate(fields)
        @assert size(f) == (nx, ny) "All fields must have same dimensions (field $i has $(size(f)), expected ($nx, $ny))"
    end

    SpatialRainfallEvent{T}(times, fields, interpolation)
end

"""
    SpatialRainfallEvent(grid::Grid, uniform_rainfall::RainfallEvent)

Convert a uniform rainfall event to a spatial event (constant across space).
"""
function SpatialRainfallEvent(grid::Grid{T}, rainfall::RainfallEvent{T}) where T
    fields = [fill(rainfall.intensities[i], grid.nx, grid.ny)
              for i in 1:length(rainfall.times)]
    SpatialRainfallEvent{T}(copy(rainfall.times), fields, :linear)
end

"""
    spatial_rainfall_rate(rainfall::SpatialRainfallEvent, t::Real)

Get the rainfall intensity field at time t (mm/hr).
"""
function spatial_rainfall_rate(rainfall::SpatialRainfallEvent{T}, t::Real) where T
    times = rainfall.times
    fields = rainfall.fields

    # Before first time point
    if t <= times[1]
        return fields[1]
    end

    # After last time point
    if t >= times[end]
        return fields[end]
    end

    # Find bracketing interval
    idx = searchsortedlast(times, T(t))

    if rainfall.interpolation == :nearest
        # Return nearest field
        if idx == length(times)
            return fields[end]
        end
        t_mid = (times[idx] + times[idx+1]) / 2
        return t <= t_mid ? fields[idx] : fields[idx+1]
    else
        # Linear interpolation
        t1, t2 = times[idx], times[idx+1]
        α = (T(t) - t1) / (t2 - t1)

        # Interpolate field values
        result = similar(fields[1])
        @inbounds for j in axes(result, 2), i in axes(result, 1)
            result[i, j] = (one(T) - α) * fields[idx][i, j] + α * fields[idx+1][i, j]
        end
        return result
    end
end

"""
    spatial_rainfall_rate_ms(rainfall::SpatialRainfallEvent, t::Real)

Get the rainfall intensity field at time t in m/s.
"""
function spatial_rainfall_rate_ms(rainfall::SpatialRainfallEvent{T}, t::Real) where T
    field_mmhr = spatial_rainfall_rate(rainfall, t)
    # Convert mm/hr to m/s
    conversion = T(1.0 / (1000.0 * 3600.0))
    return field_mmhr .* conversion
end

# =============================================================================
# Rainfall Application Functions
# =============================================================================

"""
    apply_rainfall!(h::Matrix, rainfall::RainfallEvent, t::Real, dt::Real)

Apply rainfall to the domain for a timestep.

Adds rainfall depth uniformly across the domain.

# Arguments
- `h`: Water depth matrix (modified in-place)
- `rainfall`: Rainfall event with time series
- `t`: Current simulation time (s)
- `dt`: Timestep duration (s)
"""
function apply_rainfall!(h::Matrix{T}, rainfall::RainfallEvent, t::Real, dt::Real) where T
    # Get rainfall rate in m/s
    rate_ms = rainfall_rate_ms(rainfall, t)

    if rate_ms > 0
        # Add rainfall depth: dh = rate * dt
        dh = T(rate_ms * dt)
        @inbounds for j in axes(h, 2), i in axes(h, 1)
            h[i, j] += dh
        end
    end

    nothing
end

"""
    apply_rainfall!(h::Matrix, rainfall::SpatialRainfallEvent, t::Real, dt::Real)

Apply spatially varying rainfall to the domain.

# Arguments
- `h`: Water depth matrix (modified in-place)
- `rainfall`: Spatial rainfall event with time-varying fields
- `t`: Current simulation time (s)
- `dt`: Timestep duration (s)

# Returns
- Total rainfall volume added (m³/m²)
"""
function apply_rainfall!(h::Matrix{T}, rainfall::SpatialRainfallEvent{T},
                          t::Real, dt::Real) where T
    # Get rainfall field in m/s
    rate_field = spatial_rainfall_rate_ms(rainfall, t)

    total_depth = zero(T)
    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if rate_field[i, j] > zero(T)
            dh = rate_field[i, j] * T(dt)
            h[i, j] += dh
            total_depth += dh
        end
    end

    total_depth
end

"""
    apply_rainfall_spatial!(h::Matrix, rainfall_field::Matrix, dt::Real)

Apply a single rainfall field (spatially varying, constant in time).

# Arguments
- `h`: Water depth matrix (modified in-place)
- `rainfall_field`: Rainfall intensity field (mm/hr)
- `dt`: Timestep duration (s)
"""
function apply_rainfall_spatial!(h::Matrix{T}, rainfall_field::Matrix{T}, dt::Real) where T
    # Rainfall field is in mm/hr, convert to m/s
    conversion = T(1.0 / (1000.0 * 3600.0))

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        h[i, j] += rainfall_field[i, j] * conversion * dt
    end

    nothing
end

# =============================================================================
# Rainfall Analysis Functions
# =============================================================================

"""
    cumulative_rainfall(rainfall::RainfallEvent, t::Real)

Calculate cumulative rainfall depth up to time t (mm).
"""
function cumulative_rainfall(rainfall::RainfallEvent{T}, t::Real) where T
    if t <= rainfall.times[1]
        return zero(T)
    end

    total = zero(T)
    for i in 1:length(rainfall.times)-1
        t1, t2 = rainfall.times[i], rainfall.times[i+1]
        if t <= t1
            break
        end

        # Integration interval
        t_start = t1
        t_end = min(t2, t)

        dt_hr = (t_end - t_start) / 3600.0  # Convert to hours
        avg_intensity = (rainfall.intensities[i] + rainfall.intensities[i+1]) / 2
        total += avg_intensity * dt_hr

        if t <= t2
            break
        end
    end

    total
end

"""
    cumulative_rainfall(rainfall::SpatialRainfallEvent, t::Real)

Calculate cumulative rainfall field up to time t (mm).
Returns a matrix of cumulative depths.
"""
function cumulative_rainfall(rainfall::SpatialRainfallEvent{T}, t::Real) where T
    if t <= rainfall.times[1]
        return zeros(T, size(rainfall.fields[1]))
    end

    total = zeros(T, size(rainfall.fields[1]))

    for i in 1:length(rainfall.times)-1
        t1, t2 = rainfall.times[i], rainfall.times[i+1]
        if t <= t1
            break
        end

        # Integration interval
        t_start = t1
        t_end = min(t2, t)
        dt_hr = (t_end - t_start) / T(3600.0)

        # Average intensity field
        @inbounds for k in axes(total, 2), j in axes(total, 1)
            avg = (rainfall.fields[i][j, k] + rainfall.fields[i+1][j, k]) / 2
            total[j, k] += avg * dt_hr
        end

        if t <= t2
            break
        end
    end

    total
end

"""
    total_rainfall_volume(rainfall::SpatialRainfallEvent, grid::Grid)

Calculate total rainfall volume over the entire event (m³).
"""
function total_rainfall_volume(rainfall::SpatialRainfallEvent{T}, grid::Grid{T}) where T
    cumulative_field = cumulative_rainfall(rainfall, rainfall.times[end])
    total_mm = sum(cumulative_field)
    # Convert mm to m, multiply by cell area
    return total_mm / T(1000.0) * cell_area(grid)
end

"""
    max_intensity(rainfall::SpatialRainfallEvent)

Find the maximum rainfall intensity across all space and time (mm/hr).
"""
function max_intensity(rainfall::SpatialRainfallEvent{T}) where T
    max_val = zero(T)
    for field in rainfall.fields
        field_max = maximum(field)
        max_val = max(max_val, field_max)
    end
    max_val
end

"""
    mean_areal_rainfall(rainfall::SpatialRainfallEvent, t::Real)

Calculate the mean areal rainfall rate at time t (mm/hr).
"""
function mean_areal_rainfall(rainfall::SpatialRainfallEvent{T}, t::Real) where T
    field = spatial_rainfall_rate(rainfall, t)
    mean(field)
end

# Import mean for mean_areal_rainfall
using Statistics: mean
