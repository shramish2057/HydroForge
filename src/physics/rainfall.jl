# HydroForge Rainfall Module
# Rainfall source term application

# Note: RainfallEvent type is defined in types/scenario.jl

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
    apply_rainfall_spatial!(h::Matrix, rainfall::SpatialRainfallEvent, t::Real, dt::Real)

Apply spatially varying rainfall (placeholder for future implementation).
"""
function apply_rainfall_spatial!(h::Matrix{T}, rainfall_field::Matrix{T}, dt::Real) where T
    # Rainfall field is in mm/hr, convert to m/s
    conversion = T(1.0 / (1000.0 * 3600.0))

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        h[i, j] += rainfall_field[i, j] * conversion * dt
    end

    nothing
end

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
