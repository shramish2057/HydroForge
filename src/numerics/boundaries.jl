# HydroForge Boundary Conditions
# Boundary condition implementations with time-varying support

# =============================================================================
# Boundary Type Enumeration
# =============================================================================

"""
    BoundaryType

Enumeration of supported boundary condition types.
"""
@enum BoundaryType begin
    CLOSED        # No flow (reflective)
    OPEN          # Free outflow (transmissive)
    FIXED_DEPTH   # Fixed water surface elevation
    INFLOW        # Prescribed inflow hydrograph
    TIDAL         # Tidal boundary (time-varying depth)
    RATING_CURVE  # Stage-discharge relationship
end

# =============================================================================
# Time Series for Boundaries
# =============================================================================

"""
    BoundaryTimeSeries{T<:AbstractFloat}

Time-varying boundary condition values.

# Fields
- `times::Vector{T}`: Time points (s)
- `values::Vector{T}`: Boundary values at each time
- `interpolation::Symbol`: Interpolation method (:linear, :nearest)
"""
struct BoundaryTimeSeries{T<:AbstractFloat}
    times::Vector{T}
    values::Vector{T}
    interpolation::Symbol
end

"""
    BoundaryTimeSeries(times, values; interpolation=:linear)

Create a boundary time series.
"""
function BoundaryTimeSeries(times::Vector{<:Real}, values::Vector{<:Real};
                            interpolation::Symbol=:linear,
                            T::Type{<:AbstractFloat}=Float64)
    @assert length(times) == length(values) "Times and values must have same length"
    @assert length(times) >= 1 "Need at least one time point"
    @assert issorted(times) "Times must be sorted"

    BoundaryTimeSeries{T}(T.(times), T.(values), interpolation)
end

"""
    interpolate_boundary(ts::BoundaryTimeSeries, t::Real)

Interpolate boundary value at time t.
"""
function interpolate_boundary(ts::BoundaryTimeSeries{T}, t::Real) where T
    times = ts.times
    values = ts.values

    if t <= times[1]
        return values[1]
    elseif t >= times[end]
        return values[end]
    end

    idx = searchsortedlast(times, T(t))

    if ts.interpolation == :nearest
        t_mid = (times[idx] + times[idx+1]) / 2
        return t <= t_mid ? values[idx] : values[idx+1]
    else  # :linear
        t1, t2 = times[idx], times[idx+1]
        v1, v2 = values[idx], values[idx+1]
        α = (T(t) - t1) / (t2 - t1)
        return (one(T) - α) * v1 + α * v2
    end
end

# =============================================================================
# Tidal Boundary
# =============================================================================

"""
    TidalBoundary{T<:AbstractFloat}

Tidal boundary condition using harmonic constituents.

# Fields
- `mean_level::T`: Mean sea level (m)
- `amplitudes::Vector{T}`: Tidal constituent amplitudes (m)
- `periods::Vector{T}`: Tidal constituent periods (s)
- `phases::Vector{T}`: Tidal constituent phases (radians)
"""
struct TidalBoundary{T<:AbstractFloat}
    mean_level::T
    amplitudes::Vector{T}
    periods::Vector{T}
    phases::Vector{T}
end

"""
    TidalBoundary(mean_level; amplitude=1.0, period=12.42*3600)

Create a simple tidal boundary with single constituent (M2).
"""
function TidalBoundary(mean_level::Real;
                       amplitude::Real=1.0,
                       period::Real=12.42*3600,  # M2 tidal period (s)
                       phase::Real=0.0,
                       T::Type{<:AbstractFloat}=Float64)
    TidalBoundary{T}(T(mean_level), T[amplitude], T[period], T[phase])
end

"""
    TidalBoundary(mean_level, amplitudes, periods, phases; T=Float64)

Create a tidal boundary with multiple harmonic constituents.
"""
function TidalBoundary(mean_level::Real,
                       amplitudes::Vector{<:Real},
                       periods::Vector{<:Real},
                       phases::Vector{<:Real};
                       T::Type{<:AbstractFloat}=Float64)
    @assert length(amplitudes) == length(periods) == length(phases) "All arrays must have same length"
    TidalBoundary{T}(T(mean_level), T.(amplitudes), T.(periods), T.(phases))
end

"""
    tidal_level(tide::TidalBoundary, t::Real)

Calculate tidal water level at time t.
"""
function tidal_level(tide::TidalBoundary{T}, t::Real) where T
    level = tide.mean_level
    for i in eachindex(tide.amplitudes)
        ω = T(2π) / tide.periods[i]
        level += tide.amplitudes[i] * cos(ω * T(t) + tide.phases[i])
    end
    level
end

# =============================================================================
# Inflow Hydrograph
# =============================================================================

"""
    InflowHydrograph{T<:AbstractFloat}

Inflow boundary condition specified as discharge time series.

# Fields
- `times::Vector{T}`: Time points (s)
- `discharges::Vector{T}`: Discharge at each time (m³/s)
- `width::T`: Boundary width for distributing flow (m)
"""
struct InflowHydrograph{T<:AbstractFloat}
    times::Vector{T}
    discharges::Vector{T}
    width::T
end

"""
    InflowHydrograph(times, discharges; width=10.0)

Create an inflow hydrograph boundary.
"""
function InflowHydrograph(times::Vector{<:Real}, discharges::Vector{<:Real};
                          width::Real=10.0,
                          T::Type{<:AbstractFloat}=Float64)
    @assert length(times) == length(discharges) "Times and discharges must have same length"
    @assert issorted(times) "Times must be sorted"

    InflowHydrograph{T}(T.(times), T.(discharges), T(width))
end

"""
    inflow_discharge(hydro::InflowHydrograph, t::Real)

Get interpolated inflow discharge at time t (m³/s).
"""
function inflow_discharge(hydro::InflowHydrograph{T}, t::Real) where T
    times = hydro.times
    Q = hydro.discharges

    if t <= times[1]
        return Q[1]
    elseif t >= times[end]
        return Q[end]
    end

    idx = searchsortedlast(times, T(t))
    t1, t2 = times[idx], times[idx+1]
    Q1, Q2 = Q[idx], Q[idx+1]
    α = (T(t) - t1) / (t2 - t1)
    return (one(T) - α) * Q1 + α * Q2
end

"""
    inflow_flux(hydro::InflowHydrograph, t::Real)

Get inflow flux per unit width at time t (m²/s).
"""
function inflow_flux(hydro::InflowHydrograph{T}, t::Real) where T
    Q = inflow_discharge(hydro, t)
    return Q / hydro.width
end

# =============================================================================
# Rating Curve Boundary
# =============================================================================

"""
    RatingCurve{T<:AbstractFloat}

Stage-discharge rating curve for outflow boundaries.

# Fields
- `stages::Vector{T}`: Stage (water depth) values (m)
- `discharges::Vector{T}`: Discharge at each stage (m³/s)
- `datum::T`: Reference datum elevation (m)
"""
struct RatingCurve{T<:AbstractFloat}
    stages::Vector{T}
    discharges::Vector{T}
    datum::T
end

"""
    RatingCurve(stages, discharges; datum=0.0)

Create a rating curve from stage-discharge pairs.
"""
function RatingCurve(stages::Vector{<:Real}, discharges::Vector{<:Real};
                     datum::Real=0.0,
                     T::Type{<:AbstractFloat}=Float64)
    @assert length(stages) == length(discharges) "Stages and discharges must have same length"
    @assert issorted(stages) "Stages must be sorted"

    RatingCurve{T}(T.(stages), T.(discharges), T(datum))
end

"""
    RatingCurve(a, b, c; datum=0.0)

Create a power-law rating curve: Q = a × (h - c)^b
"""
function RatingCurve(a::Real, b::Real, c::Real=0.0;
                     max_depth::Real=10.0,
                     T::Type{<:AbstractFloat}=Float64)
    # Generate rating curve points
    stages = collect(T, range(c, max_depth, length=100))
    discharges = [h > c ? T(a) * (h - T(c))^T(b) : zero(T) for h in stages]
    RatingCurve{T}(stages, discharges, zero(T))
end

"""
    rating_discharge(curve::RatingCurve, h::Real)

Get discharge from rating curve for given stage.
"""
function rating_discharge(curve::RatingCurve{T}, h::Real) where T
    stage = T(h) - curve.datum
    stages = curve.stages
    Q = curve.discharges

    if stage <= stages[1]
        return Q[1]
    elseif stage >= stages[end]
        return Q[end]
    end

    # Linear interpolation
    idx = searchsortedlast(stages, stage)
    s1, s2 = stages[idx], stages[idx+1]
    Q1, Q2 = Q[idx], Q[idx+1]
    α = (stage - s1) / (s2 - s1)
    return (one(T) - α) * Q1 + α * Q2
end

# =============================================================================
# Boundary Condition Specification
# =============================================================================

"""
    BoundaryCondition{T}

Complete specification of boundary conditions for a domain.

# Fields
- `type::BoundaryType`: Type of boundary condition
- `fixed_depth::T`: Fixed depth value for FIXED_DEPTH type (m)
- `sides::NTuple{4,Bool}`: Which sides to apply (left, right, bottom, top)
- `time_series::Union{BoundaryTimeSeries{T}, Nothing}`: Time-varying values
- `tidal::Union{TidalBoundary{T}, Nothing}`: Tidal boundary parameters
- `hydrograph::Union{InflowHydrograph{T}, Nothing}`: Inflow hydrograph
- `rating_curve::Union{RatingCurve{T}, Nothing}`: Stage-discharge curve
"""
struct BoundaryCondition{T<:AbstractFloat}
    type::BoundaryType
    fixed_depth::T
    sides::NTuple{4,Bool}  # (left, right, bottom, top)
    time_series::Union{BoundaryTimeSeries{T}, Nothing}
    tidal::Union{TidalBoundary{T}, Nothing}
    hydrograph::Union{InflowHydrograph{T}, Nothing}
    rating_curve::Union{RatingCurve{T}, Nothing}
end

"""
    BoundaryCondition(type::BoundaryType; kwargs...)

Create a boundary condition specification.
"""
function BoundaryCondition(type::BoundaryType;
                           fixed_depth::Real=0.0,
                           sides::NTuple{4,Bool}=(true, true, true, true),
                           time_series::Union{BoundaryTimeSeries, Nothing}=nothing,
                           tidal::Union{TidalBoundary, Nothing}=nothing,
                           hydrograph::Union{InflowHydrograph, Nothing}=nothing,
                           rating_curve::Union{RatingCurve, Nothing}=nothing,
                           T::Type{<:AbstractFloat}=Float64)
    BoundaryCondition{T}(type, T(fixed_depth), sides,
                         time_series, tidal, hydrograph, rating_curve)
end

"""
    get_boundary_value(bc::BoundaryCondition, t::Real)

Get the boundary condition value at time t.
"""
function get_boundary_value(bc::BoundaryCondition{T}, t::Real) where T
    if bc.type == FIXED_DEPTH
        if bc.time_series !== nothing
            return interpolate_boundary(bc.time_series, t)
        else
            return bc.fixed_depth
        end
    elseif bc.type == TIDAL
        if bc.tidal !== nothing
            return tidal_level(bc.tidal, t)
        else
            return bc.fixed_depth
        end
    elseif bc.type == INFLOW
        if bc.hydrograph !== nothing
            return inflow_discharge(bc.hydrograph, t)
        else
            return zero(T)
        end
    else
        return bc.fixed_depth
    end
end

# =============================================================================
# Boundary Application Functions
# =============================================================================

"""
    apply_boundaries!(state::SimulationState, boundary::BoundaryType)

Apply boundary conditions to state in-place.
"""
function apply_boundaries!(state::SimulationState{T}, boundary::BoundaryType=CLOSED) where T
    if boundary == CLOSED
        apply_closed_boundaries!(state)
    elseif boundary == OPEN
        apply_open_boundaries!(state)
    elseif boundary == FIXED_DEPTH
        # Default fixed depth of 0 (dry boundary)
        apply_fixed_depth_boundaries!(state, zero(T))
    end
    nothing
end

"""
    apply_boundaries!(state::SimulationState, bc::BoundaryCondition)

Apply boundary conditions using full specification.
"""
function apply_boundaries!(state::SimulationState{T}, bc::BoundaryCondition{T}) where T
    if bc.type == CLOSED
        apply_closed_boundaries!(state, bc.sides)
    elseif bc.type == OPEN
        apply_open_boundaries!(state, bc.sides)
    elseif bc.type == FIXED_DEPTH
        apply_fixed_depth_boundaries!(state, bc.fixed_depth, bc.sides)
    end
    nothing
end

"""
    apply_boundaries!(state::SimulationState, bc::BoundaryCondition, t::Real)

Apply time-varying boundary conditions.
"""
function apply_boundaries!(state::SimulationState{T}, bc::BoundaryCondition{T}, t::Real) where T
    if bc.type == CLOSED
        apply_closed_boundaries!(state, bc.sides)
    elseif bc.type == OPEN
        apply_open_boundaries!(state, bc.sides)
    elseif bc.type == FIXED_DEPTH || bc.type == TIDAL
        depth = get_boundary_value(bc, t)
        apply_fixed_depth_boundaries!(state, depth, bc.sides)
    elseif bc.type == INFLOW
        if bc.hydrograph !== nothing
            apply_inflow_boundaries!(state, bc.hydrograph, t, bc.sides)
        end
    end
    nothing
end

"""
    apply_boundaries!(state::SimulationState, bc::BoundaryCondition, t::Real, grid::Grid)

Apply time-varying boundary conditions with grid for flux calculations.
"""
function apply_boundaries!(state::SimulationState{T}, bc::BoundaryCondition{T},
                           t::Real, grid::Grid{T}) where T
    if bc.type == CLOSED
        apply_closed_boundaries!(state, bc.sides)
    elseif bc.type == OPEN
        outflow = apply_open_boundaries!(state, bc.sides, grid)
        return outflow
    elseif bc.type == FIXED_DEPTH || bc.type == TIDAL
        depth = get_boundary_value(bc, t)
        apply_fixed_depth_boundaries!(state, depth, bc.sides)
    elseif bc.type == INFLOW
        if bc.hydrograph !== nothing
            apply_inflow_boundaries!(state, bc.hydrograph, t, bc.sides, grid)
        end
    elseif bc.type == RATING_CURVE
        if bc.rating_curve !== nothing
            outflow = apply_rating_curve_boundaries!(state, bc.rating_curve, bc.sides, grid)
            return outflow
        end
    end
    return zero(T)
end

"""
    apply_closed_boundaries!(state::SimulationState)

Apply closed (no-flow) boundary conditions.
"""
function apply_closed_boundaries!(state::SimulationState{T}) where T
    apply_closed_boundaries!(state, (true, true, true, true))
end

"""
    apply_closed_boundaries!(state::SimulationState, sides::NTuple{4,Bool})

Apply closed boundaries to specified sides (left, right, bottom, top).
"""
function apply_closed_boundaries!(state::SimulationState{T},
                                   sides::NTuple{4,Bool}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides

    if left
        @inbounds for j in 1:ny
            state.qx[1, j] = zero(T)
        end
    end

    if right
        @inbounds for j in 1:ny
            state.qx[nx, j] = zero(T)
        end
    end

    if bottom
        @inbounds for i in 1:nx
            state.qy[i, 1] = zero(T)
        end
    end

    if top
        @inbounds for i in 1:nx
            state.qy[i, ny] = zero(T)
        end
    end

    nothing
end

"""
    apply_open_boundaries!(state::SimulationState)

Apply open (transmissive) boundary conditions.
"""
function apply_open_boundaries!(state::SimulationState{T}) where T
    apply_open_boundaries!(state, (true, true, true, true))
end

"""
    apply_open_boundaries!(state::SimulationState, sides::NTuple{4,Bool})

Apply open boundaries to specified sides.
"""
function apply_open_boundaries!(state::SimulationState{T},
                                 sides::NTuple{4,Bool}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides

    if left
        @inbounds for j in 1:ny
            state.qx[1, j] = state.qx[2, j]
            if state.qx[1, j] > 0
                state.qx[1, j] = zero(T)
            end
        end
    end

    if right
        @inbounds for j in 1:ny
            state.qx[nx, j] = state.qx[nx-1, j]
            if state.qx[nx, j] < 0
                state.qx[nx, j] = zero(T)
            end
        end
    end

    if bottom
        @inbounds for i in 1:nx
            state.qy[i, 1] = state.qy[i, 2]
            if state.qy[i, 1] > 0
                state.qy[i, 1] = zero(T)
            end
        end
    end

    if top
        @inbounds for i in 1:nx
            state.qy[i, ny] = state.qy[i, ny-1]
            if state.qy[i, ny] < 0
                state.qy[i, ny] = zero(T)
            end
        end
    end

    nothing
end

"""
    apply_open_boundaries!(state::SimulationState, sides::NTuple{4,Bool}, grid::Grid)

Apply open boundaries and return total outflow volume rate (m³/s).
"""
function apply_open_boundaries!(state::SimulationState{T},
                                 sides::NTuple{4,Bool},
                                 grid::Grid{T}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides
    total_outflow = zero(T)

    if left
        @inbounds for j in 1:ny
            state.qx[1, j] = state.qx[2, j]
            if state.qx[1, j] > 0
                state.qx[1, j] = zero(T)
            else
                total_outflow -= state.qx[1, j] * grid.dy
            end
        end
    end

    if right
        @inbounds for j in 1:ny
            state.qx[nx, j] = state.qx[nx-1, j]
            if state.qx[nx, j] < 0
                state.qx[nx, j] = zero(T)
            else
                total_outflow += state.qx[nx, j] * grid.dy
            end
        end
    end

    if bottom
        @inbounds for i in 1:nx
            state.qy[i, 1] = state.qy[i, 2]
            if state.qy[i, 1] > 0
                state.qy[i, 1] = zero(T)
            else
                total_outflow -= state.qy[i, 1] * grid.dx
            end
        end
    end

    if top
        @inbounds for i in 1:nx
            state.qy[i, ny] = state.qy[i, ny-1]
            if state.qy[i, ny] < 0
                state.qy[i, ny] = zero(T)
            else
                total_outflow += state.qy[i, ny] * grid.dx
            end
        end
    end

    total_outflow
end

"""
    apply_fixed_depth_boundaries!(state::SimulationState, fixed_depth::Real)

Apply fixed water depth at all boundaries.
"""
function apply_fixed_depth_boundaries!(state::SimulationState{T},
                                        fixed_depth::Real) where T
    apply_fixed_depth_boundaries!(state, T(fixed_depth), (true, true, true, true))
end

"""
    apply_fixed_depth_boundaries!(state::SimulationState, fixed_depth::Real, sides::NTuple{4,Bool})

Apply fixed water depth at specified boundaries.
"""
function apply_fixed_depth_boundaries!(state::SimulationState{T},
                                        fixed_depth::T,
                                        sides::NTuple{4,Bool}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides

    if left
        @inbounds for j in 1:ny
            state.h[1, j] = fixed_depth
        end
    end

    if right
        @inbounds for j in 1:ny
            state.h[nx, j] = fixed_depth
        end
    end

    if bottom
        @inbounds for i in 1:nx
            state.h[i, 1] = fixed_depth
        end
    end

    if top
        @inbounds for i in 1:nx
            state.h[i, ny] = fixed_depth
        end
    end

    nothing
end

"""
    apply_inflow_boundaries!(state, hydrograph, t, sides)

Apply inflow hydrograph at boundaries.
"""
function apply_inflow_boundaries!(state::SimulationState{T},
                                   hydrograph::InflowHydrograph{T},
                                   t::Real,
                                   sides::NTuple{4,Bool}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides

    # Get current inflow flux (m²/s)
    q = inflow_flux(hydrograph, t)

    # Apply to boundaries
    if left
        @inbounds for j in 1:ny
            state.qx[1, j] = q  # Positive = inflow from left
        end
    end

    if right
        @inbounds for j in 1:ny
            state.qx[nx, j] = -q  # Negative = inflow from right
        end
    end

    if bottom
        @inbounds for i in 1:nx
            state.qy[i, 1] = q  # Positive = inflow from bottom
        end
    end

    if top
        @inbounds for i in 1:nx
            state.qy[i, ny] = -q  # Negative = inflow from top
        end
    end

    nothing
end

"""
    apply_inflow_boundaries!(state, hydrograph, t, sides, grid)

Apply inflow hydrograph and return inflow volume rate (m³/s).
"""
function apply_inflow_boundaries!(state::SimulationState{T},
                                   hydrograph::InflowHydrograph{T},
                                   t::Real,
                                   sides::NTuple{4,Bool},
                                   grid::Grid{T}) where T
    apply_inflow_boundaries!(state, hydrograph, t, sides)
    return inflow_discharge(hydrograph, t)
end

"""
    apply_rating_curve_boundaries!(state, curve, sides, grid)

Apply rating curve boundary conditions and return outflow (m³/s).
"""
function apply_rating_curve_boundaries!(state::SimulationState{T},
                                         curve::RatingCurve{T},
                                         sides::NTuple{4,Bool},
                                         grid::Grid{T}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides
    total_outflow = zero(T)

    if left
        @inbounds for j in 1:ny
            Q = rating_discharge(curve, state.h[1, j])
            q = Q / grid.dy  # Flux per unit width
            state.qx[1, j] = -q  # Outflow (negative x direction)
            total_outflow += Q
        end
    end

    if right
        @inbounds for j in 1:ny
            Q = rating_discharge(curve, state.h[nx, j])
            q = Q / grid.dy
            state.qx[nx, j] = q  # Outflow (positive x direction)
            total_outflow += Q
        end
    end

    if bottom
        @inbounds for i in 1:nx
            Q = rating_discharge(curve, state.h[i, 1])
            q = Q / grid.dx
            state.qy[i, 1] = -q  # Outflow (negative y direction)
            total_outflow += Q
        end
    end

    if top
        @inbounds for i in 1:nx
            Q = rating_discharge(curve, state.h[i, ny])
            q = Q / grid.dx
            state.qy[i, ny] = q  # Outflow (positive y direction)
            total_outflow += Q
        end
    end

    total_outflow
end

# =============================================================================
# Positive Depth Enforcement
# =============================================================================

"""
    enforce_positive_depth!(state::SimulationState, h_min)

Enforce non-negative water depths (with small threshold).
"""
function enforce_positive_depth!(state::SimulationState{T}, h_min::T) where T
    negative_count = 0

    @inbounds for j in axes(state.h, 2), i in axes(state.h, 1)
        if state.h[i, j] < zero(T)
            negative_count += 1
            state.h[i, j] = zero(T)
        elseif state.h[i, j] < h_min
            # Keep very small depths but zero out discharge
            state.qx[i, j] = zero(T)
            state.qy[i, j] = zero(T)
        end
    end

    if negative_count > 0
        @warn "Corrected $negative_count negative depth cells"
    end

    nothing
end
