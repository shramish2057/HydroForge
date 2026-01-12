# HydroForge Timestep Module
# CFL-based adaptive timestepping

"""
    TimestepController{T}

Manages adaptive timestepping with smoothing and history tracking.

# Fields
- `dt_history::Vector{T}`: Recent timestep values
- `history_size::Int`: Maximum history size
- `smoothing_factor::T`: Smoothing for timestep changes (0.5-0.9)
- `min_dt_warning::T`: Warn if dt falls below this value
- `warning_issued::Bool`: Track if warning was issued
"""
mutable struct TimestepController{T<:AbstractFloat}
    dt_history::Vector{T}
    history_size::Int
    smoothing_factor::T
    min_dt_warning::T
    warning_issued::Bool
end

"""
    TimestepController(T=Float64; history_size=10, smoothing_factor=0.7, min_dt_warning=0.001)

Create a timestep controller with default settings.
"""
function TimestepController(::Type{T}=Float64;
                            history_size::Int=10,
                            smoothing_factor::Real=0.7,
                            min_dt_warning::Real=0.001) where T<:AbstractFloat
    TimestepController{T}(
        T[],
        history_size,
        T(smoothing_factor),
        T(min_dt_warning),
        false
    )
end

"""
    compute_dt_smooth!(controller::TimestepController, dt_raw::T) where T

Apply smoothing to prevent sudden timestep changes.
Returns smoothed timestep and updates history.
"""
function compute_dt_smooth!(controller::TimestepController{T}, dt_raw::T) where T
    # Add to history
    push!(controller.dt_history, dt_raw)
    if length(controller.dt_history) > controller.history_size
        popfirst!(controller.dt_history)
    end

    # If first timestep, no smoothing
    if length(controller.dt_history) == 1
        return dt_raw
    end

    # Smooth with previous value
    dt_prev = controller.dt_history[end-1]
    α = controller.smoothing_factor

    # Don't allow more than 50% reduction in single step
    dt_min_allowed = dt_prev * 0.5
    dt_candidate = α * dt_prev + (1 - α) * dt_raw

    dt_smooth = max(dt_candidate, dt_min_allowed)

    # But always respect CFL (dt_raw is the CFL limit)
    dt_final = min(dt_smooth, dt_raw)

    # Issue warning if very small
    if dt_final < controller.min_dt_warning && !controller.warning_issued
        @warn "Timestep has become very small" dt=dt_final min_warning=controller.min_dt_warning
        controller.warning_issued = true
    end

    dt_final
end

"""
    reset!(controller::TimestepController)

Reset the timestep controller history.
"""
function reset!(controller::TimestepController{T}) where T
    empty!(controller.dt_history)
    controller.warning_issued = false
    controller
end

"""
    compute_dt(state::SimulationState, grid::Grid, params::SimulationParameters)

Compute stable timestep using CFL condition.

dt ≤ CFL × min(dx, dy) / max(√(gh) + |u|)

# Returns
- `dt`: Safe timestep (s), clamped to dt_max
"""
function compute_dt(state::SimulationState{T}, grid::Grid{T},
                    params::SimulationParameters{T}) where T
    h = state.h
    qx = state.qx
    qy = state.qy
    g = params.g
    h_min = params.h_min
    cfl = params.cfl

    min_dx = min(grid.dx, grid.dy)
    max_wave_speed = zero(T)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > h_min
            # Wave celerity
            c = sqrt(g * h[i, j])

            # Velocity magnitude
            u = qx[i, j] / h[i, j]
            v = qy[i, j] / h[i, j]

            # Maximum wave speed in either direction
            wave_x = c + abs(u)
            wave_y = c + abs(v)
            max_wave_speed = max(max_wave_speed, wave_x, wave_y)
        end
    end

    if max_wave_speed > zero(T)
        dt = cfl * min_dx / max_wave_speed
    else
        dt = params.dt_max
    end

    # Clamp to maximum timestep
    min(dt, params.dt_max)
end

"""
    compute_dt_array(h, qx, qy, dx, dy, g, cfl, h_min)

Low-level timestep computation for benchmarking.
"""
function compute_dt_array(h::Matrix{T}, qx::Matrix{T}, qy::Matrix{T},
                          dx::T, dy::T, g::T, cfl::T, h_min::T) where T
    min_dx = min(dx, dy)
    max_wave_speed = zero(T)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > h_min
            c = sqrt(g * h[i, j])
            u = qx[i, j] / h[i, j]
            v = qy[i, j] / h[i, j]
            wave_x = c + abs(u)
            wave_y = c + abs(v)
            max_wave_speed = max(max_wave_speed, wave_x, wave_y)
        end
    end

    if max_wave_speed > zero(T)
        cfl * min_dx / max_wave_speed
    else
        T(Inf)
    end
end

"""
    check_cfl(state::SimulationState, grid::Grid, params::SimulationParameters, dt::Real)

Check if given timestep satisfies CFL condition.

# Returns
- `Bool`: true if dt is stable
"""
function check_cfl(state::SimulationState{T}, grid::Grid{T},
                   params::SimulationParameters{T}, dt::Real) where T
    stable_dt = compute_dt(state, grid, params)
    dt <= stable_dt
end
