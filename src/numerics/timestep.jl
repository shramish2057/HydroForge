# HydroForge Timestep Module
# CFL-based adaptive timestepping

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
