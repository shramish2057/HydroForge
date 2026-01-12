# HydroForge Parallel Numerical Kernels
#
# Threaded and GPU-ready implementations of core numerical operations.
# These kernels are optimized for parallel execution on multi-core CPUs
# and can be extended for GPU execution.

using Base.Threads: @threads, nthreads

# =============================================================================
# Threaded Flux Computation
# =============================================================================

"""
    compute_flux_x_threaded!(qx_new, qx, h, z, n, grid, params, dt)

Compute x-direction fluxes using multi-threading.
Parallelizes over rows (j index) for better cache locality.
"""
function compute_flux_x_threaded!(qx_new::Matrix{T}, qx::Matrix{T}, h::Matrix{T},
                                   z::Matrix{T}, n::Matrix{T}, grid::Grid{T},
                                   params::SimulationParameters{T}, dt::T) where T
    g = params.g
    h_min = params.h_min
    dx = grid.dx
    nx, ny = grid.nx, grid.ny
    max_froude = T(0.9)

    @threads for j in 1:ny
        @inbounds for i in 1:nx-1
            # Face depth using upwind reconstruction
            η_L = h[i, j] + z[i, j]
            η_R = h[i+1, j] + z[i+1, j]
            z_face = max(z[i, j], z[i+1, j])
            η_face = max(η_L, η_R)
            h_f = max(η_face - z_face, zero(T))

            if h_f > h_min
                # Water surface gradient
                dη_dx = (η_R - η_L) / dx

                # Face roughness (average)
                n_f = (n[i, j] + n[i+1, j]) / 2

                # Current flux at face (interpolated)
                q_f = (qx[i, j] + qx[i+1, j]) / 2

                # Friction factor (semi-implicit)
                D = one(T) + g * dt * n_f^2 * abs(q_f) / h_f^(T(10)/T(3))

                # New flux from momentum equation
                q_new = (q_f - g * h_f * dt * dη_dx) / D

                # Apply Froude limiting
                if h_f > zero(T)
                    v_max = max_froude * sqrt(g * h_f)
                    q_max = h_f * v_max
                    if abs(q_new) > q_max
                        q_new = sign(q_new) * q_max
                    end
                end

                qx_new[i, j] = q_new
            else
                qx_new[i, j] = zero(T)
            end
        end
        # Boundary (no flow)
        @inbounds qx_new[nx, j] = zero(T)
    end

    nothing
end

"""
    compute_flux_y_threaded!(qy_new, qy, h, z, n, grid, params, dt)

Compute y-direction fluxes using multi-threading.
Parallelizes over rows (j index).
"""
function compute_flux_y_threaded!(qy_new::Matrix{T}, qy::Matrix{T}, h::Matrix{T},
                                   z::Matrix{T}, n::Matrix{T}, grid::Grid{T},
                                   params::SimulationParameters{T}, dt::T) where T
    g = params.g
    h_min = params.h_min
    dy = grid.dy
    nx, ny = grid.nx, grid.ny
    max_froude = T(0.9)

    @threads for j in 1:ny-1
        @inbounds for i in 1:nx
            # Face depth using upwind reconstruction
            η_B = h[i, j] + z[i, j]
            η_T = h[i, j+1] + z[i, j+1]
            z_face = max(z[i, j], z[i, j+1])
            η_face = max(η_B, η_T)
            h_f = max(η_face - z_face, zero(T))

            if h_f > h_min
                # Water surface gradient
                dη_dy = (η_T - η_B) / dy

                # Face roughness (average)
                n_f = (n[i, j] + n[i, j+1]) / 2

                # Current flux at face (interpolated)
                q_f = (qy[i, j] + qy[i, j+1]) / 2

                # Friction factor (semi-implicit)
                D = one(T) + g * dt * n_f^2 * abs(q_f) / h_f^(T(10)/T(3))

                # New flux from momentum equation
                q_new = (q_f - g * h_f * dt * dη_dy) / D

                # Apply Froude limiting
                if h_f > zero(T)
                    v_max = max_froude * sqrt(g * h_f)
                    q_max = h_f * v_max
                    if abs(q_new) > q_max
                        q_new = sign(q_new) * q_max
                    end
                end

                qy_new[i, j] = q_new
            else
                qy_new[i, j] = zero(T)
            end
        end
    end

    # Boundary (no flow) - single thread is fine for 1D array
    @inbounds for i in 1:nx
        qy_new[i, ny] = zero(T)
    end

    nothing
end

# =============================================================================
# Threaded Depth Update
# =============================================================================

"""
    update_depth_threaded!(h, qx, qy, grid, dt)

Update water depths from flux divergence using multi-threading.
"""
function update_depth_threaded!(h::Matrix{T}, qx::Matrix{T}, qy::Matrix{T},
                                 grid::Grid{T}, dt::T) where T
    dx = grid.dx
    dy = grid.dy
    nx, ny = size(h)

    @threads for j in 1:ny
        @inbounds for i in 1:nx
            # x-direction flux divergence
            qx_east = i < nx ? qx[i, j] : zero(T)
            qx_west = i > 1 ? qx[i-1, j] : zero(T)
            div_qx = (qx_east - qx_west) / dx

            # y-direction flux divergence
            qy_north = j < ny ? qy[i, j] : zero(T)
            qy_south = j > 1 ? qy[i, j-1] : zero(T)
            div_qy = (qy_north - qy_south) / dy

            # Update depth
            h[i, j] -= dt * (div_qx + div_qy)
        end
    end

    nothing
end

# =============================================================================
# Threaded Results Update
# =============================================================================

"""
    update_results_threaded!(results, state, dt; g=9.81)

Update accumulated results with current state using multi-threading.
"""
function update_results_threaded!(results::ResultsAccumulator{T}, state::SimulationState{T},
                                   dt::T=zero(T); g::T=T(9.81)) where T
    h = state.h
    qx = state.qx
    qy = state.qy
    t = state.t
    h_min = results.arrival_threshold

    nx, ny = size(h)

    @threads for j in 1:ny
        @inbounds for i in 1:nx
            depth = h[i, j]

            # Update max depth
            if depth > results.max_depth[i, j]
                results.max_depth[i, j] = depth
            end

            # Update arrival time
            if depth > h_min && results.arrival_time[i, j] == T(Inf)
                results.arrival_time[i, j] = t
            end

            # Velocity-based metrics for wet cells
            if depth > h_min
                u = qx[i, j] / depth
                v = qy[i, j] / depth
                vel = sqrt(u^2 + v^2)

                # Max velocity
                if vel > results.max_velocity[i, j]
                    results.max_velocity[i, j] = vel
                end

                # Max hazard rating (h × v)
                hv = depth * vel
                if hv > results.max_hazard[i, j]
                    results.max_hazard[i, j] = hv
                end

                # Max Froude number
                fr = vel / sqrt(g * depth)
                if fr > results.max_froude[i, j]
                    results.max_froude[i, j] = fr
                end

                # Duration tracking
                results.total_duration[i, j] += dt
                results.last_wet[i, j] = true
            else
                results.last_wet[i, j] = false
            end
        end
    end

    nothing
end

# =============================================================================
# Threaded Wet/Dry Limiting
# =============================================================================

"""
    limit_flux_wetdry_threaded!(qx, qy, h, h_min)

Limit fluxes at wet/dry interfaces using multi-threading.
"""
function limit_flux_wetdry_threaded!(qx::Matrix{T}, qy::Matrix{T}, h::Matrix{T}, h_min::T) where T
    nx, ny = size(h)
    h_transition = T(2) * h_min

    # Helper function for wet_dry_factor
    @inline function wdf(depth::T)
        if depth <= h_min
            zero(T)
        elseif depth >= h_min + h_transition
            one(T)
        else
            x = (depth - h_min) / h_transition
            x * x * (T(3) - T(2) * x)
        end
    end

    # Limit x-direction fluxes
    @threads for j in 1:ny
        @inbounds for i in 1:nx-1
            factor_L = wdf(h[i, j])
            factor_R = wdf(h[i+1, j])
            factor = min(factor_L, factor_R)
            qx[i, j] *= factor
        end
    end

    # Limit y-direction fluxes
    @threads for j in 1:ny-1
        @inbounds for i in 1:nx
            factor_B = wdf(h[i, j])
            factor_T = wdf(h[i, j+1])
            factor = min(factor_B, factor_T)
            qy[i, j] *= factor
        end
    end

    nothing
end

# =============================================================================
# Threaded Timestep Computation
# =============================================================================

"""
    compute_dt_threaded(state, grid, params)

Compute stable timestep using multi-threading for wave speed calculation.
"""
function compute_dt_threaded(state::SimulationState{T}, grid::Grid{T},
                              params::SimulationParameters{T}) where T
    h = state.h
    qx = state.qx
    qy = state.qy
    g = params.g
    h_min = params.h_min
    cfl = params.cfl

    nx, ny = size(h)
    nt = nthreads()
    chunk_size = cld(ny, nt)

    # Per-thread maximum wave speed
    max_speeds = zeros(T, nt)

    @threads for tid in 1:nt
        j_start = (tid - 1) * chunk_size + 1
        j_end = min(tid * chunk_size, ny)

        local_max = zero(T)

        @inbounds for j in j_start:j_end
            for i in 1:nx
                depth = h[i, j]
                if depth > h_min
                    # Wave speed
                    c = sqrt(g * depth)

                    # Velocity
                    u = abs(qx[i, j] / depth)
                    v = abs(qy[i, j] / depth)

                    # Maximum characteristic speed
                    speed_x = u + c
                    speed_y = v + c

                    local_max = max(local_max, speed_x, speed_y)
                end
            end
        end

        max_speeds[tid] = local_max
    end

    max_speed = maximum(max_speeds)

    # Compute timestep
    if max_speed > zero(T)
        dx_min = min(grid.dx, grid.dy)
        dt = cfl * dx_min / max_speed
        return min(dt, params.dt_max)
    else
        return params.dt_max
    end
end

# =============================================================================
# Threaded Rainfall Application
# =============================================================================

"""
    apply_rainfall_threaded!(h, rainfall, t, dt)

Apply rainfall uniformly using multi-threading.
"""
function apply_rainfall_threaded!(h::Matrix{T}, rainfall::RainfallEvent{T},
                                   t::T, dt::T) where T
    rate = rainfall_rate_ms(rainfall, t)

    if rate > zero(T)
        @threads for j in axes(h, 2)
            @inbounds for i in axes(h, 1)
                h[i, j] += rate * dt
            end
        end
    end

    nothing
end

# =============================================================================
# Threaded Boundary Conditions
# =============================================================================

"""
    enforce_positive_depth_threaded!(state, h_min)

Enforce positive depths using multi-threading.
"""
function enforce_positive_depth_threaded!(state::SimulationState{T}, h_min::T) where T
    h = state.h
    qx = state.qx
    qy = state.qy

    @threads for j in axes(h, 2)
        @inbounds for i in axes(h, 1)
            if h[i, j] < zero(T)
                h[i, j] = zero(T)
                qx[i, j] = zero(T)
                qy[i, j] = zero(T)
            elseif h[i, j] < h_min
                qx[i, j] = zero(T)
                qy[i, j] = zero(T)
            end
        end
    end

    nothing
end

# =============================================================================
# Threaded Surface Gradient
# =============================================================================

"""
    surface_gradient_threaded!(dη_dx, dη_dy, η, grid)

Compute water surface gradients using multi-threading.
"""
function surface_gradient_threaded!(dη_dx::Matrix{T}, dη_dy::Matrix{T},
                                     η::Matrix{T}, grid::Grid{T}) where T
    nx, ny = size(η)
    dx = grid.dx
    dy = grid.dy

    # Interior points: central differences (parallel)
    @threads for j in 2:ny-1
        @inbounds for i in 2:nx-1
            dη_dx[i, j] = (η[i+1, j] - η[i-1, j]) / (T(2) * dx)
            dη_dy[i, j] = (η[i, j+1] - η[i, j-1]) / (T(2) * dy)
        end
    end

    # Boundaries (serial - small portion)
    @inbounds for j in 1:ny
        dη_dx[1, j] = (η[2, j] - η[1, j]) / dx
        dη_dx[nx, j] = (η[nx, j] - η[nx-1, j]) / dx
    end

    @inbounds for i in 1:nx
        dη_dy[i, 1] = (η[i, 2] - η[i, 1]) / dy
        dη_dy[i, ny] = (η[i, ny] - η[i, ny-1]) / dy
    end

    @inbounds for j in [1, ny], i in 2:nx-1
        dη_dx[i, j] = (η[i+1, j] - η[i-1, j]) / (T(2) * dx)
    end

    @inbounds for i in [1, nx], j in 2:ny-1
        dη_dy[i, j] = (η[i, j+1] - η[i, j-1]) / (T(2) * dy)
    end

    nothing
end

# =============================================================================
# Threaded Water Surface Elevation
# =============================================================================

"""
    water_surface_elevation_threaded!(η, h, z)

Compute water surface elevation in-place using multi-threading.
"""
function water_surface_elevation_threaded!(η::Matrix{T}, h::Matrix{T}, z::Matrix{T}) where T
    @threads for j in axes(h, 2)
        @inbounds for i in axes(h, 1)
            η[i, j] = h[i, j] + z[i, j]
        end
    end
    nothing
end

# =============================================================================
# Threaded Copy
# =============================================================================

"""
    copyto_threaded!(dst, src)

Copy array using multi-threading.
"""
function copyto_threaded!(dst::Matrix{T}, src::Matrix{T}) where T
    @threads for j in axes(src, 2)
        @inbounds for i in axes(src, 1)
            dst[i, j] = src[i, j]
        end
    end
    nothing
end
