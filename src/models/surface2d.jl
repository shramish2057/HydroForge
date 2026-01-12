# HydroForge 2D Surface Flow Solver
# Main solver implementation

# =============================================================================
# Types (must be defined before functions that use them)
# =============================================================================

"""
    SimulationWorkspace{T}

Preallocated arrays for simulation.
"""
struct SimulationWorkspace{T<:AbstractFloat}
    grid::Grid{T}
    qx_new::Matrix{T}
    qy_new::Matrix{T}
    η::Matrix{T}
end

"""
    SimulationWorkspace(grid::Grid{T}) where T

Create workspace for simulation.
"""
function SimulationWorkspace(grid::Grid{T}) where T
    nx, ny = grid.nx, grid.ny
    SimulationWorkspace{T}(
        grid,
        zeros(T, nx, ny),
        zeros(T, nx, ny),
        zeros(T, nx, ny)
    )
end

# =============================================================================
# Core Functions
# =============================================================================

"""
    step!(state, topo, params, rainfall, dt, work)

Perform a single simulation timestep.

# Arguments
- `state`: Current simulation state (modified in-place)
- `topo`: Topography data
- `params`: Simulation parameters
- `rainfall`: Rainfall event
- `dt`: Timestep size
- `work`: Workspace arrays for intermediate calculations
"""
function step!(state::SimulationState{T}, topo::Topography{T},
               params::SimulationParameters{T}, rainfall::RainfallEvent{T},
               dt::T, work::SimulationWorkspace{T}) where T

    grid = work.grid
    h = state.h
    qx = state.qx
    qy = state.qy
    z = topo.elevation
    n = topo.roughness

    # 1. Compute x-direction fluxes
    compute_flux_x!(work.qx_new, qx, h, z, n, grid, params, dt)

    # 2. Compute y-direction fluxes
    compute_flux_y!(work.qy_new, qy, h, z, n, grid, params, dt)

    # 3. Update water depths from flux divergence
    update_depth!(h, work.qx_new, work.qy_new, grid, dt)

    # 4. Apply rainfall source
    apply_rainfall!(h, rainfall, state.t, dt)

    # 5. Update discharge arrays
    copyto!(qx, work.qx_new)
    copyto!(qy, work.qy_new)

    # 6. Apply boundary conditions
    apply_boundaries!(state, CLOSED)

    # 7. Enforce positive depths
    enforce_positive_depth!(state, params.h_min)

    # 8. Advance time
    state.t += dt

    nothing
end

"""
    update_depth!(h, qx, qy, grid, dt)

Update water depths from flux divergence.

∂h/∂t + ∂qx/∂x + ∂qy/∂y = 0

h_new = h - dt * (∂qx/∂x + ∂qy/∂y)
"""
function update_depth!(h::Matrix{T}, qx::Matrix{T}, qy::Matrix{T},
                       grid::Grid{T}, dt::T) where T
    dx = grid.dx
    dy = grid.dy
    nx, ny = size(h)

    @inbounds for j in 1:ny
        for i in 1:nx
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

"""
    run_simulation!(state, scenario; progress_callback=nothing)

Run the full simulation.

# Arguments
- `state`: Initial state (modified in-place)
- `scenario`: Complete scenario specification
- `progress_callback`: Optional callback(fraction) for progress updates

# Returns
- `ResultsAccumulator`: Accumulated results
"""
function run_simulation!(state::SimulationState{T}, scenario::Scenario{T};
                         progress_callback=nothing) where T
    params = scenario.parameters
    topo = scenario.topography
    rainfall = scenario.rainfall
    grid = scenario.grid

    # Create workspace
    work = SimulationWorkspace(grid)

    # Create results accumulator
    results = ResultsAccumulator(grid, scenario.output_points)

    # Simulation loop
    step_count = 0
    last_output_time = state.t

    while state.t < params.t_end
        # Compute stable timestep
        dt = compute_dt(state, grid, params)

        # Don't overshoot end time
        if state.t + dt > params.t_end
            dt = params.t_end - state.t
        end

        # Perform timestep
        step!(state, topo, params, rainfall, dt, work)
        step_count += 1

        # Update results
        update_results!(results, state)

        # Progress callback
        if progress_callback !== nothing
            progress_callback(state.t / params.t_end)
        end

        # Periodic output
        if state.t - last_output_time >= params.output_interval
            record_output!(results, state)
            last_output_time = state.t
        end
    end

    results
end
