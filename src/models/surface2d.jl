# HydroForge 2D Surface Flow Solver
# Main solver implementation

# =============================================================================
# Results Accumulator (used by run_simulation!)
# =============================================================================

"""
    ResultsAccumulator{T}

Accumulates simulation results over time.
"""
mutable struct ResultsAccumulator{T<:AbstractFloat}
    max_depth::Matrix{T}
    arrival_time::Matrix{T}
    max_velocity::Matrix{T}
    point_hydrographs::Dict{Tuple{Int,Int}, Vector{Tuple{T,T}}}
    output_points::Vector{Tuple{Int,Int}}
    arrival_threshold::T
end

"""
    ResultsAccumulator(grid::Grid{T}, output_points) where T

Create results accumulator for given grid and output points.
"""
function ResultsAccumulator(grid::Grid{T}, output_points::Vector{Tuple{Int,Int}};
                            arrival_threshold::T=T(0.01)) where T
    nx, ny = grid.nx, grid.ny
    ResultsAccumulator{T}(
        zeros(T, nx, ny),           # max_depth
        fill(T(Inf), nx, ny),       # arrival_time (Inf = never arrived)
        zeros(T, nx, ny),           # max_velocity
        Dict(pt => Tuple{T,T}[] for pt in output_points),
        output_points,
        arrival_threshold
    )
end

"""
    update_results!(results::ResultsAccumulator, state::SimulationState)

Update accumulated results with current state.
"""
function update_results!(results::ResultsAccumulator{T}, state::SimulationState{T}) where T
    h = state.h
    qx = state.qx
    qy = state.qy
    t = state.t
    h_min = results.arrival_threshold

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        # Update max depth
        if h[i, j] > results.max_depth[i, j]
            results.max_depth[i, j] = h[i, j]
        end

        # Update arrival time
        if h[i, j] > results.arrival_threshold && results.arrival_time[i, j] == T(Inf)
            results.arrival_time[i, j] = t
        end

        # Update max velocity
        if h[i, j] > h_min
            u = qx[i, j] / h[i, j]
            v = qy[i, j] / h[i, j]
            vel = sqrt(u^2 + v^2)
            if vel > results.max_velocity[i, j]
                results.max_velocity[i, j] = vel
            end
        end
    end

    nothing
end

"""
    record_output!(results::ResultsAccumulator, state::SimulationState)

Record point hydrograph values at current time.
"""
function record_output!(results::ResultsAccumulator{T}, state::SimulationState{T}) where T
    for (i, j) in results.output_points
        push!(results.point_hydrographs[(i, j)], (state.t, state.h[i, j]))
    end
    nothing
end

# =============================================================================
# Workspace Types
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
    step!(state, topo, params, rainfall, dt, work; infiltration=nothing, infil_state=nothing)

Perform a single simulation timestep.

# Arguments
- `state`: Current simulation state (modified in-place)
- `topo`: Topography data
- `params`: Simulation parameters
- `rainfall`: Rainfall event
- `dt`: Timestep size
- `work`: Workspace arrays for intermediate calculations
- `infiltration`: Optional infiltration parameters
- `infil_state`: Optional infiltration state for cumulative tracking

# Returns
- `infiltrated`: Volume infiltrated this timestep (m³) or 0 if no infiltration
"""
function step!(state::SimulationState{T}, topo::Topography{T},
               params::SimulationParameters{T}, rainfall::RainfallEvent{T},
               dt::T, work::SimulationWorkspace{T};
               infiltration::Union{InfiltrationParameters{T}, Nothing}=nothing,
               infil_state::Union{InfiltrationState{T}, Nothing}=nothing) where T

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

    # 5. Apply infiltration losses (if enabled)
    infiltrated = zero(T)
    if infiltration !== nothing
        if infil_state !== nothing
            # Green-Ampt with cumulative tracking
            infiltrated_depth = apply_infiltration!(h, infil_state, infiltration, dt)
            infiltrated = infiltrated_depth * cell_area(grid)
        else
            # Simple constant-rate infiltration
            apply_infiltration!(h, infiltration, dt)
            infiltrated = infiltration.hydraulic_conductivity * dt * cell_area(grid) * grid.nx * grid.ny
        end
    end

    # 6. Update discharge arrays
    copyto!(qx, work.qx_new)
    copyto!(qy, work.qy_new)

    # 7. Apply boundary conditions
    apply_boundaries!(state, CLOSED)

    # 8. Enforce positive depths
    enforce_positive_depth!(state, params.h_min)

    # 9. Advance time
    state.t += dt

    infiltrated
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
    SimulationResults{T}

Complete simulation results including accumulated data and mass balance.
"""
struct SimulationResults{T<:AbstractFloat}
    accumulator::ResultsAccumulator{T}
    mass_balance::MassBalance{T}
    infil_state::Union{InfiltrationState{T}, Nothing}
    step_count::Int
    wall_time::Float64
end

"""
    log_progress(state, params, step_count, mass_balance; verbosity=1)

Log simulation progress.

# Arguments
- `state`: Current simulation state
- `params`: Simulation parameters
- `step_count`: Number of timesteps completed
- `mass_balance`: Mass balance tracker
- `verbosity`: 0=silent, 1=basic, 2=detailed
"""
function log_progress(state::SimulationState{T}, params::SimulationParameters{T},
                      step_count::Int, mass_balance::MassBalance{T};
                      verbosity::Int=1) where T
    if verbosity < 1
        return
    end

    progress = state.t / params.t_end * 100
    max_h = maximum(state.h)

    if verbosity == 1
        @info "Progress" t=round(state.t, digits=1) progress=round(progress, digits=1) max_depth=round(max_h, digits=4)
    else
        rel_error = relative_mass_error(mass_balance)
        @info "Progress" t=round(state.t, digits=1) progress=round(progress, digits=1) max_depth=round(max_h, digits=4) steps=step_count mass_error_pct=round(rel_error*100, digits=4)
    end
end

"""
    run_simulation!(state, scenario; progress_callback=nothing, log_interval=nothing, verbosity=1)

Run the full simulation.

# Arguments
- `state`: Initial state (modified in-place)
- `scenario`: Complete scenario specification
- `progress_callback`: Optional callback(fraction) for progress updates
- `log_interval`: Time interval for progress logging (seconds). Nothing = no logging.
- `verbosity`: Logging verbosity (0=silent, 1=basic, 2=detailed)

# Returns
- `SimulationResults`: Complete results including accumulator and mass balance
"""
function run_simulation!(state::SimulationState{T}, scenario::Scenario{T};
                         progress_callback=nothing,
                         log_interval::Union{T, Nothing}=nothing,
                         verbosity::Int=1) where T
    params = scenario.parameters
    topo = scenario.topography
    rainfall = scenario.rainfall
    grid = scenario.grid
    infiltration = scenario.infiltration

    # Create workspace
    work = SimulationWorkspace(grid)

    # Create results accumulator
    results = ResultsAccumulator(grid, scenario.output_points)

    # Create infiltration state if needed
    infil_state = if infiltration !== nothing
        InfiltrationState(grid)
    else
        nothing
    end

    # Initialize mass balance
    mass_balance = MassBalance(state, grid)

    # Simulation loop
    step_count = 0
    last_output_time = state.t
    last_log_time = state.t
    start_wall_time = time()

    while state.t < params.t_end
        # Compute stable timestep
        dt = compute_dt(state, grid, params)

        # Don't overshoot end time
        if state.t + dt > params.t_end
            dt = params.t_end - state.t
        end

        # Track rainfall volume added this step
        rainfall_rate_m_s = rainfall_rate_ms(rainfall, state.t)
        rainfall_volume = rainfall_rate_m_s * dt * total_area(grid)
        add_rainfall!(mass_balance, rainfall_volume)

        # Perform timestep (returns infiltrated volume)
        infiltrated = step!(state, topo, params, rainfall, dt, work;
                           infiltration=infiltration, infil_state=infil_state)
        step_count += 1

        # Track infiltration in mass balance
        if infiltrated > zero(T)
            add_infiltration!(mass_balance, infiltrated)
        end

        # Update current volume in mass balance
        update_volume!(mass_balance, state, grid)

        # Update results
        update_results!(results, state)

        # Progress callback
        if progress_callback !== nothing
            progress_callback(state.t / params.t_end)
        end

        # Progress logging
        if log_interval !== nothing && state.t - last_log_time >= log_interval
            log_progress(state, params, step_count, mass_balance; verbosity=verbosity)
            last_log_time = state.t
        end

        # Periodic output
        if state.t - last_output_time >= params.output_interval
            record_output!(results, state)
            last_output_time = state.t
        end
    end

    wall_time = time() - start_wall_time

    # Final logging
    if verbosity >= 1
        @info "Simulation complete" steps=step_count wall_time=round(wall_time, digits=2) mass_error_pct=round(relative_mass_error(mass_balance)*100, digits=4)
    end

    SimulationResults{T}(results, mass_balance, infil_state, step_count, wall_time)
end

# Backward-compatible version returning just the accumulator
"""
    run_simulation_simple!(state, scenario; progress_callback=nothing)

Run simulation and return only the ResultsAccumulator (backward compatible).
"""
function run_simulation_simple!(state::SimulationState{T}, scenario::Scenario{T};
                                progress_callback=nothing) where T
    result = run_simulation!(state, scenario; progress_callback=progress_callback, verbosity=0)
    result.accumulator
end
