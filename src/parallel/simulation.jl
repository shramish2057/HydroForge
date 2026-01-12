# HydroForge Parallel Simulation
#
# Parallel implementations of the main simulation loop.
# Automatically selects the appropriate backend based on current settings.

using Base.Threads: nthreads
using Printf: @sprintf

# =============================================================================
# Parallel Simulation Workspace
# =============================================================================

"""
    ParallelWorkspace{T}

Extended workspace for parallel simulation with backend selection.
"""
struct ParallelWorkspace{T<:AbstractFloat}
    grid::Grid{T}
    qx_new::Matrix{T}
    qy_new::Matrix{T}
    η::Matrix{T}
    backend::ComputeBackend
end

"""
    ParallelWorkspace(grid::Grid{T}; backend=get_backend()) where T

Create parallel workspace with specified backend.
"""
function ParallelWorkspace(grid::Grid{T}; backend::ComputeBackend=get_backend()) where T
    nx, ny = grid.nx, grid.ny
    ParallelWorkspace{T}(
        grid,
        zeros(T, nx, ny),
        zeros(T, nx, ny),
        zeros(T, nx, ny),
        backend
    )
end

# =============================================================================
# Backend-Aware Step Function
# =============================================================================

"""
    step_parallel!(state, topo, params, rainfall, dt, work; infiltration=nothing, infil_state=nothing)

Perform a single simulation timestep using the appropriate parallel backend.
"""
function step_parallel!(state::SimulationState{T}, topo::Topography{T},
                        params::SimulationParameters{T}, rainfall::RainfallEvent{T},
                        dt::T, work::ParallelWorkspace{T};
                        infiltration::Union{InfiltrationParameters{T}, Nothing}=nothing,
                        infil_state::Union{InfiltrationState{T}, Nothing}=nothing) where T

    backend = work.backend
    grid = work.grid
    h = state.h
    qx = state.qx
    qy = state.qy
    z = topo.elevation
    n = topo.roughness

    if backend isa ThreadedBackend
        # Use threaded implementations
        _step_threaded!(state, topo, params, rainfall, dt, work, infiltration, infil_state)
    elseif backend isa GPUBackend
        # Use GPU implementations
        _step_gpu!(state, topo, params, rainfall, dt, work, infiltration, infil_state)
    else
        # Fall back to serial implementation
        _step_serial!(state, topo, params, rainfall, dt, work, infiltration, infil_state)
    end
end

"""
    _step_serial!(state, topo, params, rainfall, dt, work, infiltration, infil_state)

Serial (single-threaded) implementation of simulation step.
"""
function _step_serial!(state::SimulationState{T}, topo::Topography{T},
                       params::SimulationParameters{T}, rainfall::RainfallEvent{T},
                       dt::T, work::ParallelWorkspace{T},
                       infiltration, infil_state) where T

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

    # 3. Update water depths
    update_depth!(h, work.qx_new, work.qy_new, grid, dt)

    # 4. Apply rainfall
    apply_rainfall!(h, rainfall, state.t, dt)

    # 5. Apply infiltration
    infiltrated = zero(T)
    if infiltration !== nothing
        if infil_state !== nothing
            infiltrated_depth = apply_infiltration!(h, infil_state, infiltration, dt)
            infiltrated = infiltrated_depth * cell_area(grid)
        else
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
    _step_threaded!(state, topo, params, rainfall, dt, work, infiltration, infil_state)

Multi-threaded implementation of simulation step.
"""
function _step_threaded!(state::SimulationState{T}, topo::Topography{T},
                         params::SimulationParameters{T}, rainfall::RainfallEvent{T},
                         dt::T, work::ParallelWorkspace{T},
                         infiltration, infil_state) where T

    grid = work.grid
    h = state.h
    qx = state.qx
    qy = state.qy
    z = topo.elevation
    n = topo.roughness

    # 1. Compute x-direction fluxes (threaded)
    compute_flux_x_threaded!(work.qx_new, qx, h, z, n, grid, params, dt)

    # 2. Compute y-direction fluxes (threaded)
    compute_flux_y_threaded!(work.qy_new, qy, h, z, n, grid, params, dt)

    # 3. Update water depths (threaded)
    update_depth_threaded!(h, work.qx_new, work.qy_new, grid, dt)

    # 4. Apply rainfall (threaded)
    apply_rainfall_threaded!(h, rainfall, state.t, dt)

    # 5. Apply infiltration (serial - usually small overhead)
    infiltrated = zero(T)
    if infiltration !== nothing
        if infil_state !== nothing
            infiltrated_depth = apply_infiltration!(h, infil_state, infiltration, dt)
            infiltrated = infiltrated_depth * cell_area(grid)
        else
            apply_infiltration!(h, infiltration, dt)
            infiltrated = infiltration.hydraulic_conductivity * dt * cell_area(grid) * grid.nx * grid.ny
        end
    end

    # 6. Update discharge arrays (threaded)
    copyto_threaded!(qx, work.qx_new)
    copyto_threaded!(qy, work.qy_new)

    # 7. Apply boundary conditions
    apply_boundaries!(state, CLOSED)

    # 8. Enforce positive depths (threaded)
    enforce_positive_depth_threaded!(state, params.h_min)

    # 9. Advance time
    state.t += dt

    infiltrated
end

"""
    _step_gpu!(state, topo, params, rainfall, dt, work, infiltration, infil_state)

GPU implementation of simulation step.
"""
function _step_gpu!(state::SimulationState{T}, topo::Topography{T},
                    params::SimulationParameters{T}, rainfall::RainfallEvent{T},
                    dt::T, work::ParallelWorkspace{T},
                    infiltration, infil_state) where T

    if !GPU_AVAILABLE[]
        error("GPU backend not available. Load CUDA.jl first.")
    end

    # GPU implementation requires arrays to be on GPU
    # This is a simplified version - full implementation would keep arrays on GPU
    grid = work.grid
    nx, ny = grid.nx, grid.ny

    # For now, fall back to threaded if GPU arrays not set up
    @warn "GPU step requires GPU array management. Falling back to threaded." maxlog=1
    _step_threaded!(state, topo, params, rainfall, dt, work, infiltration, infil_state)
end

# =============================================================================
# Parallel Simulation Runner
# =============================================================================

"""
    run_simulation_parallel!(state, scenario; backend=get_backend(), kwargs...)

Run simulation using parallel backend.

# Arguments
- `state`: Initial state (modified in-place)
- `scenario`: Complete scenario specification
- `backend`: Computation backend (:serial, :threaded, :gpu)
- `progress_callback`: Optional callback(fraction) for progress updates
- `log_interval`: Time interval for progress logging (seconds)
- `verbosity`: Logging verbosity (0=silent, 1=basic, 2=detailed)

# Returns
- `SimulationResults`: Complete results including accumulator and mass balance
"""
function run_simulation_parallel!(state::SimulationState{T}, scenario::Scenario{T};
                                  backend::ComputeBackend=get_backend(),
                                  progress_callback=nothing,
                                  log_interval::Union{T, Nothing}=nothing,
                                  verbosity::Int=1) where T

    params = scenario.parameters
    topo = scenario.topography
    rainfall = scenario.rainfall
    grid = scenario.grid
    infiltration = scenario.infiltration

    # Log backend info
    if verbosity >= 1
        backend_name = if backend isa ThreadedBackend
            "Threaded ($(backend.nthreads) threads)"
        elseif backend isa GPUBackend
            "GPU"
        else
            "Serial"
        end
        @info "Starting parallel simulation" backend=backend_name grid_size="$(grid.nx)×$(grid.ny)"
    end

    # Create parallel workspace
    work = ParallelWorkspace(grid; backend=backend)

    # Create results accumulator
    results = ResultsAccumulator(grid, scenario.output_points)

    # Create infiltration state if needed
    infil_state = infiltration !== nothing ? InfiltrationState(grid) : nothing

    # Initialize mass balance
    mass_balance = MassBalance(state, grid)

    # Simulation loop
    step_count = 0
    last_output_time = state.t
    last_log_time = state.t
    start_wall_time = time()

    while state.t < params.t_end
        # Compute stable timestep (use threaded for larger grids)
        dt = if backend isa ThreadedBackend && grid.nx * grid.ny > 10000
            compute_dt_threaded(state, grid, params)
        else
            compute_dt(state, grid, params)
        end

        # Don't overshoot end time
        if state.t + dt > params.t_end
            dt = params.t_end - state.t
        end

        # Track rainfall
        rainfall_rate_m_s = rainfall_rate_ms(rainfall, state.t)
        rainfall_volume = rainfall_rate_m_s * dt * total_area(grid)
        add_rainfall!(mass_balance, rainfall_volume)

        # Perform timestep
        infiltrated = step_parallel!(state, topo, params, rainfall, dt, work;
                                    infiltration=infiltration, infil_state=infil_state)
        step_count += 1

        # Track infiltration
        if infiltrated > zero(T)
            add_infiltration!(mass_balance, infiltrated)
        end

        # Update mass balance
        update_volume!(mass_balance, state, grid)

        # Update results (use threaded for larger grids)
        if backend isa ThreadedBackend && grid.nx * grid.ny > 10000
            update_results_threaded!(results, state, dt; g=params.g)
        else
            update_results!(results, state, dt; g=params.g)
        end

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
        cells_per_sec = (grid.nx * grid.ny * step_count) / wall_time
        @info "Parallel simulation complete" steps=step_count wall_time=round(wall_time, digits=2) cells_per_sec=@sprintf("%.2e", cells_per_sec)
    end

    SimulationResults{T}(results, mass_balance, infil_state, step_count, wall_time)
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    benchmark_backends(grid_size::Int; duration=10.0, verbose=true)

Benchmark different backends on a synthetic problem.

Returns Dict with timing results for each backend.
"""
function benchmark_backends(grid_size::Int=100; duration::Float64=10.0, verbose::Bool=true)
    T = Float64

    # Create synthetic scenario
    grid = Grid(grid_size, grid_size, T(1.0), T(1.0))
    elevation = zeros(T, grid_size, grid_size)
    roughness = fill(T(0.03), grid_size, grid_size)
    topo = Topography(elevation, roughness)

    params = SimulationParameters(
        t_end=duration,
        dt_max=1.0,
        cfl=0.5,
        T=T
    )

    # Simple rainfall event
    rainfall = RainfallEvent{T}(
        times=T[0, duration/2, duration],
        rates=T[50, 50, 0]  # mm/hr
    )

    scenario = Scenario(grid, topo, params, rainfall, "benchmark")

    results = Dict{Symbol, NamedTuple}()

    # Test serial backend
    if verbose
        println("Benchmarking Serial backend...")
    end
    state = SimulationState(grid)
    t_serial = @elapsed begin
        run_simulation_parallel!(state, scenario; backend=SerialBackend(), verbosity=0)
    end
    results[:serial] = (time=t_serial, cells_per_sec=(grid_size^2 * state.t / params.dt_max) / t_serial)

    # Test threaded backend
    if nthreads() > 1
        if verbose
            println("Benchmarking Threaded backend ($(nthreads()) threads)...")
        end
        state = SimulationState(grid)
        t_threaded = @elapsed begin
            run_simulation_parallel!(state, scenario; backend=ThreadedBackend(), verbosity=0)
        end
        results[:threaded] = (time=t_threaded, cells_per_sec=(grid_size^2 * state.t / params.dt_max) / t_threaded)

        if verbose
            speedup = t_serial / t_threaded
            println("  Speedup: $(round(speedup, digits=2))x")
        end
    end

    # Test GPU backend if available
    if GPU_AVAILABLE[]
        if verbose
            println("Benchmarking GPU backend...")
        end
        state = SimulationState(grid)
        t_gpu = @elapsed begin
            run_simulation_parallel!(state, scenario; backend=GPUBackend(), verbosity=0)
        end
        results[:gpu] = (time=t_gpu, cells_per_sec=(grid_size^2 * state.t / params.dt_max) / t_gpu)

        if verbose
            speedup = t_serial / t_gpu
            println("  Speedup: $(round(speedup, digits=2))x")
        end
    end

    if verbose
        println("\nResults:")
        for (backend, r) in results
            println("  $backend: $(round(r.time, digits=3))s, $(round(r.cells_per_sec/1e6, digits=2))M cells/sec")
        end
    end

    results
end

"""
    auto_select_backend(grid::Grid)

Automatically select the best backend based on grid size and available resources.
"""
function auto_select_backend(grid::Grid)
    ncells = grid.nx * grid.ny

    # GPU for very large grids (if available)
    if GPU_AVAILABLE[] && ncells > 500_000
        return GPUBackend()
    end

    # Threading for medium to large grids (if multiple threads available)
    if nthreads() > 1 && ncells > 10_000
        return ThreadedBackend()
    end

    # Serial for small grids
    return SerialBackend()
end
