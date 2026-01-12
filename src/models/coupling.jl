# HydroForge 1D-2D Coupling Module
# Couples surface flow (2D) with drainage network (1D)
#
# Based on:
# - Leandro et al. (2016) "A methodology for linking 2D overland flow models with the sewer network model"
# - Chen et al. (2007) "Urban flood inundation and damage assessment"

"""
    InletFlowType

Flow regime through an inlet.
"""
@enum InletFlowType begin
    WEIR_FLOW       # Unsubmerged weir flow (low tailwater)
    ORIFICE_FLOW    # Submerged orifice flow (high tailwater)
    TRANSITION_FLOW # Transition between weir and orifice
    NO_FLOW         # Dry or reverse gradient
end

"""
    compute_inlet_flow(inlet::Inlet, h_surface, h_junction, z_surface, z_invert, g)

Compute flow through an inlet from surface to drainage network.

Uses weir equation for unsubmerged conditions:
    Q = C_w × L × h^(3/2)

Uses orifice equation for submerged conditions:
    Q = C_o × A × √(2g×Δh)

# Arguments
- `inlet`: Inlet definition
- `h_surface`: Water depth on surface above inlet (m)
- `h_junction`: Water depth in receiving junction (m)
- `z_surface`: Ground elevation at inlet (m)
- `z_invert`: Junction invert elevation (m)
- `g`: Gravity (m/s²)

# Returns
- (flow, flow_type): Flow rate (m³/s, positive into network) and flow regime
"""
function compute_inlet_flow(inlet::Inlet{T}, h_surface::T, h_junction::T,
                           z_surface::T, z_invert::T, g::T) where T
    # Effective inlet crest elevation
    z_crest = z_surface - inlet.depression

    # Head above crest
    h_above = h_surface + z_surface - z_crest

    if h_above <= zero(T)
        return (zero(T), NO_FLOW)
    end

    # Junction water surface elevation
    η_junction = z_invert + h_junction

    # Submergence ratio
    submergence = (η_junction - z_crest) / h_above

    if submergence < zero(T)
        # Unsubmerged weir flow
        # Q = C_w × L × h^(3/2)
        L_eff = inlet_perimeter(inlet)
        Q = inlet.weir_coeff * L_eff * h_above^(T(3)/T(2))
        return (Q, WEIR_FLOW)

    elseif submergence < T(0.8)
        # Transition zone - blend weir and orifice
        # Weir component
        L_eff = inlet_perimeter(inlet)
        Q_weir = inlet.weir_coeff * L_eff * h_above^(T(3)/T(2))

        # Orifice component
        Δh = h_surface + z_surface - η_junction
        A_eff = inlet_opening_area(inlet)
        Q_orifice = inlet.orifice_coeff * A_eff * sqrt(T(2) * g * max(Δh, zero(T)))

        # Blend based on submergence
        blend = submergence / T(0.8)
        Q = (one(T) - blend) * Q_weir + blend * Q_orifice

        return (Q, TRANSITION_FLOW)

    else
        # Fully submerged - orifice flow
        Δh = h_surface + z_surface - η_junction

        if Δh <= zero(T)
            # Reverse gradient - water coming up from network
            # (simplified: assume no reverse flow through inlet)
            return (zero(T), NO_FLOW)
        end

        A_eff = inlet_opening_area(inlet)
        Q = inlet.orifice_coeff * A_eff * sqrt(T(2) * g * Δh)

        return (Q, ORIFICE_FLOW)
    end
end


"""
    compute_outlet_return(outlet::Outlet, h_junction, z_junction,
                         h_surface, z_surface, g, t)

Compute flow from drainage network back to surface at an outlet.

This handles cases where the drainage system surcharges and water
returns to the surface (e.g., manhole overflow).

# Arguments
- `outlet`: Outlet definition
- `h_junction`: Water depth in junction (m)
- `z_junction`: Junction invert elevation (m)
- `h_surface`: Water depth on surface (m)
- `z_surface`: Ground elevation (m)
- `g`: Gravity (m/s²)
- `t`: Current time (for tidal boundaries)

# Returns
- Flow rate (m³/s, positive = into surface, negative = out of surface)
"""
function compute_outlet_return(outlet::Outlet{T}, h_junction::T, z_junction::T,
                               h_surface::T, z_surface::T, g::T, t::T) where T
    # Junction water surface
    η_junction = z_junction + h_junction

    # Determine external water level
    if outlet.outlet_type == :FREE
        η_external = z_surface + h_surface
    elseif outlet.outlet_type == :FIXED
        η_external = outlet.fixed_stage
    elseif outlet.outlet_type == :TIDAL && outlet.tide_curve !== nothing
        η_external = interpolate_tide(outlet.tide_curve, t)
    else
        η_external = outlet.invert
    end

    # Head difference
    Δη = η_junction - η_external

    if Δη <= zero(T)
        # No flow or inflow from external
        return zero(T)
    end

    # Assume outlet behaves like orifice
    # Typical manhole opening area ~ 0.5 m²
    A_outlet = T(0.5)

    # Apply flap gate loss if present
    loss_factor = one(T) - outlet.flap_loss

    Q = loss_factor * T(0.6) * A_outlet * sqrt(T(2) * g * Δη)

    return Q  # Positive = water returning to surface
end

"""
    interpolate_tide(tide_curve::Vector{Tuple{T,T}}, t)

Interpolate tidal water level at time t.
"""
function interpolate_tide(tide_curve::Vector{Tuple{T,T}}, t::T) where T
    if isempty(tide_curve)
        return zero(T)
    end

    # Before first point
    if t <= tide_curve[1][1]
        return tide_curve[1][2]
    end

    # After last point - use periodic extension or hold
    if t >= tide_curve[end][1]
        return tide_curve[end][2]
    end

    # Linear interpolation
    for i in 2:length(tide_curve)
        if t <= tide_curve[i][1]
            t1, η1 = tide_curve[i-1]
            t2, η2 = tide_curve[i]
            frac = (t - t1) / (t2 - t1)
            return η1 + frac * (η2 - η1)
        end
    end

    return tide_curve[end][2]
end


"""
    CoupledState{T}

Combined state for 1D-2D coupled simulation.

# Fields
- `surface::SimulationState{T}`: 2D surface flow state
- `drainage::DrainageState{T}`: 1D drainage network state
- `inlet_exchange::Vector{T}`: Flow through each inlet (m³/s)
- `outlet_return::Vector{T}`: Return flow at outlets (m³/s)
- `t::T`: Current time (s)
"""
mutable struct CoupledState{T<:AbstractFloat}
    surface::SimulationState{T}
    drainage::DrainageState{T}
    inlet_exchange::Vector{T}
    outlet_return::Vector{T}
    t::T
end

"""
    CoupledState(grid::Grid, network::DrainageNetwork)

Create initial coupled state.
"""
function CoupledState(grid::Grid{T}, network::DrainageNetwork{T}) where T
    CoupledState{T}(
        SimulationState(grid),
        DrainageState(network),
        zeros(T, n_inlets(network)),
        zeros(T, length(network.outlets)),
        zero(T)
    )
end


"""
    CoupledWorkspace{T}

Pre-allocated workspace for coupled solver.
"""
struct CoupledWorkspace{T<:AbstractFloat}
    surface_work::SimulationWorkspace{T}
    drainage_work::DrainageWorkspace{T}
    junction_inlet_flows::Vector{T}  # Aggregated inlet flows per junction
end

"""
    CoupledWorkspace(grid::Grid, network::DrainageNetwork)

Create workspace for coupled simulation.
"""
function CoupledWorkspace(grid::Grid{T}, network::DrainageNetwork{T}) where T
    CoupledWorkspace{T}(
        SimulationWorkspace(grid),
        DrainageWorkspace(network),
        zeros(T, n_junctions(network))
    )
end


"""
    CoupledScenario{T}

Complete scenario for coupled 1D-2D simulation.

# Fields
- `name::String`: Scenario name
- `grid::Grid{T}`: Computational grid
- `topography::Topography{T}`: Surface topography
- `parameters::SimulationParameters{T}`: Simulation parameters
- `rainfall::RainfallEvent{T}`: Rainfall event
- `network::DrainageNetwork{T}`: Drainage network
- `infiltration::Union{Nothing, InfiltrationParameters{T}}`: Infiltration model
- `output_points::Vector{Tuple{Int,Int}}`: Surface monitoring points
- `output_dir::String`: Output directory
"""
struct CoupledScenario{T<:AbstractFloat}
    name::String
    grid::Grid{T}
    topography::Topography{T}
    parameters::SimulationParameters{T}
    rainfall::RainfallEvent{T}
    network::DrainageNetwork{T}
    infiltration::Union{Nothing, InfiltrationParameters{T}}
    output_points::Vector{Tuple{Int,Int}}
    output_dir::String
end

"""
    CoupledScenario(scenario::Scenario, network::DrainageNetwork)

Create coupled scenario from existing surface scenario and drainage network.
"""
function CoupledScenario(scenario::Scenario{T}, network::DrainageNetwork{T}) where T
    CoupledScenario{T}(
        scenario.name,
        scenario.grid,
        scenario.topography,
        scenario.parameters,
        scenario.rainfall,
        network,
        scenario.infiltration,
        scenario.output_points,
        scenario.output_dir
    )
end


"""
    compute_exchange_flows!(state::CoupledState, scenario::CoupledScenario, g)

Compute all inlet and outlet exchange flows between surface and drainage.
Updates state.inlet_exchange and state.outlet_return.

# Returns
- Total inlet flow (positive into network, m³/s)
- Total outlet return (positive onto surface, m³/s)
"""
function compute_exchange_flows!(state::CoupledState{T},
                                 scenario::CoupledScenario{T},
                                 g::T) where T
    network = scenario.network
    grid = scenario.grid
    topo = scenario.topography

    total_inlet = zero(T)
    total_outlet = zero(T)

    # Compute inlet flows
    for (i, inlet) in enumerate(network.inlets)
        # Get surface state at inlet location
        h_surface = state.surface.h[inlet.grid_i, inlet.grid_j]
        z_surface = topo.elevation[inlet.grid_i, inlet.grid_j]

        # Get junction state
        j_idx = network.junction_index[inlet.junction_id]
        h_junction = state.drainage.depth[j_idx]
        z_invert = network.junctions[j_idx].invert

        # Compute flow
        Q, _ = compute_inlet_flow(inlet, h_surface, h_junction, z_surface, z_invert, g)
        state.inlet_exchange[i] = Q
        total_inlet += Q
    end

    # Compute outlet returns (surcharge)
    for (i, outlet) in enumerate(network.outlets)
        j_idx = network.junction_index[outlet.junction_id]
        junction = network.junctions[j_idx]
        h_junction = state.drainage.depth[j_idx]

        # Check if junction is surcharged
        if h_junction > junction.max_depth && outlet.grid_i > 0 && outlet.grid_j > 0
            h_surface = state.surface.h[outlet.grid_i, outlet.grid_j]
            z_surface = topo.elevation[outlet.grid_i, outlet.grid_j]

            Q = compute_outlet_return(outlet, h_junction, junction.invert,
                                      h_surface, z_surface, g, state.t)
            state.outlet_return[i] = Q
            total_outlet += Q
        else
            state.outlet_return[i] = zero(T)
        end
    end

    (total_inlet, total_outlet)
end


"""
    apply_exchange_to_surface!(state::CoupledState, scenario::CoupledScenario, dt)

Apply exchange flows to surface water depths.
Removes water at inlets, adds water at surcharged outlets.
"""
function apply_exchange_to_surface!(state::CoupledState{T},
                                    scenario::CoupledScenario{T},
                                    dt::T) where T
    network = scenario.network
    grid = scenario.grid

    cell_area = grid.dx * grid.dy

    # Remove water at inlets
    for (i, inlet) in enumerate(network.inlets)
        Q = state.inlet_exchange[i]
        if Q > zero(T)
            # Volume removed
            vol_removed = Q * dt

            # Current volume in cell
            h = state.surface.h[inlet.grid_i, inlet.grid_j]
            vol_available = h * cell_area

            # Don't remove more than available
            vol_actual = min(vol_removed, vol_available)

            # Update depth
            state.surface.h[inlet.grid_i, inlet.grid_j] -= vol_actual / cell_area

            # Adjust exchange if limited
            if vol_actual < vol_removed
                state.inlet_exchange[i] = vol_actual / dt
            end
        end
    end

    # Add water at surcharged outlets
    for (i, outlet) in enumerate(network.outlets)
        Q = state.outlet_return[i]
        if Q > zero(T) && outlet.grid_i > 0 && outlet.grid_j > 0
            vol_added = Q * dt
            state.surface.h[outlet.grid_i, outlet.grid_j] += vol_added / cell_area
        end
    end
end


"""
    aggregate_junction_inflows!(work::CoupledWorkspace, state::CoupledState,
                                network::DrainageNetwork)

Aggregate inlet flows to junction-level inflows for 1D solver.
"""
function aggregate_junction_inflows!(work::CoupledWorkspace{T},
                                     state::CoupledState{T},
                                     network::DrainageNetwork{T}) where T
    fill!(work.junction_inlet_flows, zero(T))

    for (i, inlet) in enumerate(network.inlets)
        j_idx = network.junction_index[inlet.junction_id]
        work.junction_inlet_flows[j_idx] += state.inlet_exchange[i]
    end
end


"""
    step_coupled!(state::CoupledState, scenario::CoupledScenario,
                  work::CoupledWorkspace, dt)

Advance coupled 1D-2D system by one timestep.

Uses sequential coupling:
1. Compute exchange flows based on current state
2. Apply inlet/outlet exchanges to surface
3. Step 2D surface solver
4. Aggregate inlet flows to junctions
5. Step 1D drainage solver

# Arguments
- `state`: Current coupled state (modified in-place)
- `scenario`: Coupled scenario
- `work`: Pre-allocated workspace
- `dt`: Time step (s)

# Returns
- Dictionary with step diagnostics
"""
function step_coupled!(state::CoupledState{T}, scenario::CoupledScenario{T},
                       work::CoupledWorkspace{T}, dt::T) where T
    g = scenario.parameters.g

    # 1. Compute exchange flows
    total_inlet, total_outlet = compute_exchange_flows!(state, scenario, g)

    # 2. Apply exchanges to surface
    apply_exchange_to_surface!(state, scenario, dt)

    # 3. Step 2D surface solver
    # Create temporary scenario without drainage for surface step
    if scenario.infiltration !== nothing
        infil_state = InfiltrationState(scenario.grid)
        step!(state.surface, scenario.topography, scenario.parameters, scenario.rainfall,
              dt, work.surface_work;
              infiltration=scenario.infiltration, infil_state=infil_state)
    else
        step!(state.surface, scenario.topography, scenario.parameters, scenario.rainfall,
              dt, work.surface_work)
    end

    # 4. Aggregate inlet flows to junctions
    aggregate_junction_inflows!(work, state, scenario.network)

    # 5. Step 1D drainage solver
    step_drainage!(state.drainage, scenario.network, work.drainage_work, dt;
                   g=g, inlet_flows=work.junction_inlet_flows)

    # Update time
    state.t = state.surface.t

    Dict{String,Any}(
        "inlet_flow" => total_inlet,
        "outlet_return" => total_outlet,
        "surface_max_depth" => max_depth(state.surface),
        "drainage_max_depth" => maximum(state.drainage.depth)
    )
end


"""
    compute_dt_coupled(state::CoupledState, scenario::CoupledScenario)

Compute stable timestep for coupled system.
Takes minimum of surface and drainage constraints.
"""
function compute_dt_coupled(state::CoupledState{T},
                            scenario::CoupledScenario{T}) where T
    params = scenario.parameters

    # Surface CFL
    dt_surface = compute_dt(state.surface, scenario.grid, params)

    # Drainage CFL
    dt_drainage = compute_dt_drainage(state.drainage, scenario.network;
                                      g=params.g, cfl=params.cfl, dt_max=params.dt_max)

    min(dt_surface, dt_drainage)
end


"""
    CoupledResults{T}

Results from coupled simulation.
"""
struct CoupledResults{T<:AbstractFloat}
    surface_results::ResultsAccumulator{T}
    drainage_summary::Dict{String,Any}
    total_inlet_volume::T
    total_outlet_volume::T
    step_count::Int
    wall_time::Float64
end


"""
    run_coupled!(state::CoupledState, scenario::CoupledScenario;
                 verbosity, log_interval)

Run coupled 1D-2D simulation.

# Arguments
- `state`: Initial coupled state (modified in-place)
- `scenario`: Coupled scenario
- `verbosity`: Logging level (0=silent, 1=info, 2=debug)
- `log_interval`: Time between progress logs (s)

# Returns
- CoupledResults with simulation summary
"""
function run_coupled!(state::CoupledState{T}, scenario::CoupledScenario{T};
                      verbosity::Int=1,
                      log_interval::Union{T,Nothing}=nothing) where T
    params = scenario.parameters
    t_end = params.t_end

    work = CoupledWorkspace(scenario.grid, scenario.network)

    # Initialize results
    surface_results = ResultsAccumulator(scenario.grid, scenario.output_points)

    step_count = 0
    total_inlet_vol = zero(T)
    total_outlet_vol = zero(T)
    next_log_time = zero(T)

    start_time = time()

    if verbosity >= 1
        @info "Starting coupled 1D-2D simulation" t_end=t_end n_cells=scenario.grid.nx*scenario.grid.ny n_pipes=n_pipes(scenario.network) n_inlets=n_inlets(scenario.network)
    end

    while state.t < t_end
        # Compute timestep
        dt = compute_dt_coupled(state, scenario)
        dt = min(dt, t_end - state.t)

        if dt <= zero(T)
            break
        end

        # Step
        diagnostics = step_coupled!(state, scenario, work, dt)

        # Track volumes
        total_inlet_vol += diagnostics["inlet_flow"] * dt
        total_outlet_vol += diagnostics["outlet_return"] * dt

        # Update results
        update_results!(surface_results, state.surface, dt; g=params.g)

        step_count += 1

        # Progress logging
        if verbosity >= 1 && log_interval !== nothing && state.t >= next_log_time
            @info "Coupled step $step_count" t=round(state.t, digits=1) surface_max_h=round(diagnostics["surface_max_depth"], digits=3) drainage_max_h=round(diagnostics["drainage_max_depth"], digits=3) inlet_Q=round(diagnostics["inlet_flow"], digits=4)
            next_log_time += log_interval
        end
    end

    wall_time = time() - start_time

    # Compute drainage summary
    drainage_summary = Dict{String,Any}(
        "final_volume" => total_volume(state.drainage, scenario.network),
        "max_depth" => maximum(state.drainage.depth),
        "total_inlet_volume" => total_inlet_vol,
        "total_outlet_volume" => total_outlet_vol
    )

    if verbosity >= 1
        @info "Coupled simulation complete" steps=step_count wall_time=round(wall_time, digits=2) inlet_vol=round(total_inlet_vol, digits=2) outlet_vol=round(total_outlet_vol, digits=2)
    end

    CoupledResults{T}(
        surface_results,
        drainage_summary,
        total_inlet_vol,
        total_outlet_vol,
        step_count,
        wall_time
    )
end
