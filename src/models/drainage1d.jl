# HydroForge 1D Drainage Network Solver
# Solves pipe flow using kinematic wave / dynamic wave approximation
#
# Based on:
# - SWMM (Storm Water Management Model) algorithms
# - Rossman, L.A. (2017) "Storm Water Management Model Reference Manual Volume II"

"""
    manning_flow(A, R, S, n)

Compute flow using Manning's equation.
Q = (1/n) × A × R^(2/3) × S^(1/2)

# Arguments
- `A`: Flow area (m²)
- `R`: Hydraulic radius (m)
- `S`: Friction slope (m/m, positive for downstream flow)
- `n`: Manning's roughness coefficient

# Returns
- Flow rate (m³/s), positive for downstream direction
"""
function manning_flow(A::T, R::T, S::T, n::T) where T<:AbstractFloat
    if A <= zero(T) || R <= zero(T) || n <= zero(T)
        return zero(T)
    end

    if S >= zero(T)
        return (one(T) / n) * A * R^(T(2)/T(3)) * sqrt(S)
    else
        # Adverse slope - reverse flow
        return -(one(T) / n) * A * R^(T(2)/T(3)) * sqrt(-S)
    end
end

"""
    normal_depth(pipe::PipeSegment, Q)

Compute normal depth for a given discharge using bisection.
Returns the depth at which Manning flow equals Q.
"""
function normal_depth(pipe::PipeSegment{T}, Q::T; tol::T=T(1e-6), max_iter::Int=50) where T
    if abs(Q) < tol
        return zero(T)
    end

    S = slope(pipe)
    n = pipe.roughness
    d_full = full_depth(pipe.section)

    # Bisection search
    d_lo = zero(T)
    d_hi = d_full * T(2)  # Allow for surcharge

    for _ in 1:max_iter
        d_mid = (d_lo + d_hi) / 2
        A = flow_area(pipe.section, d_mid)
        R = hydraulic_radius(pipe.section, d_mid)
        Q_mid = manning_flow(A, R, abs(S), n)

        if abs(Q_mid - abs(Q)) < tol
            return d_mid
        elseif Q_mid < abs(Q)
            d_lo = d_mid
        else
            d_hi = d_mid
        end
    end

    return (d_lo + d_hi) / 2
end

"""
    full_flow_capacity(pipe::PipeSegment)

Compute full-pipe flow capacity using Manning's equation.
"""
function full_flow_capacity(pipe::PipeSegment{T}) where T
    d_full = full_depth(pipe.section)
    A = flow_area(pipe.section, d_full)
    R = hydraulic_radius(pipe.section, d_full)
    S = abs(slope(pipe))
    n = pipe.roughness

    manning_flow(A, R, S, n)
end

"""
    froude_number_1d(Q, A, T_w, g)

Compute Froude number for 1D channel flow.
Fr = Q / (A × √(g × A / T))

# Arguments
- `Q`: Flow rate (m³/s)
- `A`: Flow area (m²)
- `T_w`: Top width (m)
- `g`: Gravity (m/s²)
"""
function froude_number_1d(Q::T, A::T, T_w::T, g::T) where T<:AbstractFloat
    if A <= zero(T) || T_w <= zero(T)
        return zero(T)
    end

    v = Q / A
    depth_hydraulic = A / T_w
    c = sqrt(g * depth_hydraulic)

    return abs(v) / c
end


"""
    PipeFlowResult{T}

Result of pipe flow computation.
"""
struct PipeFlowResult{T<:AbstractFloat}
    flow::T              # Flow rate (m³/s)
    depth_up::T          # Upstream depth (m)
    depth_down::T        # Downstream depth (m)
    velocity::T          # Average velocity (m/s)
    froude::T            # Froude number
    flow_regime::Symbol  # :free_surface, :pressurized, :dry
end

"""
    compute_pipe_flow(pipe::PipeSegment, h_up, h_down, dt, Q_prev, g)

Compute flow through a pipe using dynamic wave approximation.

Uses a simplified momentum equation with:
- Gravity driving force
- Manning friction resistance
- Inertial term for stability

# Arguments
- `pipe`: Pipe segment
- `h_up`: Upstream junction water depth (m)
- `h_down`: Downstream junction water depth (m)
- `dt`: Time step (s)
- `Q_prev`: Previous flow rate (m³/s)
- `g`: Gravity (m/s²)

# Returns
- PipeFlowResult with computed flow and hydraulic properties
"""
function compute_pipe_flow(pipe::PipeSegment{T}, h_up::T, h_down::T,
                           dt::T, Q_prev::T, g::T) where T
    # Water surface elevations
    η_up = pipe.invert_up + h_up
    η_down = pipe.invert_down + h_down

    # Compute representative depth in pipe
    # Use average of upstream and downstream
    h_mid = max((h_up + h_down) / 2, zero(T))

    # Check for dry pipe
    if h_mid < T(1e-6)
        return PipeFlowResult{T}(zero(T), h_up, h_down, zero(T), zero(T), :dry)
    end

    # Determine flow regime
    d_full = full_depth(pipe.section)
    is_pressurized = h_mid >= d_full

    # Compute flow area and hydraulic radius
    if is_pressurized
        # Pressurized flow - use full pipe properties
        A = full_area(pipe.section)
        R = A / wetted_perimeter(pipe.section, d_full)
        regime = :pressurized
    else
        A = flow_area(pipe.section, h_mid)
        R = hydraulic_radius(pipe.section, h_mid)
        regime = :free_surface
    end

    # Compute water surface gradient (driving force)
    S_ws = (η_up - η_down) / pipe.length

    # Semi-implicit Manning friction
    n = pipe.roughness

    if R > zero(T) && A > zero(T)
        # Friction coefficient: K = n² × |Q| / (A² × R^(4/3))
        # Using previous flow for semi-implicit treatment
        K = n^2 * abs(Q_prev) / (A^2 * R^(T(4)/T(3)))

        # Momentum equation (simplified):
        # dQ/dt = g × A × S_ws - g × A × K × Q
        # Semi-implicit: Q_new = (Q_prev + g × A × S_ws × dt) / (1 + g × A × K × dt)

        numerator = Q_prev + g * A * S_ws * dt
        denominator = one(T) + g * A * K * dt

        Q_new = numerator / max(denominator, T(0.1))
    else
        Q_new = zero(T)
    end

    # Limit flow to physical capacity
    Q_max = full_flow_capacity(pipe) * T(2)  # Allow some surcharge capacity
    Q_new = clamp(Q_new, -Q_max, Q_max)

    # Compute velocity and Froude number
    velocity = A > zero(T) ? Q_new / A : zero(T)
    T_w = is_pressurized ? zero(T) : top_width(pipe.section, h_mid)
    froude = froude_number_1d(Q_new, A, T_w, g)

    PipeFlowResult{T}(Q_new, h_up, h_down, velocity, froude, regime)
end


"""
    DrainageWorkspace{T}

Pre-allocated workspace for drainage solver.
"""
mutable struct DrainageWorkspace{T<:AbstractFloat}
    # Junction inflows/outflows
    junction_inflow::Vector{T}    # Total inflow to each junction
    junction_outflow::Vector{T}   # Total outflow from each junction

    # Pipe flows (new and old)
    pipe_flow_new::Vector{T}

    # Temporary storage
    dh_dt::Vector{T}              # Rate of change of junction depth
end

"""
    DrainageWorkspace(network::DrainageNetwork{T})

Create workspace for drainage network.
"""
function DrainageWorkspace(network::DrainageNetwork{T}) where T
    n_j = n_junctions(network)
    n_p = n_pipes(network)

    DrainageWorkspace{T}(
        zeros(T, n_j),
        zeros(T, n_j),
        zeros(T, n_p),
        zeros(T, n_j)
    )
end


"""
    compute_dt_drainage(state::DrainageState, network::DrainageNetwork, params)

Compute stable timestep for drainage network.
Uses CFL condition based on wave celerity in pipes.
"""
function compute_dt_drainage(state::DrainageState{T}, network::DrainageNetwork{T};
                             g::T=T(9.81), cfl::T=T(0.9), dt_max::T=T(60.0)) where T
    dt = dt_max

    for (i, pipe) in enumerate(network.pipes)
        up_idx = network.junction_index[pipe.upstream_node]
        down_idx = network.junction_index[pipe.downstream_node]

        h_avg = (state.depth[up_idx] + state.depth[down_idx]) / 2

        if h_avg > T(0.01)
            # Wave celerity: c = √(g × h) for open channel
            c = sqrt(g * h_avg)

            # Pipe travel time
            dt_pipe = cfl * pipe.length / c
            dt = min(dt, dt_pipe)
        end
    end

    # Enforce minimum timestep
    max(dt, T(0.01))
end


"""
    step_drainage!(state::DrainageState, network::DrainageNetwork,
                   work::DrainageWorkspace, dt; g, inlet_flows)

Advance drainage network by one timestep.

# Arguments
- `state`: Current network state (modified in-place)
- `network`: Network definition
- `work`: Pre-allocated workspace
- `dt`: Time step (s)
- `g`: Gravity (m/s²)
- `inlet_flows`: External inflows to each junction (m³/s), length = n_junctions

# Returns
- Total outlet flow (m³/s)
"""
function step_drainage!(state::DrainageState{T}, network::DrainageNetwork{T},
                        work::DrainageWorkspace{T}, dt::T;
                        g::T=T(9.81),
                        inlet_flows::Vector{T}=zeros(T, n_junctions(network))) where T

    n_j = n_junctions(network)
    n_p = n_pipes(network)

    # Reset junction flows
    fill!(work.junction_inflow, zero(T))
    fill!(work.junction_outflow, zero(T))

    # Add external inlet flows
    for i in 1:n_j
        work.junction_inflow[i] = inlet_flows[i]
    end

    # Compute pipe flows
    for (i, pipe) in enumerate(network.pipes)
        up_idx = network.junction_index[pipe.upstream_node]
        down_idx = network.junction_index[pipe.downstream_node]

        result = compute_pipe_flow(pipe,
                                   state.depth[up_idx],
                                   state.depth[down_idx],
                                   dt,
                                   state.flow[i],
                                   g)

        work.pipe_flow_new[i] = result.flow

        # Track junction inflows/outflows
        if result.flow >= zero(T)
            # Flow going downstream
            work.junction_outflow[up_idx] += result.flow
            work.junction_inflow[down_idx] += result.flow
        else
            # Reverse flow
            work.junction_inflow[up_idx] += abs(result.flow)
            work.junction_outflow[down_idx] += abs(result.flow)
        end
    end

    # Compute outlet flows
    total_outlet = zero(T)
    for (i, outlet) in enumerate(network.outlets)
        j_idx = network.junction_index[outlet.junction_id]
        junction = network.junctions[j_idx]

        # Compute outlet flow based on type
        h = state.depth[j_idx]
        if h > zero(T)
            if outlet.outlet_type == :FREE
                # Free outfall - critical depth discharge
                # Q ≈ area × √(g × h)
                A_outlet = T(1.0)  # Approximate outlet area
                state.outlet_flow[i] = A_outlet * sqrt(g * h) * h
            elseif outlet.outlet_type == :FIXED
                # Fixed stage - head difference drives flow
                Δh = h + junction.invert - outlet.fixed_stage
                if Δh > zero(T)
                    state.outlet_flow[i] = T(1.0) * sqrt(T(2) * g * Δh)
                else
                    state.outlet_flow[i] = zero(T)
                end
            else
                state.outlet_flow[i] = zero(T)
            end
        else
            state.outlet_flow[i] = zero(T)
        end

        work.junction_outflow[j_idx] += state.outlet_flow[i]
        total_outlet += state.outlet_flow[i]
    end

    # Update junction depths using continuity
    for (i, junction) in enumerate(network.junctions)
        # Storage area (m²)
        if junction.storage_curve !== nothing
            # Use storage curve
            A_storage = interpolate_storage_area(junction, state.depth[i])
        else
            # Default manhole area
            A_storage = T(1.0)
        end

        # Add ponded area if surcharged
        if state.depth[i] > junction.max_depth && junction.ponded_area > zero(T)
            A_storage += junction.ponded_area
        end

        # dh/dt = (Q_in - Q_out) / A
        net_flow = work.junction_inflow[i] - work.junction_outflow[i]
        work.dh_dt[i] = net_flow / max(A_storage, T(0.1))
    end

    # Update depths
    for i in 1:n_j
        new_depth = state.depth[i] + work.dh_dt[i] * dt

        # Enforce non-negative depth
        state.depth[i] = max(new_depth, zero(T))

        # Handle surcharge
        junction = network.junctions[i]
        if state.depth[i] > junction.max_depth && junction.ponded_area <= zero(T)
            # No ponding allowed - cap at max depth
            state.depth[i] = junction.max_depth
        end
    end

    # Update pipe flows
    for i in 1:n_p
        state.flow[i] = work.pipe_flow_new[i]
    end

    # Update time
    state.t += dt

    total_outlet
end

"""
    interpolate_storage_area(junction::Junction, depth)

Interpolate storage area from storage curve at given depth.
"""
function interpolate_storage_area(junction::Junction{T}, depth::T) where T
    if junction.storage_curve === nothing
        return T(1.0)
    end

    curve = junction.storage_curve

    if depth <= curve[1][1]
        return curve[1][2]
    end

    for i in 2:length(curve)
        if depth <= curve[i][1]
            # Linear interpolation
            d1, a1 = curve[i-1]
            d2, a2 = curve[i]
            frac = (depth - d1) / (d2 - d1)
            return a1 + frac * (a2 - a1)
        end
    end

    # Extrapolate with last area
    return curve[end][2]
end


"""
    run_drainage!(state::DrainageState, network::DrainageNetwork, t_end;
                  g, dt_max, cfl, inlet_flows_func, verbosity)

Run drainage simulation for specified duration.

# Arguments
- `state`: Initial state (modified in-place)
- `network`: Network definition
- `t_end`: End time (s)
- `g`: Gravity (m/s²)
- `dt_max`: Maximum timestep (s)
- `cfl`: CFL number
- `inlet_flows_func`: Function (t) → Vector{T} of inlet flows at time t
- `verbosity`: Logging level (0=silent, 1=info, 2=debug)

# Returns
- Dictionary with simulation summary
"""
function run_drainage!(state::DrainageState{T}, network::DrainageNetwork{T}, t_end::T;
                       g::T=T(9.81),
                       dt_max::T=T(60.0),
                       cfl::T=T(0.9),
                       inlet_flows_func::Function=t -> zeros(T, n_junctions(network)),
                       verbosity::Int=1) where T

    work = DrainageWorkspace(network)

    step_count = 0
    total_outlet_vol = zero(T)
    total_inlet_vol = zero(T)
    max_depth = zero(T)

    start_time = time()

    while state.t < t_end
        # Compute timestep
        dt = compute_dt_drainage(state, network; g=g, cfl=cfl, dt_max=dt_max)
        dt = min(dt, t_end - state.t)

        if dt <= zero(T)
            break
        end

        # Get inlet flows for current time
        inlet_flows = inlet_flows_func(state.t)

        # Step
        outlet_flow = step_drainage!(state, network, work, dt;
                                     g=g, inlet_flows=inlet_flows)

        # Track volumes
        total_outlet_vol += outlet_flow * dt
        total_inlet_vol += sum(inlet_flows) * dt

        # Track max depth
        for d in state.depth
            max_depth = max(max_depth, d)
        end

        step_count += 1

        # Progress logging
        if verbosity >= 2 && step_count % 100 == 0
            @info "Drainage step $step_count: t=$(round(state.t, digits=1))s, max_depth=$(round(max_depth, digits=3))m"
        end
    end

    wall_time = time() - start_time

    if verbosity >= 1
        @info "Drainage simulation complete" steps=step_count wall_time=round(wall_time, digits=2) max_depth=round(max_depth, digits=3)
    end

    Dict{String,Any}(
        "step_count" => step_count,
        "wall_time" => wall_time,
        "max_depth" => max_depth,
        "total_outlet_volume" => total_outlet_vol,
        "total_inlet_volume" => total_inlet_vol,
        "final_volume" => total_volume(state, network)
    )
end
