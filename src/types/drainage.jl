# HydroForge 1D Drainage Network Types
# Types for storm drain pipes, junctions, inlets, and outlets
#
# Based on standard hydraulic engineering practice:
# - FHWA HEC-22 (Urban Drainage Design Manual)
# - ASCE Manual of Practice No. 77 (Design of Urban Stormwater Controls)

"""
    PipeCrossSection

Abstract type for pipe cross-sectional geometry.
"""
abstract type PipeCrossSection end

"""
    CircularPipe{T} <: PipeCrossSection

Circular pipe cross-section (most common in urban drainage).

# Fields
- `diameter::T`: Internal diameter (m)
"""
struct CircularPipe{T<:AbstractFloat} <: PipeCrossSection
    diameter::T
end

"""
    RectangularPipe{T} <: PipeCrossSection

Rectangular/box culvert cross-section.

# Fields
- `width::T`: Internal width (m)
- `height::T`: Internal height (m)
"""
struct RectangularPipe{T<:AbstractFloat} <: PipeCrossSection
    width::T
    height::T
end

"""
    flow_area(section::PipeCrossSection, depth)

Compute flow area for a given depth in the cross-section.
Returns full area if depth exceeds pipe height.
"""
function flow_area(section::CircularPipe{T}, depth::T) where T
    D = section.diameter
    if depth <= zero(T)
        return zero(T)
    elseif depth >= D
        # Full pipe
        return T(π) * D^2 / 4
    else
        # Partial flow - circular segment area
        # A = R² × (θ - sin(θ))/2 where θ = 2×acos(1 - 2h/D)
        R = D / 2
        h_ratio = depth / D
        θ = 2 * acos(1 - 2 * h_ratio)
        return R^2 * (θ - sin(θ)) / 2
    end
end

function flow_area(section::RectangularPipe{T}, depth::T) where T
    if depth <= zero(T)
        return zero(T)
    elseif depth >= section.height
        return section.width * section.height
    else
        return section.width * depth
    end
end

"""
    wetted_perimeter(section::PipeCrossSection, depth)

Compute wetted perimeter for a given depth.
"""
function wetted_perimeter(section::CircularPipe{T}, depth::T) where T
    D = section.diameter
    if depth <= zero(T)
        return zero(T)
    elseif depth >= D
        return T(π) * D
    else
        # Partial flow
        h_ratio = depth / D
        θ = 2 * acos(1 - 2 * h_ratio)
        return (D / 2) * θ
    end
end

function wetted_perimeter(section::RectangularPipe{T}, depth::T) where T
    if depth <= zero(T)
        return zero(T)
    elseif depth >= section.height
        return 2 * section.width + 2 * section.height
    else
        return section.width + 2 * depth
    end
end

"""
    hydraulic_radius(section::PipeCrossSection, depth)

Compute hydraulic radius R = A / P.
"""
function hydraulic_radius(section::PipeCrossSection, depth::T) where T
    A = flow_area(section, depth)
    P = wetted_perimeter(section, depth)
    if P > zero(T)
        return A / P
    else
        return zero(T)
    end
end

"""
    top_width(section::PipeCrossSection, depth)

Compute top width of water surface at given depth.
"""
function top_width(section::CircularPipe{T}, depth::T) where T
    D = section.diameter
    if depth <= zero(T) || depth >= D
        return zero(T)
    else
        # Width at depth h: 2 × √(R² - (R-h)²) = 2 × √(h × (D-h))
        return 2 * sqrt(depth * (D - depth))
    end
end

function top_width(section::RectangularPipe{T}, depth::T) where T
    if depth <= zero(T) || depth >= section.height
        return zero(T)
    else
        return section.width
    end
end

"""
    full_area(section::PipeCrossSection)

Return full cross-sectional area.
"""
full_area(section::CircularPipe{T}) where T = T(π) * section.diameter^2 / 4
full_area(section::RectangularPipe{T}) where T = section.width * section.height

"""
    full_depth(section::PipeCrossSection)

Return depth when pipe is full (diameter or height).
"""
full_depth(section::CircularPipe) = section.diameter
full_depth(section::RectangularPipe) = section.height


"""
    PipeSegment{T}

A drainage pipe segment connecting two junctions.

# Fields
- `id::Int`: Unique pipe identifier
- `upstream_node::Int`: ID of upstream junction
- `downstream_node::Int`: ID of downstream junction
- `section::PipeCrossSection`: Cross-sectional geometry
- `length::T`: Pipe length (m)
- `roughness::T`: Manning's n roughness coefficient
- `invert_up::T`: Invert elevation at upstream end (m)
- `invert_down::T`: Invert elevation at downstream end (m)
"""
struct PipeSegment{T<:AbstractFloat}
    id::Int
    upstream_node::Int
    downstream_node::Int
    section::PipeCrossSection
    length::T
    roughness::T
    invert_up::T
    invert_down::T
end

"""
    slope(pipe::Pipe)

Compute pipe slope (positive for downstream flow).
"""
slope(pipe::PipeSegment{T}) where T = (pipe.invert_up - pipe.invert_down) / pipe.length

"""
    is_adverse(pipe::Pipe)

Check if pipe has adverse (negative) slope.
"""
is_adverse(pipe::PipeSegment{T}) where T = slope(pipe) < zero(T)


"""
    JunctionType

Enumeration of junction types.
"""
@enum JunctionType begin
    MANHOLE         # Standard manhole
    STORAGE         # Storage node (detention pond, tank)
    OUTFALL         # System outfall (fixed/free boundary)
    DIVIDER         # Flow divider
end

"""
    Junction{T}

A junction node where pipes connect (typically a manhole).

# Fields
- `id::Int`: Unique junction identifier
- `x::T`: X-coordinate (m)
- `y::T`: Y-coordinate (m)
- `invert::T`: Invert elevation (m)
- `ground::T`: Ground surface elevation (m)
- `junction_type::JunctionType`: Type of junction
- `storage_curve::Union{Nothing, Vector{Tuple{T,T}}}`: (depth, area) for storage nodes
- `max_depth::T`: Maximum depth before surcharge (m)
- `init_depth::T`: Initial water depth (m)
- `ponded_area::T`: Surface ponding area if surcharging (m²)
"""
mutable struct Junction{T<:AbstractFloat}
    id::Int
    x::T
    y::T
    invert::T
    ground::T
    junction_type::JunctionType
    storage_curve::Union{Nothing, Vector{Tuple{T,T}}}
    max_depth::T
    init_depth::T
    ponded_area::T
end

"""
    Junction(id, x, y, invert, ground; kwargs...)

Create a junction with defaults for optional fields.
"""
function Junction(id::Int, x::T, y::T, invert::T, ground::T;
                  junction_type::JunctionType=MANHOLE,
                  storage_curve::Union{Nothing, Vector{Tuple{T,T}}}=nothing,
                  max_depth::T=ground - invert,
                  init_depth::T=zero(T),
                  ponded_area::T=zero(T)) where T<:AbstractFloat
    Junction{T}(id, x, y, invert, ground, junction_type, storage_curve,
                max_depth, init_depth, ponded_area)
end

"""
    rim_elevation(junction::Junction)

Return rim (ground surface) elevation.
"""
rim_elevation(junction::Junction) = junction.ground

"""
    is_surcharged(junction::Junction, depth)

Check if junction is surcharged (water above rim).
"""
is_surcharged(junction::Junction{T}, depth::T) where T = depth > junction.max_depth


"""
    InletType

Types of stormwater inlets.
"""
@enum InletType begin
    GRATE           # Grate inlet (horizontal opening)
    CURB            # Curb opening inlet
    COMBINATION     # Grate + curb opening
    SLOTTED         # Slotted drain
    DROP            # Drop inlet (vertical pipe)
end

"""
    Inlet{T}

A stormwater inlet connecting surface flow to drainage network.

# Fields
- `id::Int`: Unique inlet identifier
- `junction_id::Int`: Connected junction ID
- `grid_i::Int`: Grid cell i-index (1-based)
- `grid_j::Int`: Grid cell j-index (1-based)
- `inlet_type::InletType`: Type of inlet
- `length::T`: Inlet length (m) - for curb/grate
- `width::T`: Inlet width (m) - for grate
- `clogging_factor::T`: Capacity reduction (0-1, 1=fully open)
- `weir_coeff::T`: Weir coefficient (typically 1.5-1.7)
- `orifice_coeff::T`: Orifice coefficient (typically 0.6-0.67)
- `depression::T`: Local inlet depression below gutter (m)
"""
struct Inlet{T<:AbstractFloat}
    id::Int
    junction_id::Int
    grid_i::Int
    grid_j::Int
    inlet_type::InletType
    length::T
    width::T
    clogging_factor::T
    weir_coeff::T
    orifice_coeff::T
    depression::T
end

"""
    Inlet(id, junction_id, grid_i, grid_j; kwargs...)

Create an inlet with typical default coefficients (Float64).
"""
function Inlet(id::Int, junction_id::Int, grid_i::Int, grid_j::Int;
               inlet_type::InletType=GRATE,
               length::Float64=0.6,
               width::Float64=0.6,
               clogging_factor::Float64=1.0,
               weir_coeff::Float64=1.66,
               orifice_coeff::Float64=0.67,
               depression::Float64=0.0)
    Inlet{Float64}(id, junction_id, grid_i, grid_j, inlet_type, length, width,
             clogging_factor, weir_coeff, orifice_coeff, depression)
end

"""
    inlet_opening_area(inlet::Inlet)

Compute inlet opening area (for orifice flow).
"""
inlet_opening_area(inlet::Inlet{T}) where T = inlet.length * inlet.width * inlet.clogging_factor

"""
    inlet_perimeter(inlet::Inlet)

Compute inlet perimeter (for weir flow).
"""
inlet_perimeter(inlet::Inlet{T}) where T = (2 * inlet.length + inlet.width) * inlet.clogging_factor


"""
    Outlet{T}

A system outfall or outlet structure.

# Fields
- `id::Int`: Unique outlet identifier
- `junction_id::Int`: Connected junction ID
- `grid_i::Int`: Grid cell i-index (or -1 if outside domain)
- `grid_j::Int`: Grid cell j-index (or -1 if outside domain)
- `outlet_type::Symbol`: :FREE, :FIXED, :TIDAL, :FLAP
- `invert::T`: Outlet invert elevation (m)
- `fixed_stage::T`: Fixed water stage for :FIXED type (m)
- `tide_curve::Union{Nothing, Vector{Tuple{T,T}}}`: (time, stage) for :TIDAL
- `flap_loss::T`: Head loss coefficient for flap gate
"""
struct Outlet{T<:AbstractFloat}
    id::Int
    junction_id::Int
    grid_i::Int
    grid_j::Int
    outlet_type::Symbol
    invert::T
    fixed_stage::T
    tide_curve::Union{Nothing, Vector{Tuple{T,T}}}
    flap_loss::T
end

"""
    Outlet(id, junction_id; kwargs...)

Create an outlet with defaults for a free outfall (Float64).
"""
function Outlet(id::Int, junction_id::Int;
                grid_i::Int=-1,
                grid_j::Int=-1,
                outlet_type::Symbol=:FREE,
                invert::Float64=0.0,
                fixed_stage::Float64=0.0,
                tide_curve::Union{Nothing, Vector{Tuple{Float64,Float64}}}=nothing,
                flap_loss::Float64=0.0)
    Outlet{Float64}(id, junction_id, grid_i, grid_j, outlet_type, invert,
              fixed_stage, tide_curve, flap_loss)
end


"""
    DrainageNetwork{T}

Complete drainage network definition.

# Fields
- `pipes::Vector{PipeSegment{T}}`: All pipe segments
- `junctions::Vector{Junction{T}}`: All junction nodes
- `inlets::Vector{Inlet{T}}`: All surface inlets
- `outlets::Vector{Outlet{T}}`: All system outlets
- `pipe_index::Dict{Int,Int}`: pipe.id → vector index
- `junction_index::Dict{Int,Int}`: junction.id → vector index
- `inlet_index::Dict{Int,Int}`: inlet.id → vector index
- `outlet_index::Dict{Int,Int}`: outlet.id → vector index
- `upstream_pipes::Dict{Int,Vector{Int}}`: junction_id → upstream pipe ids
- `downstream_pipes::Dict{Int,Vector{Int}}`: junction_id → downstream pipe ids
- `junction_inlets::Dict{Int,Vector{Int}}`: junction_id → inlet ids
"""
struct DrainageNetwork{T<:AbstractFloat}
    pipes::Vector{PipeSegment{T}}
    junctions::Vector{Junction{T}}
    inlets::Vector{Inlet{T}}
    outlets::Vector{Outlet{T}}

    # Index lookups
    pipe_index::Dict{Int,Int}
    junction_index::Dict{Int,Int}
    inlet_index::Dict{Int,Int}
    outlet_index::Dict{Int,Int}

    # Connectivity
    upstream_pipes::Dict{Int,Vector{Int}}
    downstream_pipes::Dict{Int,Vector{Int}}
    junction_inlets::Dict{Int,Vector{Int}}
end

"""
    DrainageNetwork(pipes, junctions, inlets, outlets)

Construct drainage network and build connectivity indices.
"""
function DrainageNetwork(pipes::Vector{PipeSegment{T}},
                         junctions::Vector{Junction{T}},
                         inlets::Vector{Inlet{T}},
                         outlets::Vector{Outlet{T}}) where T
    # Build index lookups
    pipe_index = Dict(p.id => i for (i, p) in enumerate(pipes))
    junction_index = Dict(j.id => i for (i, j) in enumerate(junctions))
    inlet_index = Dict(inlet.id => i for (i, inlet) in enumerate(inlets))
    outlet_index = Dict(o.id => i for (i, o) in enumerate(outlets))

    # Build connectivity
    upstream_pipes = Dict{Int,Vector{Int}}(j.id => Int[] for j in junctions)
    downstream_pipes = Dict{Int,Vector{Int}}(j.id => Int[] for j in junctions)

    for pipe in pipes
        # Handle potentially invalid junction references gracefully
        if haskey(downstream_pipes, pipe.upstream_node)
            push!(downstream_pipes[pipe.upstream_node], pipe.id)
        end
        if haskey(upstream_pipes, pipe.downstream_node)
            push!(upstream_pipes[pipe.downstream_node], pipe.id)
        end
    end

    # Map inlets to junctions
    junction_inlets = Dict{Int,Vector{Int}}(j.id => Int[] for j in junctions)
    for inlet in inlets
        # Handle potentially invalid junction references gracefully
        if haskey(junction_inlets, inlet.junction_id)
            push!(junction_inlets[inlet.junction_id], inlet.id)
        end
    end

    DrainageNetwork{T}(
        pipes, junctions, inlets, outlets,
        pipe_index, junction_index, inlet_index, outlet_index,
        upstream_pipes, downstream_pipes, junction_inlets
    )
end

"""
    DrainageNetwork{T}()

Create an empty drainage network.
"""
function DrainageNetwork{T}() where T
    DrainageNetwork(
        PipeSegment{T}[], Junction{T}[], Inlet{T}[], Outlet{T}[],
        Dict{Int,Int}(), Dict{Int,Int}(), Dict{Int,Int}(), Dict{Int,Int}(),
        Dict{Int,Vector{Int}}(), Dict{Int,Vector{Int}}(), Dict{Int,Vector{Int}}()
    )
end

"""
    get_pipe(network::DrainageNetwork, id)

Get pipe by ID.
"""
get_pipe(network::DrainageNetwork, id::Int) = network.pipes[network.pipe_index[id]]

"""
    get_junction(network::DrainageNetwork, id)

Get junction by ID.
"""
get_junction(network::DrainageNetwork, id::Int) = network.junctions[network.junction_index[id]]

"""
    get_inlet(network::DrainageNetwork, id)

Get inlet by ID.
"""
get_inlet(network::DrainageNetwork, id::Int) = network.inlets[network.inlet_index[id]]

"""
    n_pipes(network::DrainageNetwork)

Return number of pipes.
"""
n_pipes(network::DrainageNetwork) = length(network.pipes)

"""
    n_junctions(network::DrainageNetwork)

Return number of junctions.
"""
n_junctions(network::DrainageNetwork) = length(network.junctions)

"""
    n_inlets(network::DrainageNetwork)

Return number of inlets.
"""
n_inlets(network::DrainageNetwork) = length(network.inlets)

"""
    validate(network::DrainageNetwork)

Validate drainage network connectivity and geometry.
Returns list of issues found (empty if valid).
"""
function validate(network::DrainageNetwork{T}) where T
    issues = String[]

    # Check all pipe endpoints reference valid junctions
    junction_ids = Set(j.id for j in network.junctions)
    for pipe in network.pipes
        if pipe.upstream_node ∉ junction_ids
            push!(issues, "Pipe $(pipe.id): upstream node $(pipe.upstream_node) not found")
        end
        if pipe.downstream_node ∉ junction_ids
            push!(issues, "Pipe $(pipe.id): downstream node $(pipe.downstream_node) not found")
        end
        if pipe.length <= zero(T)
            push!(issues, "Pipe $(pipe.id): length must be positive")
        end
        if pipe.roughness <= zero(T)
            push!(issues, "Pipe $(pipe.id): roughness must be positive")
        end
    end

    # Check all inlets reference valid junctions
    for inlet in network.inlets
        if inlet.junction_id ∉ junction_ids
            push!(issues, "Inlet $(inlet.id): junction $(inlet.junction_id) not found")
        end
        if inlet.grid_i < 1 || inlet.grid_j < 1
            push!(issues, "Inlet $(inlet.id): invalid grid indices")
        end
    end

    # Check all outlets reference valid junctions
    for outlet in network.outlets
        if outlet.junction_id ∉ junction_ids
            push!(issues, "Outlet $(outlet.id): junction $(outlet.junction_id) not found")
        end
    end

    # Check for orphan junctions (no connected pipes)
    for junction in network.junctions
        up = get(network.upstream_pipes, junction.id, Int[])
        down = get(network.downstream_pipes, junction.id, Int[])
        inlets = get(network.junction_inlets, junction.id, Int[])
        is_outlet = any(o.junction_id == junction.id for o in network.outlets)

        if isempty(up) && isempty(down) && isempty(inlets) && !is_outlet
            push!(issues, "Junction $(junction.id): no connections")
        end
    end

    issues
end


"""
    DrainageState{T}

Current state of the drainage network.

# Fields
- `depth::Vector{T}`: Water depth at each junction (m)
- `flow::Vector{T}`: Flow rate in each pipe (m³/s, positive = downstream)
- `inlet_flow::Vector{T}`: Flow rate through each inlet (m³/s, positive = into network)
- `outlet_flow::Vector{T}`: Flow rate through each outlet (m³/s, positive = out of network)
- `t::T`: Current time (s)
"""
mutable struct DrainageState{T<:AbstractFloat}
    depth::Vector{T}
    flow::Vector{T}
    inlet_flow::Vector{T}
    outlet_flow::Vector{T}
    t::T
end

"""
    DrainageState(network::DrainageNetwork{T})

Create initial drainage state from network definition.
"""
function DrainageState(network::DrainageNetwork{T}) where T
    n_j = n_junctions(network)
    n_p = n_pipes(network)
    n_i = n_inlets(network)
    n_o = length(network.outlets)

    # Initialize depths from junction init_depth
    depth = T[j.init_depth for j in network.junctions]

    DrainageState{T}(
        depth,
        zeros(T, n_p),
        zeros(T, n_i),
        zeros(T, n_o),
        zero(T)
    )
end

"""
    total_volume(state::DrainageState, network::DrainageNetwork)

Compute total water volume in drainage network.
"""
function total_volume(state::DrainageState{T}, network::DrainageNetwork{T}) where T
    vol = zero(T)

    # Volume in junctions (approximate as cylindrical storage)
    for (i, junction) in enumerate(network.junctions)
        if junction.storage_curve !== nothing
            # Interpolate storage from curve
            vol += storage_volume(junction, state.depth[i])
        else
            # Default small manhole storage
            manhole_area = T(1.0)  # 1 m² default
            vol += manhole_area * state.depth[i]
        end
    end

    # Volume in pipes (approximate using average depth)
    for (i, pipe) in enumerate(network.pipes)
        up_idx = network.junction_index[pipe.upstream_node]
        down_idx = network.junction_index[pipe.downstream_node]

        # Average depth in pipe (simplified)
        avg_depth = (state.depth[up_idx] + state.depth[down_idx]) / 2

        # Flow area at average depth
        A = flow_area(pipe.section, avg_depth)
        vol += A * pipe.length
    end

    vol
end

"""
    storage_volume(junction::Junction, depth)

Compute storage volume at a junction for given depth.
"""
function storage_volume(junction::Junction{T}, depth::T) where T
    if junction.storage_curve === nothing
        # Default manhole storage
        return T(1.0) * depth
    end

    curve = junction.storage_curve
    if depth <= zero(T)
        return zero(T)
    end

    # Integrate area over depth using trapezoidal rule
    vol = zero(T)
    prev_d, prev_a = zero(T), curve[1][2]

    for (d, a) in curve
        if d >= depth
            # Interpolate to exact depth
            frac = (depth - prev_d) / (d - prev_d)
            a_interp = prev_a + frac * (a - prev_a)
            vol += (depth - prev_d) * (prev_a + a_interp) / 2
            break
        else
            vol += (d - prev_d) * (prev_a + a) / 2
            prev_d, prev_a = d, a
        end
    end

    # If depth exceeds curve, extrapolate with last area
    if depth > curve[end][1]
        vol += (depth - curve[end][1]) * curve[end][2]
    end

    vol
end
