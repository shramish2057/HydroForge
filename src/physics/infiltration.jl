# HydroForge Infiltration Module
# Green-Ampt infiltration model with spatial soil type support

# =============================================================================
# Soil Type Definitions
# =============================================================================

"""
    SOIL_PARAMETERS

Standard soil parameters from literature (Green-Ampt).
Keys are soil type symbols, values are (K, ψ, θs) tuples.
- K: Saturated hydraulic conductivity (m/s)
- ψ: Wetting front suction head (m)
- θs: Soil porosity (dimensionless)
"""
const SOIL_PARAMETERS = Dict{Symbol,NTuple{3,Float64}}(
    :sand       => (1.2e-4, 0.05, 0.44),
    :loamy_sand => (3.0e-5, 0.06, 0.44),
    :sandy_loam => (1.1e-5, 0.11, 0.45),
    :loam       => (3.4e-6, 0.09, 0.46),
    :silt_loam  => (1.9e-6, 0.17, 0.50),
    :clay_loam  => (1.0e-6, 0.21, 0.46),
    :clay       => (3.0e-7, 0.32, 0.48),
    :impervious => (0.0, 0.0, 0.0),  # No infiltration (roads, buildings)
)

"""
    SOIL_TYPE_IDS

Mapping from integer IDs to soil type symbols for raster-based soil maps.
"""
const SOIL_TYPE_IDS = Dict{Int,Symbol}(
    0 => :impervious,
    1 => :sand,
    2 => :loamy_sand,
    3 => :sandy_loam,
    4 => :loam,
    5 => :silt_loam,
    6 => :clay_loam,
    7 => :clay,
)

# =============================================================================
# Infiltration Parameters
# =============================================================================

"""
    InfiltrationParameters{T<:AbstractFloat}

Parameters for Green-Ampt infiltration model.

# Fields
- `hydraulic_conductivity::T`: Saturated hydraulic conductivity K (m/s)
- `suction_head::T`: Wetting front suction head ψ (m)
- `porosity::T`: Soil porosity θs (dimensionless, 0-1)
- `initial_moisture::T`: Initial soil moisture θi (dimensionless, 0-1)
- `max_infiltration_depth::T`: Maximum cumulative infiltration (m), limits storage
- `soil_type::Symbol`: Soil type identifier

# Soil Type Reference (typical values)
| Soil Type    | K (m/s)  | ψ (m)  | θs    |
|--------------|----------|--------|-------|
| Sand         | 1.2e-4   | 0.05   | 0.44  |
| Loamy Sand   | 3.0e-5   | 0.06   | 0.44  |
| Sandy Loam   | 1.1e-5   | 0.11   | 0.45  |
| Loam         | 3.4e-6   | 0.09   | 0.46  |
| Silt Loam    | 1.9e-6   | 0.17   | 0.50  |
| Clay Loam    | 1.0e-6   | 0.21   | 0.46  |
| Clay         | 3.0e-7   | 0.32   | 0.48  |
| Impervious   | 0.0      | 0.0    | 0.0   |
"""
struct InfiltrationParameters{T<:AbstractFloat}
    hydraulic_conductivity::T  # K (m/s)
    suction_head::T            # ψ (m)
    porosity::T                # θs (dimensionless)
    initial_moisture::T        # θi (dimensionless)
    max_infiltration_depth::T  # Maximum F (m)
    soil_type::Symbol          # Soil type identifier
end

"""
    InfiltrationParameters(; kwargs...)

Create infiltration parameters with defaults for clay loam soil.

# Keyword Arguments
- `hydraulic_conductivity`: K in m/s (default: 1e-6, clay loam)
- `suction_head`: ψ in m (default: 0.21, clay loam)
- `porosity`: θs dimensionless (default: 0.46, clay loam)
- `initial_moisture`: θi dimensionless (default: 0.2)
- `max_infiltration_depth`: Maximum F in m (default: 1.0)
- `T`: Float type (default: Float64)
"""
function InfiltrationParameters(;
    hydraulic_conductivity::Real=1e-6,
    suction_head::Real=0.21,
    porosity::Real=0.46,
    initial_moisture::Real=0.2,
    max_infiltration_depth::Real=1.0,
    soil_type::Symbol=:clay_loam,
    T::Type{<:AbstractFloat}=Float64
)
    # Validate parameters
    @assert hydraulic_conductivity >= 0 "Hydraulic conductivity must be non-negative"
    @assert suction_head >= 0 "Suction head must be non-negative"
    @assert 0 <= porosity <= 1 "Porosity must be in [0, 1]"
    @assert 0 <= initial_moisture <= porosity "Initial moisture must be in [0, porosity]"

    InfiltrationParameters{T}(
        T(hydraulic_conductivity),
        T(suction_head),
        T(porosity),
        T(initial_moisture),
        T(max_infiltration_depth),
        soil_type
    )
end

"""
    InfiltrationParameters(soil_type::Symbol; initial_moisture=0.2, T=Float64)

Create infiltration parameters for a standard soil type.

# Supported soil types
- `:sand`, `:loamy_sand`, `:sandy_loam`, `:loam`
- `:silt_loam`, `:clay_loam`, `:clay`, `:impervious`
"""
function InfiltrationParameters(soil_type::Symbol;
                                 initial_moisture::Real=0.2,
                                 max_infiltration_depth::Real=1.0,
                                 T::Type{<:AbstractFloat}=Float64)
    if !haskey(SOIL_PARAMETERS, soil_type)
        error("Unknown soil type: $soil_type. " *
              "Supported types: $(join(keys(SOIL_PARAMETERS), ", "))")
    end

    K, ψ, θs = SOIL_PARAMETERS[soil_type]

    # Handle impervious surfaces
    if soil_type == :impervious
        initial_moisture = 0.0
    end

    InfiltrationParameters(
        hydraulic_conductivity=K,
        suction_head=ψ,
        porosity=θs,
        initial_moisture=min(initial_moisture, θs),
        max_infiltration_depth=max_infiltration_depth,
        soil_type=soil_type,
        T=T
    )
end

"""
    available_storage(params::InfiltrationParameters)

Calculate available storage capacity (Δθ = θs - θi).
"""
function available_storage(params::InfiltrationParameters{T}) where T
    params.porosity - params.initial_moisture
end

# =============================================================================
# Spatial Infiltration Parameters
# =============================================================================

"""
    SpatialInfiltrationParameters{T<:AbstractFloat}

Spatially varying infiltration parameters based on soil type map.

# Fields
- `soil_map::Matrix{Symbol}`: Soil type at each grid cell
- `K::Matrix{T}`: Hydraulic conductivity at each cell (m/s)
- `ψ::Matrix{T}`: Suction head at each cell (m)
- `θs::Matrix{T}`: Porosity at each cell
- `θi::Matrix{T}`: Initial moisture at each cell
- `max_depth::T`: Maximum cumulative infiltration depth (m)
"""
struct SpatialInfiltrationParameters{T<:AbstractFloat}
    soil_map::Matrix{Symbol}
    K::Matrix{T}               # Hydraulic conductivity
    ψ::Matrix{T}               # Suction head
    θs::Matrix{T}              # Porosity
    θi::Matrix{T}              # Initial moisture
    max_depth::T               # Maximum infiltration depth
end

"""
    SpatialInfiltrationParameters(soil_type_ids::Matrix{Int}; kwargs...)

Create spatial infiltration parameters from a soil type ID raster.

# Arguments
- `soil_type_ids`: Matrix of integer soil type IDs (see SOIL_TYPE_IDS)
- `initial_moisture`: Default initial moisture (default: 0.2)
- `max_depth`: Maximum cumulative infiltration depth (default: 1.0m)
- `T`: Float type (default: Float64)

# Soil Type IDs
- 0: Impervious (roads, buildings)
- 1: Sand
- 2: Loamy sand
- 3: Sandy loam
- 4: Loam
- 5: Silt loam
- 6: Clay loam
- 7: Clay
"""
function SpatialInfiltrationParameters(soil_type_ids::Matrix{Int};
                                        initial_moisture::Real=0.2,
                                        max_depth::Real=1.0,
                                        T::Type{<:AbstractFloat}=Float64)
    nx, ny = size(soil_type_ids)

    soil_map = Matrix{Symbol}(undef, nx, ny)
    K = zeros(T, nx, ny)
    ψ = zeros(T, nx, ny)
    θs = zeros(T, nx, ny)
    θi = zeros(T, nx, ny)

    for j in 1:ny, i in 1:nx
        type_id = soil_type_ids[i, j]

        # Get soil type symbol
        if haskey(SOIL_TYPE_IDS, type_id)
            soil_type = SOIL_TYPE_IDS[type_id]
        else
            @warn "Unknown soil type ID $type_id at ($i, $j), using clay_loam" maxlog=1
            soil_type = :clay_loam
        end

        soil_map[i, j] = soil_type

        # Get parameters
        k, psi, porosity = SOIL_PARAMETERS[soil_type]
        K[i, j] = T(k)
        ψ[i, j] = T(psi)
        θs[i, j] = T(porosity)
        θi[i, j] = soil_type == :impervious ? zero(T) : T(min(initial_moisture, porosity))
    end

    SpatialInfiltrationParameters{T}(soil_map, K, ψ, θs, θi, T(max_depth))
end

"""
    SpatialInfiltrationParameters(soil_map::Matrix{Symbol}; kwargs...)

Create spatial infiltration parameters from a soil type symbol map.
"""
function SpatialInfiltrationParameters(soil_map::Matrix{Symbol};
                                        initial_moisture::Real=0.2,
                                        max_depth::Real=1.0,
                                        T::Type{<:AbstractFloat}=Float64)
    nx, ny = size(soil_map)

    K = zeros(T, nx, ny)
    ψ = zeros(T, nx, ny)
    θs = zeros(T, nx, ny)
    θi = zeros(T, nx, ny)

    for j in 1:ny, i in 1:nx
        soil_type = soil_map[i, j]

        if !haskey(SOIL_PARAMETERS, soil_type)
            @warn "Unknown soil type $soil_type at ($i, $j), using clay_loam" maxlog=1
            soil_type = :clay_loam
        end

        k, psi, porosity = SOIL_PARAMETERS[soil_type]
        K[i, j] = T(k)
        ψ[i, j] = T(psi)
        θs[i, j] = T(porosity)
        θi[i, j] = soil_type == :impervious ? zero(T) : T(min(initial_moisture, porosity))
    end

    SpatialInfiltrationParameters{T}(copy(soil_map), K, ψ, θs, θi, T(max_depth))
end

"""
    SpatialInfiltrationParameters(grid::Grid, uniform_params::InfiltrationParameters)

Create uniform spatial parameters from a single InfiltrationParameters.
"""
function SpatialInfiltrationParameters(grid::Grid{T},
                                        params::InfiltrationParameters{T}) where T
    nx, ny = grid.nx, grid.ny
    soil_map = fill(params.soil_type, nx, ny)
    K = fill(params.hydraulic_conductivity, nx, ny)
    ψ = fill(params.suction_head, nx, ny)
    θs = fill(params.porosity, nx, ny)
    θi = fill(params.initial_moisture, nx, ny)

    SpatialInfiltrationParameters{T}(soil_map, K, ψ, θs, θi, params.max_infiltration_depth)
end

# =============================================================================
# Infiltration State
# =============================================================================

"""
    InfiltrationState{T<:AbstractFloat}

Tracks cumulative infiltration for Green-Ampt model.

# Fields
- `cumulative::Matrix{T}`: Cumulative infiltration F at each cell (m)
"""
mutable struct InfiltrationState{T<:AbstractFloat}
    cumulative::Matrix{T}
end

"""
    InfiltrationState(nx::Int, ny::Int, T=Float64)

Create infiltration state for a grid.
"""
function InfiltrationState(nx::Int, ny::Int, ::Type{T}=Float64) where T<:AbstractFloat
    InfiltrationState{T}(zeros(T, nx, ny))
end

"""
    InfiltrationState(grid::Grid)

Create infiltration state matching grid dimensions.
"""
InfiltrationState(grid::Grid{T}) where T = InfiltrationState(grid.nx, grid.ny, T)

"""
    reset!(state::InfiltrationState)

Reset cumulative infiltration to zero.
"""
function reset!(state::InfiltrationState{T}) where T
    fill!(state.cumulative, zero(T))
    state
end

# =============================================================================
# Infiltration Rate Calculations
# =============================================================================

"""
    infiltration_rate(h::Real, F::Real, params::InfiltrationParameters)

Calculate Green-Ampt infiltration rate.

f = K × (1 + ψ × Δθ / F)

where:
- K = saturated hydraulic conductivity
- ψ = suction head at wetting front
- Δθ = available storage (porosity - initial moisture)
- F = cumulative infiltration

# Arguments
- `h`: Current water depth (m) - infiltration limited by available water
- `F`: Cumulative infiltration so far (m)
- `params`: Infiltration parameters

# Returns
- Infiltration rate (m/s)
"""
function infiltration_rate(h::Real, F::Real, params::InfiltrationParameters{T}) where T
    # No infiltration if no water or soil is saturated
    if h <= zero(T) || F >= params.max_infiltration_depth
        return zero(T)
    end

    # Impervious surface
    if params.hydraulic_conductivity <= zero(T)
        return zero(T)
    end

    K = params.hydraulic_conductivity
    ψ = params.suction_head
    Δθ = available_storage(params)

    # Green-Ampt equation
    # Use small value for F to avoid division by zero at start
    F_eff = max(T(F), T(1e-10))

    f = K * (one(T) + ψ * Δθ / F_eff)

    T(f)
end

"""
    infiltration_rate(h::Real, F::Real, K::Real, ψ::Real, Δθ::Real, max_depth::Real)

Calculate Green-Ampt infiltration rate from individual parameters.
"""
function infiltration_rate(h::T, F::T, K::T, ψ::T, Δθ::T, max_depth::T) where T<:AbstractFloat
    # No infiltration conditions
    if h <= zero(T) || F >= max_depth || K <= zero(T)
        return zero(T)
    end

    # Green-Ampt equation
    F_eff = max(F, T(1e-10))
    K * (one(T) + ψ * Δθ / F_eff)
end

"""
    infiltration_rate(h::Real, params::InfiltrationParameters)

Calculate infiltration rate with zero cumulative infiltration (initial rate).
Returns potential infiltration rate at the very start of an event.
"""
function infiltration_rate(h::Real, params::InfiltrationParameters{T}) where T
    if h <= zero(T)
        return zero(T)
    end
    # At F→0, rate approaches K(1 + ψΔθ/ε) which is very high
    # Use small initial F for practical calculation
    infiltration_rate(h, T(1e-6), params)
end

# =============================================================================
# Infiltration Application (Uniform Parameters)
# =============================================================================

"""
    apply_infiltration!(h::Matrix, params::InfiltrationParameters, dt::Real)

Apply infiltration losses (simplified - no cumulative tracking).
Uses constant rate based on hydraulic conductivity only.
"""
function apply_infiltration!(h::Matrix{T}, params::InfiltrationParameters, dt::Real) where T
    K = params.hydraulic_conductivity
    infiltration_depth = T(K * dt)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > infiltration_depth
            h[i, j] -= infiltration_depth
        else
            h[i, j] = zero(T)
        end
    end

    nothing
end

"""
    apply_infiltration!(h::Matrix, infil_state::InfiltrationState,
                        params::InfiltrationParameters, dt::Real)

Apply Green-Ampt infiltration with cumulative tracking.

# Arguments
- `h`: Water depth matrix (modified in-place)
- `infil_state`: Infiltration state with cumulative tracking
- `params`: Infiltration parameters
- `dt`: Timestep (s)

# Returns
- Total infiltrated depth for this timestep (m, sum over all cells)
"""
function apply_infiltration!(h::Matrix{T}, infil_state::InfiltrationState{T},
                              params::InfiltrationParameters{T}, dt::Real) where T
    F = infil_state.cumulative
    total_infiltrated = zero(T)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > zero(T) && F[i, j] < params.max_infiltration_depth
            # Calculate infiltration rate
            f = infiltration_rate(h[i, j], F[i, j], params)

            # Potential infiltration this timestep
            potential = T(f * dt)

            # Limit by available water and remaining storage
            actual = min(potential, h[i, j],
                        params.max_infiltration_depth - F[i, j])

            # Apply infiltration
            h[i, j] -= actual
            F[i, j] += actual
            total_infiltrated += actual
        end
    end

    total_infiltrated
end

# =============================================================================
# Infiltration Application (Spatial Parameters)
# =============================================================================

"""
    apply_infiltration!(h::Matrix, infil_state::InfiltrationState,
                        params::SpatialInfiltrationParameters, dt::Real)

Apply Green-Ampt infiltration with spatially varying soil parameters.

# Arguments
- `h`: Water depth matrix (modified in-place)
- `infil_state`: Infiltration state with cumulative tracking
- `params`: Spatial infiltration parameters
- `dt`: Timestep (s)

# Returns
- Total infiltrated depth for this timestep (m, sum over all cells)
"""
function apply_infiltration!(h::Matrix{T}, infil_state::InfiltrationState{T},
                              params::SpatialInfiltrationParameters{T}, dt::Real) where T
    F = infil_state.cumulative
    K = params.K
    ψ = params.ψ
    θs = params.θs
    θi = params.θi
    max_depth = params.max_depth

    total_infiltrated = zero(T)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > zero(T) && F[i, j] < max_depth && K[i, j] > zero(T)
            # Available storage at this cell
            Δθ = θs[i, j] - θi[i, j]

            # Calculate infiltration rate
            f = infiltration_rate(h[i, j], F[i, j], K[i, j], ψ[i, j], Δθ, max_depth)

            # Potential infiltration this timestep
            potential = f * T(dt)

            # Limit by available water and remaining storage
            actual = min(potential, h[i, j], max_depth - F[i, j])

            # Apply infiltration
            h[i, j] -= actual
            F[i, j] += actual
            total_infiltrated += actual
        end
    end

    total_infiltrated
end

# =============================================================================
# Analysis Functions
# =============================================================================

"""
    total_infiltration(state::InfiltrationState, grid::Grid)

Calculate total infiltrated volume (m³).
"""
function total_infiltration(state::InfiltrationState{T}, grid::Grid{T}) where T
    sum(state.cumulative) * cell_area(grid)
end

"""
    infiltration_capacity_remaining(state::InfiltrationState, params::InfiltrationParameters)

Calculate remaining infiltration capacity as fraction (0-1).
"""
function infiltration_capacity_remaining(state::InfiltrationState{T},
                                          params::InfiltrationParameters{T}) where T
    max_total = params.max_infiltration_depth * length(state.cumulative)
    current_total = sum(state.cumulative)
    return (max_total - current_total) / max_total
end

"""
    infiltration_capacity_remaining(state::InfiltrationState, params::SpatialInfiltrationParameters)

Calculate remaining infiltration capacity as fraction for spatial parameters.
"""
function infiltration_capacity_remaining(state::InfiltrationState{T},
                                          params::SpatialInfiltrationParameters{T}) where T
    max_total = params.max_depth * count(x -> x > zero(T), params.K)
    current_total = sum(state.cumulative)
    return max_total > zero(T) ? (max_total - current_total) / max_total : one(T)
end

"""
    pervious_fraction(params::SpatialInfiltrationParameters)

Calculate the fraction of domain that is pervious (can infiltrate).
"""
function pervious_fraction(params::SpatialInfiltrationParameters{T}) where T
    total_cells = length(params.K)
    pervious_cells = count(x -> x > zero(T), params.K)
    return pervious_cells / total_cells
end
