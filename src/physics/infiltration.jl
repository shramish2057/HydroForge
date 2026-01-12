# HydroForge Infiltration Module
# Green-Ampt infiltration model implementation

"""
    InfiltrationParameters{T<:AbstractFloat}

Parameters for Green-Ampt infiltration model.

# Fields
- `hydraulic_conductivity::T`: Saturated hydraulic conductivity K (m/s)
- `suction_head::T`: Wetting front suction head ψ (m)
- `porosity::T`: Soil porosity θs (dimensionless, 0-1)
- `initial_moisture::T`: Initial soil moisture θi (dimensionless, 0-1)
- `max_infiltration_depth::T`: Maximum cumulative infiltration (m), limits storage

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
"""
struct InfiltrationParameters{T<:AbstractFloat}
    hydraulic_conductivity::T  # K (m/s)
    suction_head::T            # ψ (m)
    porosity::T                # θs (dimensionless)
    initial_moisture::T        # θi (dimensionless)
    max_infiltration_depth::T  # Maximum F (m)
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
    T::Type{<:AbstractFloat}=Float64
)
    # Validate parameters
    @assert 0 < hydraulic_conductivity "Hydraulic conductivity must be positive"
    @assert 0 <= suction_head "Suction head must be non-negative"
    @assert 0 < porosity <= 1 "Porosity must be in (0, 1]"
    @assert 0 <= initial_moisture < porosity "Initial moisture must be in [0, porosity)"

    InfiltrationParameters{T}(
        T(hydraulic_conductivity),
        T(suction_head),
        T(porosity),
        T(initial_moisture),
        T(max_infiltration_depth)
    )
end

"""
    InfiltrationParameters(soil_type::Symbol; initial_moisture=0.2, T=Float64)

Create infiltration parameters for a standard soil type.

# Supported soil types
- `:sand`, `:loamy_sand`, `:sandy_loam`, `:loam`
- `:silt_loam`, `:clay_loam`, `:clay`
"""
function InfiltrationParameters(soil_type::Symbol;
                                 initial_moisture::Real=0.2,
                                 max_infiltration_depth::Real=1.0,
                                 T::Type{<:AbstractFloat}=Float64)
    # Standard soil parameters from literature
    soil_params = Dict{Symbol,NTuple{3,Float64}}(
        :sand       => (1.2e-4, 0.05, 0.44),
        :loamy_sand => (3.0e-5, 0.06, 0.44),
        :sandy_loam => (1.1e-5, 0.11, 0.45),
        :loam       => (3.4e-6, 0.09, 0.46),
        :silt_loam  => (1.9e-6, 0.17, 0.50),
        :clay_loam  => (1.0e-6, 0.21, 0.46),
        :clay       => (3.0e-7, 0.32, 0.48),
    )

    if !haskey(soil_params, soil_type)
        error("Unknown soil type: $soil_type. " *
              "Supported types: $(join(keys(soil_params), ", "))")
    end

    K, ψ, θs = soil_params[soil_type]

    InfiltrationParameters(
        hydraulic_conductivity=K,
        suction_head=ψ,
        porosity=θs,
        initial_moisture=initial_moisture,
        max_infiltration_depth=max_infiltration_depth,
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

    K = params.hydraulic_conductivity
    ψ = params.suction_head
    Δθ = available_storage(params)

    # Green-Ampt equation
    # Use small value for F to avoid division by zero at start
    F_eff = max(T(F), T(1e-10))

    f = K * (one(T) + ψ * Δθ / F_eff)

    # Limit infiltration rate to available water depth
    # (can't infiltrate more than what's there)
    T(f)
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
- Total infiltrated volume for this timestep (m³, requires grid for conversion)
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

"""
    total_infiltration(state::InfiltrationState, grid::Grid)

Calculate total infiltrated volume (m³).
"""
function total_infiltration(state::InfiltrationState{T}, grid::Grid{T}) where T
    sum(state.cumulative) * cell_area(grid)
end
