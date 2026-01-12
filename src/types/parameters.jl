# HydroForge Simulation Parameters
# Defines all runtime parameters for simulation

include("../config/defaults.jl")

"""
    SimulationParameters{T<:AbstractFloat}

Immutable struct containing all simulation parameters.

# Fields
- `dt_max::T`: Maximum timestep (s)
- `t_end::T`: End time of simulation (s)
- `cfl::T`: CFL number for timestep control (typically 0.5-0.9)
- `h_min::T`: Minimum depth to consider cell wet (m)
- `g::T`: Gravitational acceleration (m/s²)
- `output_interval::T`: Interval between outputs (s)
"""
struct SimulationParameters{T<:AbstractFloat}
    dt_max::T
    t_end::T
    cfl::T
    h_min::T
    g::T
    output_interval::T

    function SimulationParameters{T}(dt_max, t_end, cfl, h_min, g, output_interval) where T
        dt_max > 0 || throw(ArgumentError("dt_max must be positive"))
        t_end > 0 || throw(ArgumentError("t_end must be positive"))
        0 < cfl <= 1 || throw(ArgumentError("cfl must be in (0, 1]"))
        h_min >= 0 || throw(ArgumentError("h_min must be non-negative"))
        g > 0 || throw(ArgumentError("g must be positive"))
        output_interval > 0 || throw(ArgumentError("output_interval must be positive"))
        new{T}(dt_max, t_end, cfl, h_min, g, output_interval)
    end
end

# Constructor with type inference
function SimulationParameters(dt_max::T, t_end::T, cfl::T, h_min::T, g::T,
                              output_interval::T) where T<:AbstractFloat
    SimulationParameters{T}(dt_max, t_end, cfl, h_min, g, output_interval)
end

"""
    SimulationParameters(; kwargs...)

Create simulation parameters with keyword arguments and sensible defaults.

# Keyword Arguments
- `dt_max=1.0`: Maximum timestep (s)
- `t_end=3600.0`: End time (s), default 1 hour
- `cfl=0.7`: CFL number
- `h_min=0.001`: Minimum wet depth (m)
- `g=9.81`: Gravity (m/s²)
- `output_interval=60.0`: Output interval (s)
"""
function SimulationParameters(;
    dt_max::Real=DEFAULT_DT_MAX,
    t_end::Real=3600.0,
    cfl::Real=DEFAULT_CFL,
    h_min::Real=DEFAULT_H_MIN,
    g::Real=DEFAULT_G,
    output_interval::Real=DEFAULT_OUTPUT_INTERVAL,
    T::Type{<:AbstractFloat}=Float64
)
    SimulationParameters{T}(T(dt_max), T(t_end), T(cfl), T(h_min), T(g), T(output_interval))
end

"""
    validate(params::SimulationParameters)

Validate simulation parameters and return any warnings.
"""
function validate(params::SimulationParameters)
    warnings = String[]

    if params.cfl > 0.9
        push!(warnings, "CFL > 0.9 may cause instability")
    end

    if params.h_min > 0.01
        push!(warnings, "h_min > 0.01m is unusually large")
    end

    if params.dt_max > 10.0
        push!(warnings, "dt_max > 10s is unusually large for urban flooding")
    end

    if params.output_interval > params.t_end
        push!(warnings, "output_interval > t_end means no intermediate outputs")
    end

    warnings
end

# Show method
function Base.show(io::IO, params::SimulationParameters{T}) where T
    print(io, "SimulationParameters{$T}(t_end=$(params.t_end)s, cfl=$(params.cfl), h_min=$(params.h_min)m)")
end
