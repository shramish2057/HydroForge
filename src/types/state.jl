# HydroForge State Type
# Represents the simulation state at a point in time

"""
    SimulationState{T<:AbstractFloat}

Mutable struct representing the hydrodynamic state of the simulation.

# Fields
- `h::Matrix{T}`: Water depth at each cell (m)
- `qx::Matrix{T}`: Unit discharge in x direction (m²/s)
- `qy::Matrix{T}`: Unit discharge in y direction (m²/s)
- `t::T`: Current simulation time (s)
"""
mutable struct SimulationState{T<:AbstractFloat}
    h::Matrix{T}   # Water depth (m)
    qx::Matrix{T}  # Discharge in x (m²/s)
    qy::Matrix{T}  # Discharge in y (m²/s)
    t::T           # Current time (s)
end

"""
    SimulationState(nx, ny, T=Float64)

Create a zero-initialized simulation state for an nx × ny grid.
"""
function SimulationState(nx::Int, ny::Int, ::Type{T}=Float64) where T<:AbstractFloat
    SimulationState{T}(
        zeros(T, nx, ny),
        zeros(T, nx, ny),
        zeros(T, nx, ny),
        zero(T)
    )
end

"""
    SimulationState(grid::Grid, T=Float64)

Create a zero-initialized simulation state matching the grid dimensions.
"""
function SimulationState(grid::Grid, ::Type{T}=Float64) where T<:AbstractFloat
    SimulationState(grid.nx, grid.ny, T)
end

"""
    total_volume(state::SimulationState, grid::Grid)

Calculate the total water volume in the domain (m³).
"""
function total_volume(state::SimulationState, grid::Grid)
    sum(state.h) * cell_area(grid)
end

"""
    max_depth(state::SimulationState)

Return the maximum water depth in the domain (m).
"""
max_depth(state::SimulationState) = maximum(state.h)

"""
    min_depth(state::SimulationState)

Return the minimum water depth in the domain (m).
"""
min_depth(state::SimulationState) = minimum(state.h)

"""
    mean_depth(state::SimulationState)

Return the mean water depth over all cells (m).
"""
mean_depth(state::SimulationState) = sum(state.h) / length(state.h)

"""
    wet_cells(state::SimulationState, h_min)

Return the number of wet cells (depth > h_min).
"""
function wet_cells(state::SimulationState, h_min::Real)
    count(h -> h > h_min, state.h)
end

"""
    wet_fraction(state::SimulationState, h_min)

Return the fraction of cells that are wet.
"""
function wet_fraction(state::SimulationState, h_min::Real)
    wet_cells(state, h_min) / length(state.h)
end

"""
    max_velocity(state::SimulationState, h_min)

Return the maximum velocity magnitude in the domain (m/s).
"""
function max_velocity(state::SimulationState{T}, h_min::Real) where T
    vmax = zero(T)
    for j in axes(state.h, 2), i in axes(state.h, 1)
        h = state.h[i, j]
        if h > h_min
            u = state.qx[i, j] / h
            v = state.qy[i, j] / h
            vmag = sqrt(u^2 + v^2)
            vmax = max(vmax, vmag)
        end
    end
    vmax
end

"""
    copy_state(state::SimulationState)

Create a deep copy of the simulation state.
"""
function copy_state(state::SimulationState{T}) where T
    SimulationState{T}(
        copy(state.h),
        copy(state.qx),
        copy(state.qy),
        state.t
    )
end

"""
    reset!(state::SimulationState)

Reset the state to zero (in-place).
"""
function reset!(state::SimulationState{T}) where T
    fill!(state.h, zero(T))
    fill!(state.qx, zero(T))
    fill!(state.qy, zero(T))
    state.t = zero(T)
    state
end

# Show method
function Base.show(io::IO, state::SimulationState{T}) where T
    nx, ny = size(state.h)
    print(io, "SimulationState{$T}($(nx)x$(ny), t=$(state.t)s, max_h=$(round(max_depth(state), digits=4))m)")
end
