# HydroForge Boundary Conditions
# Boundary condition implementations

"""
    BoundaryType

Enumeration of supported boundary condition types.
"""
@enum BoundaryType begin
    CLOSED      # No flow (reflective)
    OPEN        # Free outflow (transmissive)
    FIXED_DEPTH # Fixed water surface elevation
end

"""
    BoundaryCondition{T}

Complete specification of boundary conditions for a domain.

# Fields
- `type::BoundaryType`: Type of boundary condition
- `fixed_depth::T`: Fixed depth value for FIXED_DEPTH type (m)
- `sides::NTuple{4,Bool}`: Which sides to apply (left, right, bottom, top)
"""
struct BoundaryCondition{T<:AbstractFloat}
    type::BoundaryType
    fixed_depth::T
    sides::NTuple{4,Bool}  # (left, right, bottom, top)
end

"""
    BoundaryCondition(type::BoundaryType; fixed_depth=0.0, sides=(true,true,true,true))

Create a boundary condition specification.
"""
function BoundaryCondition(type::BoundaryType;
                           fixed_depth::Real=0.0,
                           sides::NTuple{4,Bool}=(true, true, true, true),
                           T::Type{<:AbstractFloat}=Float64)
    BoundaryCondition{T}(type, T(fixed_depth), sides)
end

"""
    apply_boundaries!(state::SimulationState, boundary::BoundaryType)

Apply boundary conditions to state in-place.
"""
function apply_boundaries!(state::SimulationState{T}, boundary::BoundaryType=CLOSED) where T
    if boundary == CLOSED
        apply_closed_boundaries!(state)
    elseif boundary == OPEN
        apply_open_boundaries!(state)
    elseif boundary == FIXED_DEPTH
        # Default fixed depth of 0 (dry boundary)
        apply_fixed_depth_boundaries!(state, zero(T))
    end
    nothing
end

"""
    apply_boundaries!(state::SimulationState, bc::BoundaryCondition)

Apply boundary conditions using full specification.
"""
function apply_boundaries!(state::SimulationState{T}, bc::BoundaryCondition{T}) where T
    if bc.type == CLOSED
        apply_closed_boundaries!(state, bc.sides)
    elseif bc.type == OPEN
        apply_open_boundaries!(state, bc.sides)
    elseif bc.type == FIXED_DEPTH
        apply_fixed_depth_boundaries!(state, bc.fixed_depth, bc.sides)
    end
    nothing
end

"""
    apply_closed_boundaries!(state::SimulationState)

Apply closed (no-flow) boundary conditions.
Discharges at domain boundaries are set to zero.
"""
function apply_closed_boundaries!(state::SimulationState{T}) where T
    apply_closed_boundaries!(state, (true, true, true, true))
end

"""
    apply_closed_boundaries!(state::SimulationState, sides::NTuple{4,Bool})

Apply closed boundaries to specified sides (left, right, bottom, top).
"""
function apply_closed_boundaries!(state::SimulationState{T},
                                   sides::NTuple{4,Bool}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides

    # Left boundary (qx = 0)
    if left
        @inbounds for j in 1:ny
            state.qx[1, j] = zero(T)
        end
    end

    # Right boundary (qx = 0)
    if right
        @inbounds for j in 1:ny
            state.qx[nx, j] = zero(T)
        end
    end

    # Bottom boundary (qy = 0)
    if bottom
        @inbounds for i in 1:nx
            state.qy[i, 1] = zero(T)
        end
    end

    # Top boundary (qy = 0)
    if top
        @inbounds for i in 1:nx
            state.qy[i, ny] = zero(T)
        end
    end

    nothing
end

"""
    apply_open_boundaries!(state::SimulationState)

Apply open (transmissive) boundary conditions.
Zero-gradient extrapolation for outflow.
"""
function apply_open_boundaries!(state::SimulationState{T}) where T
    apply_open_boundaries!(state, (true, true, true, true))
end

"""
    apply_open_boundaries!(state::SimulationState, sides::NTuple{4,Bool})

Apply open boundaries to specified sides (left, right, bottom, top).
"""
function apply_open_boundaries!(state::SimulationState{T},
                                 sides::NTuple{4,Bool}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides

    # Left boundary: extrapolate from interior
    if left
        @inbounds for j in 1:ny
            state.qx[1, j] = state.qx[2, j]
            if state.qx[1, j] > 0  # Inflow not allowed
                state.qx[1, j] = zero(T)
            end
        end
    end

    # Right boundary
    if right
        @inbounds for j in 1:ny
            state.qx[nx, j] = state.qx[nx-1, j]
            if state.qx[nx, j] < 0  # Inflow not allowed
                state.qx[nx, j] = zero(T)
            end
        end
    end

    # Bottom boundary
    if bottom
        @inbounds for i in 1:nx
            state.qy[i, 1] = state.qy[i, 2]
            if state.qy[i, 1] > 0
                state.qy[i, 1] = zero(T)
            end
        end
    end

    # Top boundary
    if top
        @inbounds for i in 1:nx
            state.qy[i, ny] = state.qy[i, ny-1]
            if state.qy[i, ny] < 0
                state.qy[i, ny] = zero(T)
            end
        end
    end

    nothing
end

"""
    apply_fixed_depth_boundaries!(state::SimulationState, fixed_depth::Real)

Apply fixed water depth at all boundaries.
Useful for modeling connections to water bodies (lakes, rivers, tidal boundaries).
"""
function apply_fixed_depth_boundaries!(state::SimulationState{T},
                                        fixed_depth::Real) where T
    apply_fixed_depth_boundaries!(state, T(fixed_depth), (true, true, true, true))
end

"""
    apply_fixed_depth_boundaries!(state::SimulationState, fixed_depth::Real, sides::NTuple{4,Bool})

Apply fixed water depth at specified boundaries.

# Arguments
- `state`: Simulation state to modify
- `fixed_depth`: Fixed water depth at boundary (m)
- `sides`: Which sides to apply (left, right, bottom, top)
"""
function apply_fixed_depth_boundaries!(state::SimulationState{T},
                                        fixed_depth::T,
                                        sides::NTuple{4,Bool}) where T
    nx, ny = size(state.h)
    left, right, bottom, top = sides

    # Left boundary
    if left
        @inbounds for j in 1:ny
            state.h[1, j] = fixed_depth
            # Allow flow based on gradient (don't zero discharge)
        end
    end

    # Right boundary
    if right
        @inbounds for j in 1:ny
            state.h[nx, j] = fixed_depth
        end
    end

    # Bottom boundary
    if bottom
        @inbounds for i in 1:nx
            state.h[i, 1] = fixed_depth
        end
    end

    # Top boundary
    if top
        @inbounds for i in 1:nx
            state.h[i, ny] = fixed_depth
        end
    end

    nothing
end

"""
    enforce_positive_depth!(state::SimulationState, h_min)

Enforce non-negative water depths (with small threshold).
"""
function enforce_positive_depth!(state::SimulationState{T}, h_min::T) where T
    negative_count = 0

    @inbounds for j in axes(state.h, 2), i in axes(state.h, 1)
        if state.h[i, j] < zero(T)
            negative_count += 1
            state.h[i, j] = zero(T)
        elseif state.h[i, j] < h_min
            # Keep very small depths but zero out discharge
            state.qx[i, j] = zero(T)
            state.qy[i, j] = zero(T)
        end
    end

    if negative_count > 0
        @warn "Corrected $negative_count negative depth cells"
    end

    nothing
end
