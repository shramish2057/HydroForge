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
    apply_boundaries!(state::SimulationState, boundary::BoundaryType)

Apply boundary conditions to state in-place.
"""
function apply_boundaries!(state::SimulationState{T}, boundary::BoundaryType=CLOSED) where T
    if boundary == CLOSED
        apply_closed_boundaries!(state)
    elseif boundary == OPEN
        apply_open_boundaries!(state)
    else
        error("Boundary type $boundary not yet implemented")
    end
    nothing
end

"""
    apply_closed_boundaries!(state::SimulationState)

Apply closed (no-flow) boundary conditions.
Discharges at domain boundaries are set to zero.
"""
function apply_closed_boundaries!(state::SimulationState{T}) where T
    nx, ny = size(state.h)

    # Left and right boundaries (qx = 0)
    @inbounds for j in 1:ny
        state.qx[1, j] = zero(T)
        state.qx[nx, j] = zero(T)
    end

    # Bottom and top boundaries (qy = 0)
    @inbounds for i in 1:nx
        state.qy[i, 1] = zero(T)
        state.qy[i, ny] = zero(T)
    end

    nothing
end

"""
    apply_open_boundaries!(state::SimulationState)

Apply open (transmissive) boundary conditions.
Zero-gradient extrapolation for outflow.
"""
function apply_open_boundaries!(state::SimulationState{T}) where T
    nx, ny = size(state.h)

    # Left boundary: extrapolate from interior
    @inbounds for j in 1:ny
        state.qx[1, j] = state.qx[2, j]
        if state.qx[1, j] > 0  # Inflow not allowed
            state.qx[1, j] = zero(T)
        end
    end

    # Right boundary
    @inbounds for j in 1:ny
        state.qx[nx, j] = state.qx[nx-1, j]
        if state.qx[nx, j] < 0  # Inflow not allowed
            state.qx[nx, j] = zero(T)
        end
    end

    # Bottom boundary
    @inbounds for i in 1:nx
        state.qy[i, 1] = state.qy[i, 2]
        if state.qy[i, 1] > 0
            state.qy[i, 1] = zero(T)
        end
    end

    # Top boundary
    @inbounds for i in 1:nx
        state.qy[i, ny] = state.qy[i, ny-1]
        if state.qy[i, ny] < 0
            state.qy[i, ny] = zero(T)
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
