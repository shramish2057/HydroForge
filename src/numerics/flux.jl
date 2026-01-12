# HydroForge Flux Module
# Local inertial flux computation

"""
    compute_velocity(q, h, h_min)

Compute velocity from discharge, handling dry cells.

# Arguments
- `q`: Unit discharge (m²/s)
- `h`: Water depth (m)
- `h_min`: Minimum depth threshold

# Returns
- Velocity (m/s), zero for dry cells
"""
function compute_velocity(q::T, h::T, h_min::T) where T<:AbstractFloat
    if h > h_min
        q / h
    else
        zero(T)
    end
end

"""
    is_wet(h, h_min)

Check if a cell is wet (depth above minimum threshold).
"""
is_wet(h::T, h_min::T) where T = h > h_min

"""
    wet_dry_factor(h, h_min, h_transition)

Compute smooth transition factor for wetting/drying.
Returns 0 for dry cells, 1 for fully wet cells, smooth transition in between.

# Arguments
- `h`: Water depth
- `h_min`: Minimum depth (below this is dry)
- `h_transition`: Transition depth range (default 2*h_min)
"""
function wet_dry_factor(h::T, h_min::T, h_transition::T=T(2)*h_min) where T
    if h <= h_min
        zero(T)
    elseif h >= h_min + h_transition
        one(T)
    else
        # Smooth cubic transition
        x = (h - h_min) / h_transition
        x * x * (T(3) - T(2) * x)
    end
end

"""
    limit_flux_wetdry!(qx, qy, h, h_min)

Limit fluxes at wet/dry interfaces to prevent instabilities.
Reduces flux magnitude when flowing into or out of nearly-dry cells.
"""
function limit_flux_wetdry!(qx::Matrix{T}, qy::Matrix{T}, h::Matrix{T}, h_min::T) where T
    nx, ny = size(h)
    h_transition = T(2) * h_min

    # Limit x-direction fluxes
    @inbounds for j in 1:ny, i in 1:nx-1
        # Use minimum wet factor of neighboring cells
        factor_L = wet_dry_factor(h[i, j], h_min, h_transition)
        factor_R = wet_dry_factor(h[i+1, j], h_min, h_transition)
        factor = min(factor_L, factor_R)
        qx[i, j] *= factor
    end

    # Limit y-direction fluxes
    @inbounds for j in 1:ny-1, i in 1:nx
        factor_B = wet_dry_factor(h[i, j], h_min, h_transition)
        factor_T = wet_dry_factor(h[i, j+1], h_min, h_transition)
        factor = min(factor_B, factor_T)
        qy[i, j] *= factor
    end

    nothing
end

"""
    water_surface_elevation(h, z)

Compute water surface elevation η = h + z.
"""
water_surface_elevation(h::T, z::T) where T = h + z

"""
    water_surface_elevation!(η, h, z)

Compute water surface elevation in-place.
"""
function water_surface_elevation!(η::Matrix{T}, h::Matrix{T}, z::Matrix{T}) where T
    @inbounds for j in axes(h, 2), i in axes(h, 1)
        η[i, j] = h[i, j] + z[i, j]
    end
    nothing
end

"""
    surface_gradient(η::Matrix, grid::Grid)

Compute water surface gradients using central differences.

# Returns
- `dη_dx::Matrix`: Gradient in x-direction (∂η/∂x)
- `dη_dy::Matrix`: Gradient in y-direction (∂η/∂y)

Uses second-order central differences in the interior and
one-sided differences at boundaries.
"""
function surface_gradient(η::Matrix{T}, grid::Grid{T}) where T
    nx, ny = size(η)
    dη_dx = zeros(T, nx, ny)
    dη_dy = zeros(T, nx, ny)

    dx = grid.dx
    dy = grid.dy

    # Interior points: central differences
    @inbounds for j in 2:ny-1, i in 2:nx-1
        dη_dx[i, j] = (η[i+1, j] - η[i-1, j]) / (T(2) * dx)
        dη_dy[i, j] = (η[i, j+1] - η[i, j-1]) / (T(2) * dy)
    end

    # Boundary x-gradients (one-sided)
    @inbounds for j in 1:ny
        # Left boundary (forward difference)
        dη_dx[1, j] = (η[2, j] - η[1, j]) / dx
        # Right boundary (backward difference)
        dη_dx[nx, j] = (η[nx, j] - η[nx-1, j]) / dx
    end

    # Boundary y-gradients (one-sided)
    @inbounds for i in 1:nx
        # Bottom boundary (forward difference)
        dη_dy[i, 1] = (η[i, 2] - η[i, 1]) / dy
        # Top boundary (backward difference)
        dη_dy[i, ny] = (η[i, ny] - η[i, ny-1]) / dy
    end

    # Corner y-gradients already handled above
    # Interior row boundaries for y
    @inbounds for j in [1, ny], i in 2:nx-1
        dη_dx[i, j] = (η[i+1, j] - η[i-1, j]) / (T(2) * dx)
    end

    # Interior column boundaries for x
    @inbounds for i in [1, nx], j in 2:ny-1
        dη_dy[i, j] = (η[i, j+1] - η[i, j-1]) / (T(2) * dy)
    end

    (dη_dx, dη_dy)
end

"""
    surface_gradient!(dη_dx::Matrix, dη_dy::Matrix, η::Matrix, grid::Grid)

Compute water surface gradients in-place.
"""
function surface_gradient!(dη_dx::Matrix{T}, dη_dy::Matrix{T},
                           η::Matrix{T}, grid::Grid{T}) where T
    nx, ny = size(η)
    dx = grid.dx
    dy = grid.dy

    # Interior points: central differences
    @inbounds for j in 2:ny-1, i in 2:nx-1
        dη_dx[i, j] = (η[i+1, j] - η[i-1, j]) / (T(2) * dx)
        dη_dy[i, j] = (η[i, j+1] - η[i, j-1]) / (T(2) * dy)
    end

    # Boundary x-gradients (one-sided)
    @inbounds for j in 1:ny
        dη_dx[1, j] = (η[2, j] - η[1, j]) / dx
        dη_dx[nx, j] = (η[nx, j] - η[nx-1, j]) / dx
    end

    # Boundary y-gradients (one-sided)
    @inbounds for i in 1:nx
        dη_dy[i, 1] = (η[i, 2] - η[i, 1]) / dy
        dη_dy[i, ny] = (η[i, ny] - η[i, ny-1]) / dy
    end

    # Interior row boundaries for y
    @inbounds for j in [1, ny], i in 2:nx-1
        dη_dx[i, j] = (η[i+1, j] - η[i-1, j]) / (T(2) * dx)
    end

    # Interior column boundaries for x
    @inbounds for i in [1, nx], j in 2:ny-1
        dη_dy[i, j] = (η[i, j+1] - η[i, j-1]) / (T(2) * dy)
    end

    nothing
end

"""
    face_depth_x(h, z, i, j, direction)

Compute effective depth at x-face between cells (i,j) and (i+1,j).

Uses upwind reconstruction based on water surface elevation.
"""
function face_depth_x(h::Matrix{T}, z::Matrix{T}, i::Int, j::Int) where T
    η_L = h[i, j] + z[i, j]      # Left cell water surface
    η_R = h[i+1, j] + z[i+1, j]  # Right cell water surface
    z_face = max(z[i, j], z[i+1, j])  # Face bed elevation

    # Effective depth at face
    η_face = max(η_L, η_R)
    h_face = max(η_face - z_face, zero(T))

    h_face
end

"""
    face_depth_y(h, z, i, j)

Compute effective depth at y-face between cells (i,j) and (i,j+1).
"""
function face_depth_y(h::Matrix{T}, z::Matrix{T}, i::Int, j::Int) where T
    η_B = h[i, j] + z[i, j]      # Bottom cell
    η_T = h[i, j+1] + z[i, j+1]  # Top cell
    z_face = max(z[i, j], z[i, j+1])

    η_face = max(η_B, η_T)
    h_face = max(η_face - z_face, zero(T))

    h_face
end

"""
    compute_flux_x!(qx_new, qx, h, z, n, grid, params, dt)

Compute x-direction fluxes using local inertial approximation.

q^{n+1} = (q^n - g h_f dt ∂η/∂x) / (1 + g dt n² |q| / h_f^{10/3})

# Arguments
- `qx_new`: Output discharge array
- `qx`: Current discharge
- `h`: Water depth
- `z`: Bed elevation
- `n`: Manning's roughness
- `grid`: Computational grid
- `params`: Simulation parameters
- `dt`: Timestep
"""
function compute_flux_x!(qx_new::Matrix{T}, qx::Matrix{T}, h::Matrix{T},
                         z::Matrix{T}, n::Matrix{T}, grid::Grid{T},
                         params::SimulationParameters{T}, dt::T) where T
    g = params.g
    h_min = params.h_min
    dx = grid.dx
    nx, ny = grid.nx, grid.ny

    @inbounds for j in 1:ny
        for i in 1:nx-1
            # Face depth
            h_f = face_depth_x(h, z, i, j)

            if h_f > h_min
                # Water surface gradient
                η_L = h[i, j] + z[i, j]
                η_R = h[i+1, j] + z[i+1, j]
                dη_dx = (η_R - η_L) / dx

                # Face roughness (average)
                n_f = (n[i, j] + n[i+1, j]) / 2

                # Current flux at face (interpolated)
                q_f = (qx[i, j] + qx[i+1, j]) / 2

                # Friction factor (semi-implicit)
                D = one(T) + g * dt * n_f^2 * abs(q_f) / h_f^(T(10)/T(3))

                # New flux
                qx_new[i, j] = (q_f - g * h_f * dt * dη_dx) / D
            else
                qx_new[i, j] = zero(T)
            end
        end
        # Boundary (no flow)
        qx_new[nx, j] = zero(T)
    end

    nothing
end

"""
    compute_flux_y!(qy_new, qy, h, z, n, grid, params, dt)

Compute y-direction fluxes using local inertial approximation.
"""
function compute_flux_y!(qy_new::Matrix{T}, qy::Matrix{T}, h::Matrix{T},
                         z::Matrix{T}, n::Matrix{T}, grid::Grid{T},
                         params::SimulationParameters{T}, dt::T) where T
    g = params.g
    h_min = params.h_min
    dy = grid.dy
    nx, ny = grid.nx, grid.ny

    @inbounds for j in 1:ny-1
        for i in 1:nx
            # Face depth
            h_f = face_depth_y(h, z, i, j)

            if h_f > h_min
                # Water surface gradient
                η_B = h[i, j] + z[i, j]
                η_T = h[i, j+1] + z[i, j+1]
                dη_dy = (η_T - η_B) / dy

                # Face roughness
                n_f = (n[i, j] + n[i, j+1]) / 2

                # Current flux
                q_f = (qy[i, j] + qy[i, j+1]) / 2

                # Friction factor
                D = one(T) + g * dt * n_f^2 * abs(q_f) / h_f^(T(10)/T(3))

                # New flux
                qy_new[i, j] = (q_f - g * h_f * dt * dη_dy) / D
            else
                qy_new[i, j] = zero(T)
            end
        end
    end

    # Boundary (no flow)
    @inbounds for i in 1:nx
        qy_new[i, ny] = zero(T)
    end

    nothing
end
