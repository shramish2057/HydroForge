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
