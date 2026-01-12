# HydroForge Friction Module
# Manning friction implementation

"""
    friction_slope(q, h, n, h_min)

Calculate friction slope using Manning's equation.

Sf = n² |q| q / h^(10/3)

Semi-implicit formulation for numerical stability.

# Arguments
- `q`: Unit discharge (m²/s)
- `h`: Water depth (m)
- `n`: Manning's roughness coefficient
- `h_min`: Minimum depth threshold

# Returns
- Friction slope (dimensionless)
"""
function friction_slope(q::T, h::T, n::T, h_min::T) where T<:AbstractFloat
    if h <= h_min
        return zero(T)
    end
    # Sf = n² |q| q / h^(10/3)
    n^2 * abs(q) * q / h^(T(10)/T(3))
end

"""
    friction_factor(h, n, h_min, g, dt)

Calculate friction factor for semi-implicit treatment.

D = 1 + g dt n² |q| / h^(10/3)

# Arguments
- `h`: Water depth (m)
- `n`: Manning's roughness coefficient
- `h_min`: Minimum depth threshold
- `g`: Gravity (m/s²)
- `dt`: Timestep (s)

# Returns
- Friction denominator factor
"""
function friction_factor(q::T, h::T, n::T, h_min::T, g::T, dt::T) where T<:AbstractFloat
    if h <= h_min
        return one(T)
    end
    one(T) + g * dt * n^2 * abs(q) / h^(T(10)/T(3))
end

"""
    apply_friction!(qx, qy, h, n, h_min, g, dt)

Apply friction to discharge arrays in-place using semi-implicit method.

q_new = q_old / (1 + g dt n² |q| / h^(10/3))
"""
function apply_friction!(qx::Matrix{T}, qy::Matrix{T}, h::Matrix{T},
                         n::Matrix{T}, h_min::T, g::T, dt::T) where T
    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > h_min
            D_x = friction_factor(qx[i, j], h[i, j], n[i, j], h_min, g, dt)
            D_y = friction_factor(qy[i, j], h[i, j], n[i, j], h_min, g, dt)
            qx[i, j] /= D_x
            qy[i, j] /= D_y
        else
            qx[i, j] = zero(T)
            qy[i, j] = zero(T)
        end
    end
    nothing
end
