# HydroForge Topography Type
# Represents terrain and surface properties

"""
    Topography{T<:AbstractFloat}

Represents the terrain and surface properties for the simulation domain.

# Fields
- `elevation::Matrix{T}`: Digital Elevation Model (m above datum)
- `slope_x::Matrix{T}`: Terrain slope in x direction (dimensionless)
- `slope_y::Matrix{T}`: Terrain slope in y direction (dimensionless)
- `roughness::Matrix{T}`: Manning's roughness coefficient (s/m^(1/3))
"""
struct Topography{T<:AbstractFloat}
    elevation::Matrix{T}
    slope_x::Matrix{T}
    slope_y::Matrix{T}
    roughness::Matrix{T}

    function Topography{T}(elevation, slope_x, slope_y, roughness) where T
        size(elevation) == size(slope_x) == size(slope_y) == size(roughness) ||
            throw(DimensionMismatch("All topography arrays must have the same size"))
        all(n -> n > 0, roughness) ||
            throw(ArgumentError("Manning's n must be positive everywhere"))
        new{T}(elevation, slope_x, slope_y, roughness)
    end
end

"""
    Topography(elevation::Matrix{T}, roughness::Matrix{T}, grid::Grid) where T

Create topography with automatic slope computation from elevation.
"""
function Topography(elevation::Matrix{T}, roughness::Matrix{T}, grid::Grid) where T<:AbstractFloat
    nx, ny = size(elevation)
    (nx, ny) == (grid.nx, grid.ny) ||
        throw(DimensionMismatch("Elevation dimensions must match grid"))

    slope_x = compute_slope_x(elevation, grid.dx)
    slope_y = compute_slope_y(elevation, grid.dy)

    Topography{T}(elevation, slope_x, slope_y, roughness)
end

"""
    Topography(elevation::Matrix{T}, n::Real, grid::Grid) where T

Create topography with uniform roughness.
"""
function Topography(elevation::Matrix{T}, n::Real, grid::Grid) where T<:AbstractFloat
    roughness = fill(T(n), size(elevation))
    Topography(elevation, roughness, grid)
end

"""
    compute_slope_x(elevation, dx)

Compute x-direction slope using central differences.
Boundary cells use one-sided differences.
"""
function compute_slope_x(elevation::Matrix{T}, dx::Real) where T
    nx, ny = size(elevation)
    slope_x = zeros(T, nx, ny)

    for j in 1:ny
        # Left boundary: forward difference
        slope_x[1, j] = (elevation[2, j] - elevation[1, j]) / dx

        # Interior: central difference
        for i in 2:nx-1
            slope_x[i, j] = (elevation[i+1, j] - elevation[i-1, j]) / (2 * dx)
        end

        # Right boundary: backward difference
        slope_x[nx, j] = (elevation[nx, j] - elevation[nx-1, j]) / dx
    end

    slope_x
end

"""
    compute_slope_y(elevation, dy)

Compute y-direction slope using central differences.
"""
function compute_slope_y(elevation::Matrix{T}, dy::Real) where T
    nx, ny = size(elevation)
    slope_y = zeros(T, nx, ny)

    for i in 1:nx
        # Bottom boundary: forward difference
        slope_y[i, 1] = (elevation[i, 2] - elevation[i, 1]) / dy

        # Interior: central difference
        for j in 2:ny-1
            slope_y[i, j] = (elevation[i, j+1] - elevation[i, j-1]) / (2 * dy)
        end

        # Top boundary: backward difference
        slope_y[i, ny] = (elevation[i, ny] - elevation[i, ny-1]) / dy
    end

    slope_y
end

"""
    min_elevation(topo::Topography)

Return minimum elevation in the domain.
"""
min_elevation(topo::Topography) = minimum(topo.elevation)

"""
    max_elevation(topo::Topography)

Return maximum elevation in the domain.
"""
max_elevation(topo::Topography) = maximum(topo.elevation)

"""
    elevation_range(topo::Topography)

Return (min, max) elevation in the domain.
"""
elevation_range(topo::Topography) = (min_elevation(topo), max_elevation(topo))

"""
    mean_roughness(topo::Topography)

Return the mean Manning's n value.
"""
mean_roughness(topo::Topography) = sum(topo.roughness) / length(topo.roughness)

"""
    validate(topo::Topography)

Validate topography data and return warnings.
"""
function validate(topo::Topography)
    warnings = String[]

    n_min, n_max = extrema(topo.roughness)
    if n_min < 0.01
        push!(warnings, "Manning's n < 0.01 is unusually low")
    end
    if n_max > 0.5
        push!(warnings, "Manning's n > 0.5 is unusually high")
    end

    slope_mag = sqrt.(topo.slope_x.^2 .+ topo.slope_y.^2)
    if maximum(slope_mag) > 1.0
        push!(warnings, "Terrain slope > 100% detected")
    end

    warnings
end

# Show method
function Base.show(io::IO, topo::Topography{T}) where T
    nx, ny = size(topo.elevation)
    zmin, zmax = elevation_range(topo)
    print(io, "Topography{$T}($(nx)x$(ny), z=$(round(zmin,digits=2))-$(round(zmax,digits=2))m)")
end
