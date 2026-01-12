# HydroForge Grid Type
# Defines the computational grid structure

"""
    Grid{T<:AbstractFloat}

Represents a regular computational grid for 2D surface flow simulation.

# Fields
- `nx::Int`: Number of cells in x direction
- `ny::Int`: Number of cells in y direction
- `dx::T`: Cell size in x direction (meters)
- `dy::T`: Cell size in y direction (meters)
- `x_origin::T`: X coordinate of grid origin (lower-left corner)
- `y_origin::T`: Y coordinate of grid origin (lower-left corner)
- `crs::String`: Coordinate reference system (e.g., "EPSG:4326")
"""
struct Grid{T<:AbstractFloat}
    nx::Int
    ny::Int
    dx::T
    dy::T
    x_origin::T
    y_origin::T
    crs::String

    function Grid{T}(nx::Int, ny::Int, dx::T, dy::T,
                     x_origin::T, y_origin::T, crs::String) where T<:AbstractFloat
        nx > 0 || throw(ArgumentError("nx must be positive, got $nx"))
        ny > 0 || throw(ArgumentError("ny must be positive, got $ny"))
        dx > 0 || throw(ArgumentError("dx must be positive, got $dx"))
        dy > 0 || throw(ArgumentError("dy must be positive, got $dy"))
        new{T}(nx, ny, dx, dy, x_origin, y_origin, crs)
    end
end

# Convenience constructor with type inference
function Grid(nx::Int, ny::Int, dx::T, dy::T,
              x_origin::T, y_origin::T, crs::String="EPSG:4326") where T<:AbstractFloat
    Grid{T}(nx, ny, dx, dy, x_origin, y_origin, crs)
end

# Simple constructor with uniform cell size
function Grid(nx::Int, ny::Int, dx::T;
              x_origin::T=zero(T), y_origin::T=zero(T),
              crs::String="EPSG:4326") where T<:AbstractFloat
    Grid{T}(nx, ny, dx, dx, x_origin, y_origin, crs)
end

"""
    cell_area(grid::Grid)

Return the area of a single cell in square meters.
"""
cell_area(grid::Grid) = grid.dx * grid.dy

"""
    total_area(grid::Grid)

Return the total area of the grid in square meters.
"""
total_area(grid::Grid) = grid.nx * grid.ny * cell_area(grid)

"""
    extent(grid::Grid)

Return the grid extent as (xmin, xmax, ymin, ymax).
"""
function extent(grid::Grid{T}) where T
    xmin = grid.x_origin
    xmax = grid.x_origin + grid.nx * grid.dx
    ymin = grid.y_origin
    ymax = grid.y_origin + grid.ny * grid.dy
    (xmin, xmax, ymin, ymax)
end

"""
    cell_centers_x(grid::Grid)

Return vector of x-coordinates of cell centers.
"""
function cell_centers_x(grid::Grid{T}) where T
    [grid.x_origin + (i - 0.5) * grid.dx for i in 1:grid.nx]
end

"""
    cell_centers_y(grid::Grid)

Return vector of y-coordinates of cell centers.
"""
function cell_centers_y(grid::Grid{T}) where T
    [grid.y_origin + (j - 0.5) * grid.dy for j in 1:grid.ny]
end

"""
    cell_index(grid::Grid, x, y)

Return the (i, j) cell index for a given coordinate, or nothing if outside grid.
"""
function cell_index(grid::Grid, x::Real, y::Real)
    xmin, xmax, ymin, ymax = extent(grid)
    if x < xmin || x >= xmax || y < ymin || y >= ymax
        return nothing
    end
    i = floor(Int, (x - grid.x_origin) / grid.dx) + 1
    j = floor(Int, (y - grid.y_origin) / grid.dy) + 1
    (i, j)
end

# Show method for pretty printing
function Base.show(io::IO, grid::Grid{T}) where T
    print(io, "Grid{$T}($(grid.nx)x$(grid.ny), dx=$(grid.dx)m, dy=$(grid.dy)m)")
end
