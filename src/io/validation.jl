# HydroForge Input Validation
# Functions for validating input data

"""
    validate_dem(elevation::Matrix, grid::Grid)

Validate DEM data against grid specifications.

# Throws
- `DimensionMismatch`: If dimensions don't match grid
- `ArgumentError`: If data contains invalid values
"""
function validate_dem(elevation::Matrix, grid::Grid)
    (size(elevation, 1), size(elevation, 2)) == (grid.nx, grid.ny) ||
        throw(DimensionMismatch("DEM size $(size(elevation)) doesn't match grid ($(grid.nx), $(grid.ny))"))

    if any(isnan, elevation)
        throw(ArgumentError("DEM contains NaN values"))
    end

    if any(isinf, elevation)
        throw(ArgumentError("DEM contains Inf values"))
    end

    true
end

"""
    validate_roughness(roughness::Matrix, grid::Grid)

Validate roughness data.
"""
function validate_roughness(roughness::Matrix, grid::Grid)
    (size(roughness, 1), size(roughness, 2)) == (grid.nx, grid.ny) ||
        throw(DimensionMismatch("Roughness size doesn't match grid"))

    n_min, n_max = extrema(roughness)

    if n_min <= 0
        throw(ArgumentError("Manning's n must be positive, got minimum $n_min"))
    end

    if n_min < 0.01
        @warn "Manning's n value $n_min is unusually low"
    end

    if n_max > 0.5
        @warn "Manning's n value $n_max is unusually high"
    end

    true
end

"""
    validate_rainfall(rainfall::RainfallEvent)

Validate rainfall event data.
"""
function validate_rainfall(rainfall::RainfallEvent)
    if any(i -> i < 0, rainfall.intensities)
        throw(ArgumentError("Rainfall intensities must be non-negative"))
    end

    if !issorted(rainfall.times)
        throw(ArgumentError("Rainfall times must be sorted"))
    end

    if peak_intensity(rainfall) > 500
        @warn "Peak intensity $(peak_intensity(rainfall)) mm/hr is unusually high"
    end

    true
end

"""
    validate_scenario(scenario::Scenario)

Run all validations on a scenario.

# Returns
- `true` if all validations pass
- Throws if any validation fails

# Warnings
Logs warnings for unusual but valid values.
"""
function validate_scenario(scenario::Scenario)
    validate_dem(scenario.topography.elevation, scenario.grid)
    validate_roughness(scenario.topography.roughness, scenario.grid)
    validate_rainfall(scenario.rainfall)

    # Log all warnings
    warnings = validate(scenario)
    for w in warnings
        @warn w
    end

    true
end
