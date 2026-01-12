"""
    HydroForge

Real-Time Urban Flood Risk Simulator.

A high-performance, open-source platform for simulating urban stormwater
runoff and surface inundation using the shallow water equations.

## Quick Start

```julia
using HydroForge

# Run the bundled demo scenario
HydroForge.run_demo()

# Run a custom scenario
HydroForge.run("path/to/scenario.toml")
```

## Main Types

- `Grid`: Computational grid definition
- `Topography`: Terrain and surface properties
- `SimulationState`: Current simulation state (depth, discharge)
- `SimulationParameters`: Runtime parameters
- `RainfallEvent`: Rainfall time series
- `Scenario`: Complete simulation scenario

## Modules

- `HydroForge.Types`: Core data types
- `HydroForge.Numerics`: Numerical methods (flux, timestep)
- `HydroForge.Physics`: Physical processes (friction, rainfall)
- `HydroForge.IO`: Input/output functions
- `HydroForge.API`: HTTP API server (Phase 12+)

"""
module HydroForge

using Dates
using DelimitedFiles
using JSON3
using LinearAlgebra
using Random
using TOML

# Version
const HYDROFORGE_VERSION = "0.1.0-dev"

# =============================================================================
# Configuration
# =============================================================================
include("config/defaults.jl")

# =============================================================================
# Core Types (ordered for dependencies)
# =============================================================================
include("types/grid.jl")
include("types/topography.jl")
include("types/state.jl")
include("types/parameters.jl")

# Infiltration types need to be defined before Scenario
include("physics/infiltration.jl")

# Scenario uses InfiltrationParameters
include("types/scenario.jl")

# Export types
export Grid, cell_area, total_area, extent, cell_centers_x, cell_centers_y, cell_index
export Topography, min_elevation, max_elevation, elevation_range, mean_roughness
export SimulationState, total_volume, max_depth, min_depth, mean_depth
export wet_cells, wet_fraction, max_velocity, copy_state, reset!
export SimulationParameters, validate
export RainfallEvent, rainfall_rate, rainfall_rate_ms, total_rainfall, duration, peak_intensity
export Scenario

# Export infiltration (included above for dependency ordering)
export InfiltrationParameters, InfiltrationState, available_storage
export infiltration_rate, apply_infiltration!, total_infiltration

# =============================================================================
# Numerics
# =============================================================================
include("numerics/timestep.jl")
include("numerics/flux.jl")
include("numerics/boundaries.jl")

# Export numerics
export TimestepController, compute_dt, compute_dt_array, compute_dt_smooth!, check_cfl
export compute_velocity, water_surface_elevation, water_surface_elevation!
export surface_gradient, surface_gradient!
export is_wet, wet_dry_factor, limit_flux_wetdry!, limit_froude
export face_depth_x, face_depth_y, compute_flux_x!, compute_flux_y!
export BoundaryType, CLOSED, OPEN, FIXED_DEPTH, BoundaryCondition
export apply_boundaries!, apply_closed_boundaries!, apply_open_boundaries!
export apply_fixed_depth_boundaries!, enforce_positive_depth!

# =============================================================================
# Physics (infiltration already included above)
# =============================================================================
include("physics/friction.jl")
include("physics/rainfall.jl")
include("physics/mass_balance.jl")

# Export physics (infiltration exports already above)
export friction_slope, friction_factor, apply_friction!
export apply_rainfall!, apply_rainfall_spatial!, cumulative_rainfall
export MassBalance, reset!, update_volume!, add_rainfall!, add_outflow!, add_infiltration!
export mass_error, relative_mass_error, compute_mass_balance, check_mass_balance

# =============================================================================
# IO
# =============================================================================
include("io/readers.jl")
include("io/writers.jl")
include("io/validation.jl")

# Export IO
export read_geotiff, read_rainfall_csv, read_points_geojson, read_array
export read_scenario_toml, load_scenario_from_toml
export write_geotiff, write_hydrograph_csv, write_results_json, write_array
export ResultsPackage, write_results
export validate_dem, validate_roughness, validate_rainfall, validate_scenario

# =============================================================================
# Models
# =============================================================================
include("models/surface2d.jl")
include("models/simulation.jl")

# Export models
export SimulationWorkspace, step!, update_depth!, run_simulation!, run_simulation_simple!
export SimulationResults, log_progress
export ResultsAccumulator, update_results!, record_output!
export hazard_rating, froude_number, hazard_category, summarize_hazard
export RunConfig, create_run_config
export RunMetadata, create_metadata, get_git_commit
export run, run_demo, load_scenario, save_results, save_metadata

# =============================================================================
# CLI
# =============================================================================
include("cli/main.jl")
export cli_run

# =============================================================================
# API (placeholder)
# =============================================================================
include("api/jobs.jl")
include("api/routes.jl")
include("api/server.jl")

export start_server, stop_server, Job

# =============================================================================
# Package Info
# =============================================================================

"""
    version()

Return the HydroForge version string.
"""
version() = HYDROFORGE_VERSION

"""
    info()

Print information about HydroForge.
"""
function info()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║           HydroForge - Urban Flood Risk Simulator            ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║  Version: $(lpad(HYDROFORGE_VERSION, 15))                              ║")
    println("║  Julia:   $(lpad(string(VERSION), 15))                              ║")
    println("║  Status:  Pre-Alpha / Development                            ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║  License: MIT                                                ║")
    println("║  GitHub:  github.com/hydroforge/HydroForge                   ║")
    println("╚══════════════════════════════════════════════════════════════╝")
end

export version, info

end # module HydroForge
