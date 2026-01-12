# HydroForge Simulation Runner
# High-level simulation execution

using Dates
using Random

# Note: ResultsAccumulator, update_results!, and record_output! are defined in surface2d.jl

"""
    RunConfig

Configuration for a simulation run.
"""
struct RunConfig
    scenario_path::String
    output_dir::String
    run_id::String
    save_snapshots::Bool
    snapshot_interval::Float64
end

"""
    create_run_config(scenario_path; output_dir=nothing, save_snapshots=false)

Create run configuration.
"""
function create_run_config(scenario_path::String;
                           output_dir::Union{String,Nothing}=nothing,
                           save_snapshots::Bool=false,
                           snapshot_interval::Float64=300.0)
    # Generate run ID
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    random_suffix = randstring(6)
    run_id = "$(timestamp)_$(random_suffix)"

    # Set output directory
    if output_dir === nothing
        output_dir = joinpath(dirname(scenario_path), "runs", run_id)
    end

    # Create output directory
    mkpath(output_dir)

    RunConfig(scenario_path, output_dir, run_id, save_snapshots, snapshot_interval)
end


"""
    RunMetadata

Metadata for a simulation run.
"""
mutable struct RunMetadata
    run_id::String
    start_time::DateTime
    end_time::Union{DateTime, Nothing}
    julia_version::String
    hydroforge_version::String
    git_commit::Union{String, Nothing}
    scenario_name::String
    parameters::Dict{String, Any}
    status::Symbol  # :running, :completed, :failed
    error_message::Union{String, Nothing}
end

"""
    create_metadata(config::RunConfig, scenario_name::String, params)

Create run metadata.
"""
function create_metadata(config::RunConfig, scenario_name::String, params)
    RunMetadata(
        config.run_id,
        now(),
        nothing,
        string(VERSION),
        HYDROFORGE_VERSION,
        get_git_commit(),
        scenario_name,
        Dict{String, Any}(
            "t_end" => params.t_end,
            "dt_max" => params.dt_max,
            "cfl" => params.cfl
        ),
        :running,
        nothing
    )
end

"""
    get_git_commit()

Try to get current git commit hash.
"""
function get_git_commit()
    try
        strip(read(`git rev-parse HEAD`, String))
    catch
        nothing
    end
end


"""
    run(scenario_path::String; output_dir=nothing)

Run a simulation from scenario file.

# Arguments
- `scenario_path`: Path to scenario TOML file
- `output_dir`: Output directory (default: runs/<run_id> in scenario directory)

# Returns
- Results summary dictionary
"""
function run(scenario_path::String; output_dir::Union{String,Nothing}=nothing)
    # Create run configuration
    config = create_run_config(scenario_path; output_dir=output_dir)

    @info "Starting HydroForge simulation" run_id=config.run_id

    # Load scenario
    scenario = load_scenario(scenario_path)

    # Create metadata
    metadata = create_metadata(config, scenario.name, scenario.parameters)

    try
        # Initialize state
        state = SimulationState(scenario.grid)

        # Run simulation
        @info "Running simulation..." t_end=scenario.parameters.t_end
        sim_results = run_simulation!(state, scenario; verbosity=0)

        # Extract accumulator for saving
        results = sim_results.accumulator

        # Save results
        save_results(config, results, scenario)

        # Update metadata
        metadata.end_time = now()
        metadata.status = :completed
        save_metadata(config, metadata)

        @info "Simulation completed" run_id=config.run_id output_dir=config.output_dir steps=sim_results.step_count wall_time=round(sim_results.wall_time, digits=2)

        # Return summary
        Dict(
            "run_id" => config.run_id,
            "status" => :completed,
            "output_dir" => config.output_dir,
            "max_depth" => maximum(results.max_depth),
            "steps" => sim_results.step_count,
            "wall_time" => sim_results.wall_time,
            "mass_error_pct" => relative_mass_error(sim_results.mass_balance) * 100
        )

    catch e
        metadata.end_time = now()
        metadata.status = :failed
        metadata.error_message = string(e)
        save_metadata(config, metadata)
        rethrow()
    end
end

"""
    run_demo()

Run the bundled demo scenario.
"""
function run_demo()
    demo_path = joinpath(@__DIR__, "..", "..", "assets", "demo_data", "scenario.toml")

    if !isfile(demo_path)
        error("Demo scenario not found at $demo_path")
    end

    run(demo_path)
end


# IO Integration

"""
    ResultsPackage(results::ResultsAccumulator, metadata::Dict)

Create a ResultsPackage from a ResultsAccumulator and metadata.
"""
function ResultsPackage(results::ResultsAccumulator{T}, metadata::Dict{String,Any}) where T
    ResultsPackage{T}(
        results.max_depth,
        results.arrival_time,
        results.max_velocity,
        results.point_hydrographs,
        metadata
    )
end

"""
    write_results(output_dir::String, results::ResultsAccumulator, scenario::Scenario, run_id::String)

Convenience function to write results with auto-generated metadata.
"""
function write_results(output_dir::String, results::ResultsAccumulator{T},
                       scenario::Scenario{T}, run_id::String) where T
    metadata = Dict{String,Any}(
        "run_id" => run_id,
        "scenario_name" => scenario.name,
        "timestamp" => string(now()),
        "grid_nx" => scenario.grid.nx,
        "grid_ny" => scenario.grid.ny,
        "grid_dx" => scenario.grid.dx,
        "t_end" => scenario.parameters.t_end,
        "max_depth_overall" => maximum(results.max_depth),
        "total_rainfall_mm" => total_rainfall(scenario.rainfall),
    )

    package = ResultsPackage(results, metadata)
    write_results(output_dir, package, scenario.grid)
end

"""
    load_scenario(path::String)

Load a scenario from a TOML file.
"""
function load_scenario(path::String)
    load_scenario_from_toml(path)
end

"""
    save_results(config::RunConfig, results::ResultsAccumulator, scenario::Scenario)

Save simulation results to the output directory.
"""
function save_results(config::RunConfig, results::ResultsAccumulator{T},
                      scenario::Scenario{T}) where T
    write_results(config.output_dir, results, scenario, config.run_id)
end

"""
    save_metadata(config::RunConfig, metadata::RunMetadata)

Save run metadata to JSON file.
"""
function save_metadata(config::RunConfig, metadata::RunMetadata)
    metadata_dict = Dict{String,Any}(
        "run_id" => metadata.run_id,
        "start_time" => string(metadata.start_time),
        "end_time" => metadata.end_time === nothing ? nothing : string(metadata.end_time),
        "julia_version" => metadata.julia_version,
        "hydroforge_version" => metadata.hydroforge_version,
        "git_commit" => metadata.git_commit,
        "scenario_name" => metadata.scenario_name,
        "parameters" => metadata.parameters,
        "status" => string(metadata.status),
        "error_message" => metadata.error_message,
    )
    write_results_json(joinpath(config.output_dir, "run_metadata.json"), metadata_dict)
end
