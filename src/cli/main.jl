# HydroForge CLI
# Production-grade command-line interface for urban flood simulation
#
# Supports all core functionality:
# - Surface flow simulation (2D)
# - Drainage network simulation (1D)
# - Coupled 1D-2D simulation
# - Scenario validation
# - Performance benchmarking

using Dates
using Printf

# =============================================================================
# ANSI Color Codes for Terminal Output
# =============================================================================

const COLORS = Dict(
    :reset => "\e[0m",
    :bold => "\e[1m",
    :dim => "\e[2m",
    :red => "\e[31m",
    :green => "\e[32m",
    :yellow => "\e[33m",
    :blue => "\e[34m",
    :magenta => "\e[35m",
    :cyan => "\e[36m",
    :white => "\e[37m",
    :bg_red => "\e[41m",
    :bg_green => "\e[42m",
    :bg_blue => "\e[44m",
)

# Check if terminal supports colors
const USE_COLORS = get(ENV, "NO_COLOR", "") == "" && (Sys.isunix() || Sys.iswindows())

function colorize(text::String, colors::Symbol...)
    if !USE_COLORS
        return text
    end
    prefix = join([COLORS[c] for c in colors])
    return prefix * text * COLORS[:reset]
end

# Convenience functions
bold(s) = colorize(s, :bold)
dim(s) = colorize(s, :dim)
red(s) = colorize(s, :red)
green(s) = colorize(s, :green)
yellow(s) = colorize(s, :yellow)
blue(s) = colorize(s, :blue)
cyan(s) = colorize(s, :cyan)
magenta(s) = colorize(s, :magenta)

# Status indicators
function status_ok(msg)
    println("$(green("[OK]")) $msg")
end

function status_warn(msg)
    println("$(yellow("[WARN]")) $msg")
end

function status_error(msg)
    println("$(red("[ERROR]")) $msg")
end

function status_info(msg)
    println("$(blue("[INFO]")) $msg")
end

# =============================================================================
# Progress Bar
# =============================================================================

mutable struct ProgressBar
    total::Int
    current::Int
    width::Int
    start_time::Float64
    description::String
    show_eta::Bool
end

function ProgressBar(total::Int; width::Int=40, description::String="Progress", show_eta::Bool=true)
    ProgressBar(total, 0, width, time(), description, show_eta)
end

function update!(pb::ProgressBar, current::Int)
    pb.current = current
    render(pb)
end

function increment!(pb::ProgressBar, n::Int=1)
    pb.current += n
    render(pb)
end

function render(pb::ProgressBar)
    pct = pb.current / pb.total
    filled = round(Int, pct * pb.width)
    empty = pb.width - filled

    bar = "█"^filled * "░"^empty
    pct_str = @sprintf("%5.1f%%", pct * 100)

    # ETA calculation
    elapsed = time() - pb.start_time
    eta_str = ""
    if pb.show_eta && pb.current > 0 && pb.current < pb.total
        rate = pb.current / elapsed
        remaining = (pb.total - pb.current) / rate
        if remaining < 60
            eta_str = @sprintf(" ETA: %ds", round(Int, remaining))
        elseif remaining < 3600
            eta_str = @sprintf(" ETA: %dm%ds", div(remaining, 60), mod(round(Int, remaining), 60))
        else
            eta_str = @sprintf(" ETA: %dh%dm", div(remaining, 3600), div(mod(remaining, 3600), 60))
        end
    elseif pb.current >= pb.total
        eta_str = @sprintf(" (%.1fs)", elapsed)
    end

    print("\r$(pb.description): |$(cyan(bar))| $pct_str [$(pb.current)/$(pb.total)]$eta_str")
    if pb.current >= pb.total
        println()
    end
end

function finish!(pb::ProgressBar)
    pb.current = pb.total
    render(pb)
end

# =============================================================================
# CLI Argument Parsing
# =============================================================================

struct CLIOption
    short::String
    long::String
    description::String
    has_value::Bool
    default::Any
    required::Bool
end

struct CLICommand
    name::String
    description::String
    usage::String
    options::Vector{CLIOption}
    handler::Function
end

mutable struct ParsedArgs
    command::String
    positional::Vector{String}
    options::Dict{String, Any}
end

function parse_args(args::Vector{String}, commands::Vector{CLICommand})
    if isempty(args)
        return ParsedArgs("", String[], Dict{String,Any}())
    end

    # Check for global options first
    if args[1] in ["--help", "-h"]
        return ParsedArgs("help", String[], Dict{String,Any}())
    elseif args[1] in ["--version", "-V"]
        return ParsedArgs("version", String[], Dict{String,Any}())
    end

    cmd_name = args[1]
    cmd = findfirst(c -> c.name == cmd_name, commands)

    if cmd === nothing
        return ParsedArgs("unknown", [cmd_name], Dict{String,Any}())
    end

    command = commands[cmd]
    positional = String[]
    options = Dict{String, Any}()

    # Set defaults
    for opt in command.options
        if opt.default !== nothing
            options[opt.long] = opt.default
        end
    end

    i = 2
    while i <= length(args)
        arg = args[i]

        if startswith(arg, "--")
            # Long option
            key = arg[3:end]
            opt = findfirst(o -> o.long == key, command.options)
            if opt !== nothing
                if command.options[opt].has_value
                    if i + 1 <= length(args)
                        i += 1
                        options[key] = args[i]
                    else
                        error("Option --$key requires a value")
                    end
                else
                    options[key] = true
                end
            else
                error("Unknown option: $arg")
            end
        elseif startswith(arg, "-") && length(arg) == 2
            # Short option
            key = arg[2:2]
            opt = findfirst(o -> o.short == key, command.options)
            if opt !== nothing
                if command.options[opt].has_value
                    if i + 1 <= length(args)
                        i += 1
                        options[command.options[opt].long] = args[i]
                    else
                        error("Option -$key requires a value")
                    end
                else
                    options[command.options[opt].long] = true
                end
            else
                error("Unknown option: $arg")
            end
        else
            push!(positional, arg)
        end
        i += 1
    end

    ParsedArgs(cmd_name, positional, options)
end

# =============================================================================
# Command Handlers
# =============================================================================

function cmd_help(args::ParsedArgs, commands::Vector{CLICommand})
    println()
    println(bold("HydroForge") * " - Real-Time Urban Flood Risk Simulator")
    println(dim("Version $HYDROFORGE_VERSION"))
    println()
    println(bold("USAGE:"))
    println("    hydroforge <command> [options] [arguments]")
    println()
    println(bold("COMMANDS:"))

    max_len = maximum(length(c.name) for c in commands)
    for cmd in commands
        padding = " "^(max_len - length(cmd.name) + 2)
        println("    $(cyan(cmd.name))$padding$(cmd.description)")
    end

    println()
    println(bold("OPTIONS:"))
    println("    $(cyan("-h, --help"))      Show this help message")
    println("    $(cyan("-V, --version"))   Show version information")
    println()
    println("Use $(cyan("hydroforge <command> --help")) for command-specific help.")
    println()
end

function cmd_help_command(cmd::CLICommand)
    println()
    println(bold(cmd.name) * " - $(cmd.description)")
    println()
    println(bold("USAGE:"))
    println("    $(cmd.usage)")
    println()

    if !isempty(cmd.options)
        println(bold("OPTIONS:"))
        for opt in cmd.options
            short = opt.short == "" ? "  " : "-$(opt.short),"
            long = "--$(opt.long)"
            value = opt.has_value ? " <value>" : ""
            default = opt.default !== nothing ? dim(" (default: $(opt.default))") : ""
            required = opt.required ? red(" [required]") : ""
            println("    $short $(cyan(long * value))$default$required")
            println("        $(opt.description)")
        end
        println()
    end
end

function cmd_version(args::ParsedArgs)
    println()
    println(bold("HydroForge") * " v$HYDROFORGE_VERSION")
    println()
    println("Julia version: $(VERSION)")
    println("Platform: $(Sys.MACHINE)")
    println()

    # Try to get git info
    commit = get_git_commit()
    if commit !== nothing
        println("Git commit: $(commit[1:8])")
    end
    println()
end

function cmd_info(args::ParsedArgs)
    info()
    println()
    println(bold("Capabilities:"))
    println("  $(green("✓")) 2D Surface Flow (Local Inertial Approximation)")
    println("  $(green("✓")) Green-Ampt Infiltration Model")
    println("  $(green("✓")) DEFRA/EA FD2320 Hazard Analysis")
    println("  $(green("✓")) 1D Drainage Network Routing")
    println("  $(green("✓")) 1D-2D Coupled Simulation")
    println("  $(green("✓")) Inlet/Outlet Modeling (Weir/Orifice)")
    println()
end

function cmd_run(args::ParsedArgs)
    if "help" in keys(args.options)
        return nothing  # Help already shown
    end

    if isempty(args.positional)
        status_error("Missing scenario path")
        println("Usage: hydroforge run <scenario.toml> [options]")
        return 1
    end

    scenario_path = args.positional[1]

    if !isfile(scenario_path)
        status_error("Scenario file not found: $scenario_path")
        return 1
    end

    # Parse options
    output_dir = get(args.options, "output", nothing)
    save_snapshots = get(args.options, "snapshots", false) == true
    snapshot_interval = parse(Float64, get(args.options, "snapshot-interval", "300.0"))
    enable_profiling = get(args.options, "profile", false) == true
    verbosity = parse(Int, get(args.options, "verbosity", "1"))
    format = get(args.options, "format", "json")

    println()
    println(bold("HydroForge Simulation"))
    println(dim("="^50))
    println()

    status_info("Loading scenario: $(basename(scenario_path))")

    try
        # Validate first
        issues = validate_scenario_file(scenario_path)
        if !isempty(issues)
            status_warn("Scenario has $(length(issues)) issue(s):")
            for issue in issues
                println("  $(yellow("•")) $issue")
            end
            if any(occursin("error", lowercase(i)) for i in issues)
                status_error("Cannot proceed due to errors")
                return 1
            end
        end

        # Run simulation
        status_info("Starting simulation...")
        println()

        results = run(scenario_path;
                     output_dir=output_dir,
                     save_snapshots=save_snapshots,
                     snapshot_interval=snapshot_interval,
                     enable_profiling=enable_profiling,
                     verbosity=verbosity)

        println()
        println(dim("="^50))
        println(bold("Simulation Complete"))
        println()

        # Summary
        println("$(bold("Run ID:"))        $(results["run_id"])")
        println("$(bold("Status:"))        $(green(string(results["status"])))")
        println("$(bold("Output:"))        $(results["output_dir"])")
        println()
        println("$(bold("Results:"))")
        println("  Steps:         $(results["steps"])")
        println("  Wall time:     $(@sprintf("%.2f", results["wall_time"]))s")
        println("  Max depth:     $(@sprintf("%.3f", results["max_depth"]))m")
        println("  Max velocity:  $(@sprintf("%.3f", results["max_velocity"]))m/s")
        println("  Max hazard:    $(@sprintf("%.3f", results["max_hazard"]))m²/s")
        println("  Mass error:    $(@sprintf("%.4f", results["mass_error_pct"]))%")
        println()

        # Hazard summary
        hazard = results["hazard_summary"]
        println("$(bold("Hazard Analysis (DEFRA FD2320):"))")
        println("  Low hazard:    $(@sprintf("%.1f", hazard["area_low_hazard"]))m² ($(hazard["cells_low_hazard"]) cells)")
        println("  Moderate:      $(@sprintf("%.1f", hazard["area_moderate_hazard"]))m² ($(hazard["cells_moderate_hazard"]) cells)")
        println("  Significant:   $(@sprintf("%.1f", hazard["area_significant_hazard"]))m² ($(hazard["cells_significant_hazard"]) cells)")
        println("  Extreme:       $(@sprintf("%.1f", hazard["area_extreme_hazard"]))m² ($(hazard["cells_extreme_hazard"]) cells)")
        println()

        status_ok("Simulation completed successfully")
        return 0

    catch e
        println()
        status_error("Simulation failed: $(sprint(showerror, e))")
        if verbosity >= 2
            println()
            println(dim("Stack trace:"))
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
        return 1
    end
end

function cmd_run_coupled(args::ParsedArgs)
    if "help" in keys(args.options)
        return nothing
    end

    if length(args.positional) < 2
        status_error("Missing scenario or network path")
        println("Usage: hydroforge run-coupled <scenario.toml> <network.toml> [options]")
        return 1
    end

    scenario_path = args.positional[1]
    network_path = args.positional[2]

    if !isfile(scenario_path)
        status_error("Scenario file not found: $scenario_path")
        return 1
    end

    if !isfile(network_path)
        status_error("Network file not found: $network_path")
        return 1
    end

    verbosity = parse(Int, get(args.options, "verbosity", "1"))

    println()
    println(bold("HydroForge Coupled 1D-2D Simulation"))
    println(dim("="^50))
    println()

    status_info("Loading scenario: $(basename(scenario_path))")
    status_info("Loading network: $(basename(network_path))")

    try
        # Load scenario
        scenario = load_scenario(scenario_path)

        # Load network (would need a network loader)
        status_warn("Network loading from TOML not yet implemented")
        status_info("Use programmatic API for coupled simulations")

        println()
        println("$(bold("Example usage:"))")
        println(dim("""
        using HydroForge

        # Load scenario
        scenario = load_scenario("$scenario_path")

        # Create drainage network
        j1 = Junction(1, 100.0, 100.0, -2.0, 0.0)
        j2 = Junction(2, 200.0, 100.0, -2.5, -0.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, -2.0, -2.5)
        inlet = Inlet(1, 1, 10, 10)
        outlet = Outlet(1, 2; outlet_type=:FREE)
        network = DrainageNetwork([pipe], [j1, j2], [inlet], [outlet])

        # Create coupled scenario and run
        coupled = CoupledScenario(scenario, network)
        state = CoupledState(scenario.grid, network)
        results = run_coupled!(state, coupled; verbosity=1)
        """))

        return 0

    catch e
        status_error("Failed: $(sprint(showerror, e))")
        return 1
    end
end

function cmd_demo(args::ParsedArgs)
    println()
    println(bold("HydroForge Demo"))
    println(dim("="^50))
    println()

    status_info("Running bundled demo scenario...")
    println()

    try
        results = run_demo()

        println()
        println(dim("="^50))
        status_ok("Demo completed successfully!")
        println()
        println("Results saved to: $(results["output_dir"])")
        println()

        return 0
    catch e
        status_error("Demo failed: $(sprint(showerror, e))")
        return 1
    end
end

function cmd_validate(args::ParsedArgs)
    if isempty(args.positional)
        status_error("Missing file path")
        println("Usage: hydroforge validate <scenario.toml>")
        return 1
    end

    file_path = args.positional[1]

    if !isfile(file_path)
        status_error("File not found: $file_path")
        return 1
    end

    println()
    println(bold("Validating: ") * file_path)
    println()

    issues = validate_scenario_file(file_path)

    if isempty(issues)
        status_ok("Scenario is valid")
        println()

        # Show scenario summary
        try
            scenario = load_scenario(file_path)
            println(bold("Scenario Summary:"))
            println("  Name:       $(scenario.name)")
            println("  Grid:       $(scenario.grid.nx) × $(scenario.grid.ny) cells")
            println("  Cell size:  $(scenario.grid.dx)m × $(scenario.grid.dy)m")
            println("  Domain:     $(scenario.grid.nx * scenario.grid.dx)m × $(scenario.grid.ny * scenario.grid.dy)m")
            println("  Duration:   $(scenario.parameters.t_end)s")
            println("  Rainfall:   $(total_rainfall(scenario.rainfall))mm")
            if scenario.infiltration !== nothing
                println("  Infiltration: $(scenario.infiltration.soil_type)")
            end
            println()
        catch e
            status_warn("Could not load scenario details: $e")
        end

        return 0
    else
        status_error("Validation failed with $(length(issues)) issue(s):")
        println()
        for (i, issue) in enumerate(issues)
            if occursin("error", lowercase(issue))
                println("  $(red("$i.")) $issue")
            else
                println("  $(yellow("$i.")) $issue")
            end
        end
        println()
        return 1
    end
end

function cmd_benchmark(args::ParsedArgs)
    println()
    println(bold("HydroForge Performance Benchmark"))
    println(dim("="^50))
    println()

    grid_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]

    if "quick" in keys(args.options)
        grid_sizes = [(32, 32), (64, 64)]
    elseif "full" in keys(args.options)
        push!(grid_sizes, (512, 512))
    end

    results = []

    for (nx, ny) in grid_sizes
        println(bold("Grid: $(nx)×$(ny)"))

        # Setup
        grid = Grid(nx, ny, 10.0)
        elevation = zeros(nx, ny)
        roughness = fill(0.03, nx, ny)
        topo = Topography(elevation, roughness, grid)
        params = SimulationParameters(t_end=60.0, dt_max=1.0, cfl=0.9)
        rain = RainfallEvent([0.0, 100.0], [36.0, 36.0])
        scenario = Scenario("benchmark", grid, topo, params, rain, Tuple{Int,Int}[], "")

        state = SimulationState(grid)

        # Warmup
        print("  Warmup...")
        run_simulation!(copy_state(state), scenario; verbosity=0)
        println(" $(green("done"))")

        # Benchmark
        print("  Running...")
        t_start = time()
        sim_results = run_simulation!(state, scenario; verbosity=0)
        t_end = time()
        wall_time = t_end - t_start
        println(" $(green("done"))")

        steps = sim_results.step_count
        cells = nx * ny
        cells_per_sec = cells * steps / wall_time

        println("  Steps:      $steps")
        println("  Wall time:  $(@sprintf("%.3f", wall_time))s")
        println("  Cells/sec:  $(@sprintf("%.2e", cells_per_sec))")
        println()

        push!(results, (nx=nx, ny=ny, steps=steps, wall_time=wall_time, cells_per_sec=cells_per_sec))
    end

    # Summary table
    println(bold("Summary:"))
    println(dim("-"^60))
    println("  Grid      │  Steps  │  Time (s)  │  Cells/sec")
    println(dim("-"^60))
    for r in results
        println(@sprintf("  %3d×%-3d   │  %5d  │  %8.3f  │  %.2e", r.nx, r.ny, r.steps, r.wall_time, r.cells_per_sec))
    end
    println(dim("-"^60))
    println()

    status_ok("Benchmark complete")
    return 0
end

function cmd_network_validate(args::ParsedArgs)
    status_info("Network validation from file not yet implemented")
    println()
    println("Use programmatic API to validate networks:")
    println(dim("""
    using HydroForge

    # Create network
    network = DrainageNetwork(pipes, junctions, inlets, outlets)

    # Validate
    issues = validate(network)
    if isempty(issues)
        println("Network is valid")
    else
        for issue in issues
            println(issue)
        end
    end
    """))
    return 0
end

function cmd_network_info(args::ParsedArgs)
    status_info("Network info from file not yet implemented")
    return 0
end

# =============================================================================
# Command Definitions
# =============================================================================

const CLI_COMMANDS = [
    CLICommand(
        "run",
        "Run a surface flow simulation",
        "hydroforge run <scenario.toml> [options]",
        [
            CLIOption("o", "output", "Output directory", true, nothing, false),
            CLIOption("", "snapshots", "Save intermediate snapshots", false, false, false),
            CLIOption("", "snapshot-interval", "Interval between snapshots (seconds)", true, "300.0", false),
            CLIOption("", "profile", "Enable performance profiling", false, false, false),
            CLIOption("v", "verbosity", "Verbosity level (0-2)", true, "1", false),
            CLIOption("", "format", "Output format (json, csv)", true, "json", false),
            CLIOption("h", "help", "Show help for this command", false, nothing, false),
        ],
        cmd_run
    ),
    CLICommand(
        "run-coupled",
        "Run a coupled 1D-2D simulation",
        "hydroforge run-coupled <scenario.toml> <network.toml> [options]",
        [
            CLIOption("o", "output", "Output directory", true, nothing, false),
            CLIOption("v", "verbosity", "Verbosity level (0-2)", true, "1", false),
            CLIOption("h", "help", "Show help for this command", false, nothing, false),
        ],
        cmd_run_coupled
    ),
    CLICommand(
        "demo",
        "Run the bundled demo scenario",
        "hydroforge demo",
        [],
        cmd_demo
    ),
    CLICommand(
        "validate",
        "Validate a scenario file",
        "hydroforge validate <scenario.toml>",
        [
            CLIOption("", "strict", "Treat warnings as errors", false, false, false),
        ],
        cmd_validate
    ),
    CLICommand(
        "info",
        "Show package information and capabilities",
        "hydroforge info",
        [],
        cmd_info
    ),
    CLICommand(
        "benchmark",
        "Run performance benchmarks",
        "hydroforge benchmark [options]",
        [
            CLIOption("", "quick", "Run quick benchmark (smaller grids)", false, false, false),
            CLIOption("", "full", "Run full benchmark (includes large grids)", false, false, false),
        ],
        cmd_benchmark
    ),
    CLICommand(
        "network",
        "Drainage network utilities",
        "hydroforge network <subcommand>",
        [
            CLIOption("h", "help", "Show help", false, nothing, false),
        ],
        cmd_network_info
    ),
]

# =============================================================================
# Main Entry Point
# =============================================================================

"""
    cli_run(args::Vector{String})

Main CLI entry point for HydroForge.

# Commands
- `run <scenario>`: Run surface flow simulation
- `run-coupled <scenario> <network>`: Run coupled 1D-2D simulation
- `demo`: Run bundled demo scenario
- `validate <scenario>`: Validate scenario file
- `info`: Show package information
- `benchmark`: Run performance benchmarks
- `network`: Drainage network utilities

# Examples
```bash
# Run a simulation
julia --project -e 'using HydroForge; cli_run(["run", "scenario.toml"])'

# Run with options
julia --project -e 'using HydroForge; cli_run(["run", "scenario.toml", "-o", "results", "--snapshots"])'

# Validate scenario
julia --project -e 'using HydroForge; cli_run(["validate", "scenario.toml"])'

# Run benchmark
julia --project -e 'using HydroForge; cli_run(["benchmark", "--quick"])'
```
"""
function cli_run(args::Vector{String}=ARGS)
    try
        parsed = parse_args(args, CLI_COMMANDS)

        if parsed.command == "" || parsed.command == "help"
            cmd_help(parsed, CLI_COMMANDS)
            return 0
        elseif parsed.command == "version"
            cmd_version(parsed)
            return 0
        elseif parsed.command == "unknown"
            status_error("Unknown command: $(parsed.positional[1])")
            println("Run $(cyan("hydroforge --help")) for available commands.")
            return 1
        end

        # Find and execute command
        cmd_idx = findfirst(c -> c.name == parsed.command, CLI_COMMANDS)
        if cmd_idx !== nothing
            cmd = CLI_COMMANDS[cmd_idx]

            # Check for help flag
            if get(parsed.options, "help", false) == true
                cmd_help_command(cmd)
                return 0
            end

            result = cmd.handler(parsed)
            return result === nothing ? 0 : result
        else
            status_error("Command not found: $(parsed.command)")
            return 1
        end

    catch e
        if e isa InterruptException
            println()
            status_warn("Interrupted by user")
            return 130
        else
            status_error("Unexpected error: $(sprint(showerror, e))")
            return 1
        end
    end
end

"""
    main()

Entry point for standalone executable.
"""
function main()
    exit_code = cli_run(ARGS)
    exit(exit_code)
end

# Export for use
export cli_run, main
