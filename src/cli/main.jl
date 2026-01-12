# HydroForge CLI
# Command-line interface (placeholder for Phase 11)

"""
    cli_run(args)

Main CLI entry point.

# Commands
- `run <scenario>`: Run simulation from scenario file
- `demo`: Run the bundled demo scenario
- `validate <scenario>`: Validate a scenario file
- `version`: Show version information
"""
function cli_run(args::Vector{String})
    if isempty(args)
        println("HydroForge - Real-Time Urban Flood Risk Simulator")
        println("Usage: hydroforge <command> [options]")
        println()
        println("Commands:")
        println("  run <scenario.toml>   Run a simulation")
        println("  demo                  Run the demo scenario")
        println("  validate <scenario>   Validate scenario file")
        println("  version               Show version")
        return
    end

    cmd = args[1]

    if cmd == "version"
        println("HydroForge v$(HYDROFORGE_VERSION)")
    elseif cmd == "demo"
        run_demo()
    elseif cmd == "run"
        if length(args) < 2
            error("Missing scenario path")
        end
        run(args[2])
    elseif cmd == "validate"
        if length(args) < 2
            error("Missing scenario path")
        end
        println("Validation not yet implemented")
    else
        error("Unknown command: $cmd")
    end
end
