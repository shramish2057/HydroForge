# HydroForge CLI Tests
# Tests for the command-line interface

using Test
using HydroForge

@testset "CLI" begin
    # Test helper to capture CLI output using sprint
    function capture_cli(args)
        output = sprint() do io
            redirect_stdout(io) do
                try
                    cli_run(args)
                catch e
                    # Some commands may throw on invalid input - that's expected
                    if !(e isa SystemExit || e isa InterruptException)
                        println(io, "[ERROR] ", sprint(showerror, e))
                    end
                end
            end
        end
        return output
    end

    # Alternative: just call cli_run and check it doesn't error
    function run_cli_no_error(args)
        try
            cli_run(args)
            return true
        catch e
            if e isa SystemExit || e isa InterruptException
                return true  # These are expected for some commands
            end
            return false
        end
    end

    @testset "CLI Module Exports" begin
        # Verify cli_run is exported and callable
        @test isdefined(HydroForge, :cli_run)
        @test cli_run isa Function

        # Check the version constant is accessible
        @test HydroForge.HYDROFORGE_VERSION isa String
        @test length(HydroForge.HYDROFORGE_VERSION) > 0
    end

    @testset "Help Commands" begin
        # Test that help commands don't error
        @test run_cli_no_error(["--help"])
        @test run_cli_no_error(["-h"])
        @test run_cli_no_error(["run", "--help"])
        @test run_cli_no_error(["run", "-h"])
        @test run_cli_no_error(["run-coupled", "--help"])
        @test run_cli_no_error(["validate", "--help"])
        @test run_cli_no_error(["benchmark", "--help"])
        @test run_cli_no_error(["network", "--help"])
        @test run_cli_no_error(["info"])
    end

    @testset "Version Command" begin
        @test run_cli_no_error(["--version"])
        @test run_cli_no_error(["-V"])
    end

    @testset "Benchmark Command" begin
        @test run_cli_no_error(["benchmark"])
    end

    @testset "Network Subcommand" begin
        @test run_cli_no_error(["network"])
        @test run_cli_no_error(["network", "--help"])
    end

    @testset "Error Handling" begin
        # These should not crash, but may print errors
        @test run_cli_no_error(["unknowncommand"])
        @test run_cli_no_error(["run"])  # Missing required file
        @test run_cli_no_error(["run-coupled"])  # Missing required files
        @test run_cli_no_error(["validate", "nonexistent.toml"])  # Missing file
    end
end
