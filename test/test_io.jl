# Tests for HydroForge IO functions

using Test
using HydroForge
using DelimitedFiles

@testset "IO Readers" begin
    # Create temp directory for test files
    temp_dir = mktempdir()

    @testset "read_array" begin
        # Create test file
        test_data = [1.0 2.0 3.0; 4.0 5.0 6.0]
        test_path = joinpath(temp_dir, "test_array.txt")
        writedlm(test_path, test_data, ' ')

        # Test reading
        result = read_array(test_path)
        @test size(result) == (2, 3)
        @test result[1, 1] == 1.0
        @test result[2, 3] == 6.0

        # Test missing file
        @test_throws ArgumentError read_array(joinpath(temp_dir, "nonexistent.txt"))
    end

    @testset "read_rainfall_csv" begin
        # Create test CSV
        csv_path = joinpath(temp_dir, "rainfall.csv")
        open(csv_path, "w") do io
            println(io, "time,intensity")
            println(io, "0,0")
            println(io, "1800,30")
            println(io, "3600,0")
        end

        # Test reading
        rainfall = read_rainfall_csv(csv_path)
        @test length(rainfall.times) == 3
        @test rainfall.times[1] == 0.0
        @test rainfall.times[end] == 3600.0
        @test rainfall.intensities[2] == 30.0

        # Test missing file
        @test_throws ArgumentError read_rainfall_csv(joinpath(temp_dir, "missing.csv"))
    end

    @testset "read_scenario_toml" begin
        # Create test scenario
        toml_path = joinpath(temp_dir, "scenario.toml")
        open(toml_path, "w") do io
            println(io, """
            [scenario]
            name = "Test Scenario"

            [grid]
            nx = 10
            ny = 10
            dx = 5.0

            [input]
            dem = "dem.txt"
            roughness = 0.03
            rainfall = "rainfall.csv"

            [parameters]
            t_end = 3600.0
            """)
        end

        # Create DEM
        dem = fill(10.0, 10, 10)
        writedlm(joinpath(temp_dir, "dem.txt"), dem, ' ')

        # Create rainfall
        open(joinpath(temp_dir, "rainfall.csv"), "w") do io
            println(io, "time,intensity")
            println(io, "0,0")
            println(io, "3600,0")
        end

        # Test reading TOML
        config = read_scenario_toml(toml_path)
        @test config["scenario"]["name"] == "Test Scenario"
        @test config["grid"]["nx"] == 10

        # Test loading full scenario
        scenario = load_scenario_from_toml(toml_path)
        @test scenario.name == "Test Scenario"
        @test scenario.grid.nx == 10
        @test scenario.grid.dx == 5.0

        # Test missing file
        @test_throws ArgumentError read_scenario_toml(joinpath(temp_dir, "missing.toml"))
    end

    # Cleanup
    rm(temp_dir; recursive=true)
end

@testset "IO Writers" begin
    temp_dir = mktempdir()

    @testset "write_array" begin
        test_data = [1.0 2.0; 3.0 4.0]
        out_path = joinpath(temp_dir, "output.txt")

        write_array(out_path, test_data)
        @test isfile(out_path)

        # Verify content
        result = readdlm(out_path)
        @test result == test_data
    end

    @testset "write_hydrograph_csv" begin
        hydrograph = [(0.0, 0.0), (60.0, 0.1), (120.0, 0.2)]
        out_path = joinpath(temp_dir, "hydrograph.csv")

        write_hydrograph_csv(out_path, hydrograph)
        @test isfile(out_path)

        # Verify content
        lines = readlines(out_path)
        @test lines[1] == "time,depth"
        @test startswith(lines[2], "0.0,0.0")
    end

    @testset "write_results_json" begin
        metadata = Dict{String,Any}(
            "run_id" => "test_123",
            "max_depth" => 0.5,
        )
        out_path = joinpath(temp_dir, "metadata.json")

        write_results_json(out_path, metadata)
        @test isfile(out_path)

        # Verify JSON is valid
        content = read(out_path, String)
        @test occursin("test_123", content)
        @test occursin("max_depth", content)
    end

    # Cleanup
    rm(temp_dir; recursive=true)
end

@testset "IO Validation" begin
    @testset "validate_dem" begin
        grid = Grid(10, 10, 5.0)

        # Valid DEM
        valid_dem = rand(10, 10)
        @test validate_dem(valid_dem, grid) == true

        # Wrong dimensions
        wrong_size = rand(5, 5)
        @test_throws DimensionMismatch validate_dem(wrong_size, grid)

        # Contains NaN
        nan_dem = rand(10, 10)
        nan_dem[5, 5] = NaN
        @test_throws ArgumentError validate_dem(nan_dem, grid)

        # Contains Inf
        inf_dem = rand(10, 10)
        inf_dem[5, 5] = Inf
        @test_throws ArgumentError validate_dem(inf_dem, grid)
    end

    @testset "validate_roughness" begin
        grid = Grid(10, 10, 5.0)

        # Valid roughness
        valid_n = fill(0.03, 10, 10)
        @test validate_roughness(valid_n, grid) == true

        # Wrong dimensions
        wrong_size = fill(0.03, 5, 5)
        @test_throws DimensionMismatch validate_roughness(wrong_size, grid)

        # Negative value
        negative_n = fill(0.03, 10, 10)
        negative_n[1, 1] = -0.01
        @test_throws ArgumentError validate_roughness(negative_n, grid)

        # Zero value
        zero_n = fill(0.03, 10, 10)
        zero_n[1, 1] = 0.0
        @test_throws ArgumentError validate_roughness(zero_n, grid)
    end

    @testset "validate_rainfall" begin
        # Valid rainfall
        valid = RainfallEvent([0.0, 1800.0, 3600.0], [0.0, 30.0, 0.0])
        @test validate_rainfall(valid) == true
    end
end

@testset "Round-trip IO" begin
    temp_dir = mktempdir()

    # Create scenario, write it, load it back
    grid = Grid(20, 20, 10.0)
    elevation = zeros(20, 20)
    for i in 1:20, j in 1:20
        elevation[i, j] = 10.0 + 0.01 * i
    end

    topo = Topography(elevation, 0.03, grid)
    rainfall = RainfallEvent([0.0, 1800.0, 3600.0], [0.0, 50.0, 0.0])
    params = SimulationParameters(t_end=3600.0)
    scenario = Scenario("Round-trip Test", grid, topo, params, rainfall, Tuple{Int,Int}[], temp_dir)

    # Write inputs
    writedlm(joinpath(temp_dir, "dem.txt"), elevation, ' ')
    open(joinpath(temp_dir, "rainfall.csv"), "w") do io
        println(io, "time,intensity")
        for (t, i) in zip(rainfall.times, rainfall.intensities)
            println(io, "$t,$i")
        end
    end

    # Write scenario TOML
    open(joinpath(temp_dir, "scenario.toml"), "w") do io
        println(io, """
        [scenario]
        name = "Round-trip Test"

        [grid]
        nx = 20
        ny = 20
        dx = 10.0

        [input]
        dem = "dem.txt"
        roughness = 0.03
        rainfall = "rainfall.csv"

        [parameters]
        t_end = 3600.0
        """)
    end

    # Load back
    loaded = load_scenario_from_toml(joinpath(temp_dir, "scenario.toml"))
    @test loaded.name == "Round-trip Test"
    @test loaded.grid.nx == 20
    @test loaded.grid.dx == 10.0
    @test size(loaded.topography.elevation) == (20, 20)

    # Cleanup
    rm(temp_dir; recursive=true)
end
