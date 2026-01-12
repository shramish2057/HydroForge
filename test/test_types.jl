# Tests for HydroForge types

using Test
using HydroForge

@testset "Grid" begin
    @testset "Construction" begin
        grid = Grid(100, 100, 10.0)
        @test grid.nx == 100
        @test grid.ny == 100
        @test grid.dx == 10.0
        @test grid.dy == 10.0
        @test grid.crs == "EPSG:4326"
    end

    @testset "Validation" begin
        @test_throws ArgumentError Grid(0, 100, 10.0)
        @test_throws ArgumentError Grid(100, 0, 10.0)
        @test_throws ArgumentError Grid(100, 100, 0.0)
        @test_throws ArgumentError Grid(100, 100, -10.0)
    end

    @testset "Helpers" begin
        grid = Grid(10, 10, 5.0)
        @test cell_area(grid) == 25.0
        @test total_area(grid) == 2500.0

        xmin, xmax, ymin, ymax = extent(grid)
        @test xmin == 0.0
        @test xmax == 50.0
        @test ymin == 0.0
        @test ymax == 50.0
    end
end

@testset "SimulationState" begin
    @testset "Construction" begin
        state = SimulationState(50, 50)
        @test size(state.h) == (50, 50)
        @test all(state.h .== 0)
        @test state.t == 0.0
    end

    @testset "From Grid" begin
        grid = Grid(64, 64, 10.0)
        state = SimulationState(grid)
        @test size(state.h) == (64, 64)
    end

    @testset "Functions" begin
        grid = Grid(10, 10, 1.0)
        state = SimulationState(grid)
        state.h[5, 5] = 1.0

        @test total_volume(state, grid) == 1.0
        @test max_depth(state) == 1.0
        @test wet_cells(state, 0.001) == 1
    end
end

@testset "SimulationParameters" begin
    @testset "Defaults" begin
        params = SimulationParameters()
        @test params.cfl == 0.7
        @test params.g == 9.81
        @test params.h_min == 0.001
    end

    @testset "Custom" begin
        params = SimulationParameters(t_end=7200.0, cfl=0.5)
        @test params.t_end == 7200.0
        @test params.cfl == 0.5
    end

    @testset "Validation" begin
        @test_throws ArgumentError SimulationParameters(cfl=1.5)
        @test_throws ArgumentError SimulationParameters(dt_max=-1.0)
    end
end

@testset "RainfallEvent" begin
    times = [0.0, 1800.0, 3600.0]
    intensities = [0.0, 50.0, 0.0]

    @testset "Construction" begin
        rain = RainfallEvent(times, intensities)
        @test length(rain.times) == 3
        @test peak_intensity(rain) == 50.0
    end

    @testset "Interpolation" begin
        rain = RainfallEvent(times, intensities)
        @test rainfall_rate(rain, 0.0) == 0.0
        @test rainfall_rate(rain, 1800.0) == 50.0
        @test rainfall_rate(rain, 900.0) ≈ 25.0  # Linear interpolation
    end

    @testset "Total rainfall" begin
        rain = RainfallEvent(times, intensities)
        total = total_rainfall(rain)
        @test total > 0  # Should have some rainfall
    end
end

@testset "Topography" begin
    @testset "Slope Computation" begin
        grid = Grid(5, 5, 10.0)
        elevation = zeros(5, 5)
        # Create a simple slope in x direction
        for i in 1:5, j in 1:5
            elevation[i, j] = Float64(i)
        end

        topo = Topography(elevation, 0.03, grid)

        # Slope should be approximately 1/10 = 0.1
        @test all(abs.(topo.slope_x .- 0.1) .< 0.01)
    end
end
