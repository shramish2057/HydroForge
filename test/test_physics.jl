# Tests for HydroForge physics modules

using Test
using HydroForge

@testset "Manning Friction" begin
    @testset "Friction Slope" begin
        # Test Manning equation: Sf = n² |q| q / h^(10/3)
        q = 1.0   # m²/s
        h = 1.0   # m
        n = 0.03
        h_min = 0.001

        Sf = friction_slope(q, h, n, h_min)

        # Expected: 0.03² * 1 * 1 / 1^(10/3) = 0.0009
        @test Sf ≈ 0.0009 atol=1e-10
    end

    @testset "Dry Cell" begin
        Sf = friction_slope(1.0, 0.0001, 0.03, 0.001)
        @test Sf == 0.0
    end

    @testset "Friction Factor" begin
        D = friction_factor(1.0, 1.0, 0.03, 0.001, 9.81, 0.1)
        @test D > 1.0  # Should increase discharge decay
    end
end

@testset "Rainfall Application" begin
    @testset "Uniform Rainfall" begin
        h = zeros(10, 10)
        times = [0.0, 3600.0]
        intensities = [36.0, 36.0]  # 36 mm/hr = 0.01 mm/s = 1e-5 m/s
        rain = RainfallEvent(times, intensities)

        dt = 60.0  # 60 seconds

        apply_rainfall!(h, rain, 0.0, dt)

        # Expected depth: 36 mm/hr * (1hr/3600s) * 60s / 1000 = 0.0006 m
        expected = 36.0 / 1000.0 / 3600.0 * 60.0
        @test all(h .≈ expected)
    end

    @testset "Zero Rainfall" begin
        h = zeros(10, 10)
        times = [0.0, 3600.0]
        intensities = [0.0, 0.0]
        rain = RainfallEvent(times, intensities)

        apply_rainfall!(h, rain, 0.0, 60.0)

        @test all(h .== 0)
    end
end

@testset "Cumulative Rainfall" begin
    # Triangular hyetograph
    times = [0.0, 1800.0, 3600.0]
    intensities = [0.0, 50.0, 0.0]  # Peak at 30 minutes
    rain = RainfallEvent(times, intensities)

    # Total should be area of triangle
    # Base = 1 hour, height = 50 mm/hr
    # Area = 0.5 * 1 * 50 = 25 mm
    total = total_rainfall(rain)
    @test total ≈ 25.0 atol=0.1
end

@testset "Infiltration Placeholder" begin
    params = InfiltrationParameters()
    @test infiltration_rate(0.1, params) == 0.0  # Placeholder returns 0
end

@testset "Mass Balance" begin
    # Mass balance tracking (conceptual test)
    grid = Grid(10, 10, 10.0)
    state = SimulationState(grid)

    initial_volume = total_volume(state, grid)
    @test initial_volume == 0.0

    # Add water uniformly
    state.h .= 0.1  # 10 cm everywhere

    final_volume = total_volume(state, grid)
    expected = 10 * 10 * 100 * 0.1  # nx * ny * cell_area * depth
    @test final_volume ≈ expected
end
