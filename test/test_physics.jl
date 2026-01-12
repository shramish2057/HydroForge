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

        # Test with negative discharge (flow in opposite direction)
        Sf_neg = friction_slope(-1.0, 1.0, 0.03, 0.001)
        @test Sf_neg ≈ -0.0009 atol=1e-10  # Should be negative

        # Test with different roughness
        Sf_rough = friction_slope(1.0, 1.0, 0.06, 0.001)
        @test Sf_rough ≈ 0.0036 atol=1e-10  # 4x higher with 2x roughness
    end

    @testset "Dry Cell" begin
        Sf = friction_slope(1.0, 0.0001, 0.03, 0.001)
        @test Sf == 0.0
    end

    @testset "Friction Factor" begin
        D = friction_factor(1.0, 1.0, 0.03, 0.001, 9.81, 0.1)
        @test D > 1.0  # Should increase discharge decay

        # Dry cell should return 1.0
        D_dry = friction_factor(1.0, 0.0005, 0.03, 0.001, 9.81, 0.1)
        @test D_dry == 1.0

        # Higher roughness should give larger factor
        D_rough = friction_factor(1.0, 1.0, 0.06, 0.001, 9.81, 0.1)
        @test D_rough > D
    end

    @testset "apply_friction!" begin
        qx = ones(5, 5)
        qy = ones(5, 5)
        h = ones(5, 5)
        n = fill(0.03, 5, 5)
        h_min = 0.001
        g = 9.81
        dt = 0.1

        apply_friction!(qx, qy, h, n, h_min, g, dt)

        # All discharges should be reduced
        @test all(qx .< 1.0)
        @test all(qy .< 1.0)

        # Test with dry cells
        qx2 = ones(5, 5)
        qy2 = ones(5, 5)
        h2 = zeros(5, 5)
        h2[3, 3] = 1.0  # Only center is wet

        apply_friction!(qx2, qy2, h2, n, h_min, g, dt)

        # Dry cells should have zero discharge
        @test qx2[1, 1] == 0.0
        @test qy2[1, 1] == 0.0
        # Wet cell should have reduced discharge
        @test 0 < qx2[3, 3] < 1.0
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

    @testset "Rainfall Rate Interpolation" begin
        times = [0.0, 1800.0, 3600.0]
        intensities = [0.0, 60.0, 0.0]  # Triangular
        rain = RainfallEvent(times, intensities)

        # At t=0: should be 0
        @test rainfall_rate(rain, 0.0) ≈ 0.0

        # At t=900 (midway to peak): should be 30 mm/hr
        @test rainfall_rate(rain, 900.0) ≈ 30.0 atol=0.1

        # At t=1800 (peak): should be 60 mm/hr
        @test rainfall_rate(rain, 1800.0) ≈ 60.0

        # At t=2700 (descending): should be 30 mm/hr
        @test rainfall_rate(rain, 2700.0) ≈ 30.0 atol=0.1
    end

    @testset "Cumulative Rainfall Accumulation" begin
        h = zeros(5, 5)
        times = [0.0, 3600.0]
        intensities = [36.0, 36.0]  # Constant 36 mm/hr
        rain = RainfallEvent(times, intensities)

        # Apply multiple timesteps
        for t in 0.0:60.0:300.0
            apply_rainfall!(h, rain, t, 60.0)
        end

        # 6 timesteps * 60s each = 360s total
        # 36 mm/hr * (360/3600) hr = 3.6 mm = 0.0036 m
        @test all(isapprox.(h, 0.0036, atol=1e-6))
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

    # Partial cumulative
    partial = cumulative_rainfall(rain, 1800.0)  # Up to peak
    @test partial ≈ 12.5 atol=0.1  # Half of total

    # Before start
    @test cumulative_rainfall(rain, 0.0) ≈ 0.0
end

@testset "Green-Ampt Infiltration" begin
    @testset "Parameter Construction" begin
        # Default parameters (clay loam)
        params = InfiltrationParameters()
        @test params.hydraulic_conductivity ≈ 1e-6
        @test params.suction_head ≈ 0.21
        @test params.porosity ≈ 0.46
        @test params.initial_moisture ≈ 0.2

        # Soil type constructor
        sand_params = InfiltrationParameters(:sand)
        @test sand_params.hydraulic_conductivity ≈ 1.2e-4
        @test sand_params.porosity ≈ 0.44

        clay_params = InfiltrationParameters(:clay)
        @test clay_params.hydraulic_conductivity ≈ 3.0e-7

        # Available storage
        @test available_storage(params) ≈ 0.26  # 0.46 - 0.2
    end

    @testset "Infiltration Rate Calculation" begin
        params = InfiltrationParameters(:sandy_loam)

        # Rate with cumulative infiltration
        rate1 = infiltration_rate(0.1, 0.001, params)
        rate2 = infiltration_rate(0.1, 0.01, params)
        rate3 = infiltration_rate(0.1, 0.1, params)

        # Rate should decrease as cumulative infiltration increases
        @test rate1 > rate2 > rate3
        @test rate1 > 0
        @test rate3 > 0

        # No infiltration with no water
        @test infiltration_rate(0.0, 0.01, params) == 0.0

        # Rate limited by max infiltration
        params_limited = InfiltrationParameters(max_infiltration_depth=0.05)
        @test infiltration_rate(0.1, 0.05, params_limited) == 0.0
    end

    @testset "Infiltration State" begin
        grid = Grid(10, 10, 10.0)
        state = InfiltrationState(grid)

        @test size(state.cumulative) == (10, 10)
        @test all(state.cumulative .== 0)

        # Modify and reset
        state.cumulative[5, 5] = 0.1
        reset!(state)
        @test all(state.cumulative .== 0)
    end

    @testset "Simple apply_infiltration!" begin
        params = InfiltrationParameters(hydraulic_conductivity=1e-5)
        h = fill(0.1, 5, 5)

        apply_infiltration!(h, params, 100.0)

        # Should have infiltrated K * dt = 1e-5 * 100 = 0.001 m
        @test all(h .≈ 0.099)
    end

    @testset "Green-Ampt apply_infiltration!" begin
        params = InfiltrationParameters(:sandy_loam)
        h = fill(0.1, 5, 5)
        infil_state = InfiltrationState(5, 5)

        # Apply infiltration
        infiltrated = apply_infiltration!(h, infil_state, params, 60.0)

        # Should have infiltrated some water
        @test infiltrated > 0
        @test all(h .< 0.1)
        @test all(infil_state.cumulative .> 0)

        # Cumulative should equal depth reduction
        depth_reduction = 0.1 - h[1, 1]
        @test infil_state.cumulative[1, 1] ≈ depth_reduction
    end

    @testset "Infiltration Limiting" begin
        # Test that infiltration is limited by available water
        params = InfiltrationParameters(:sand)  # High conductivity
        h = fill(0.0001, 5, 5)  # Very shallow water
        infil_state = InfiltrationState(5, 5)

        apply_infiltration!(h, infil_state, params, 60.0)

        # Should have infiltrated all available water
        @test all(isapprox.(h, 0.0, atol=1e-10))

        # Test limiting by max depth
        params_limited = InfiltrationParameters(max_infiltration_depth=0.001)
        h2 = fill(0.1, 5, 5)
        infil_state2 = InfiltrationState(5, 5)
        infil_state2.cumulative .= 0.001  # Already at max

        apply_infiltration!(h2, infil_state2, params_limited, 60.0)

        # No additional infiltration
        @test all(h2 .≈ 0.1)
    end

    @testset "Total Infiltration Volume" begin
        grid = Grid(10, 10, 10.0)
        state = InfiltrationState(grid)
        state.cumulative .= 0.01  # 1 cm everywhere

        vol = total_infiltration(state, grid)
        expected = 10 * 10 * 100 * 0.01  # 100 m³
        @test vol ≈ expected
    end
end

@testset "Mass Balance" begin
    @testset "Volume Calculation" begin
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

    @testset "MassBalance Tracker" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)
        state.h .= 0.1

        mb = MassBalance(state, grid)
        @test mb.initial_volume ≈ 1000.0  # 10*10*100*0.1
        @test mb.rainfall_volume == 0.0
        @test mb.outflow_volume == 0.0

        # Add rainfall
        add_rainfall!(mb, 100.0)
        @test mb.rainfall_volume ≈ 100.0

        # Add outflow
        add_outflow!(mb, 50.0)
        @test mb.outflow_volume ≈ 50.0

        # Update current volume
        state.h .= 0.105  # Slightly more water
        update_volume!(mb, state, grid)
        @test mb.current_volume ≈ 1050.0
    end

    @testset "Mass Error Calculation" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)
        state.h .= 0.1

        mb = MassBalance(state, grid)

        # Perfect balance initially
        @test mass_error(mb) ≈ 0.0 atol=1e-10
        @test relative_mass_error(mb) ≈ 0.0 atol=1e-10

        # Add rainfall but don't update volume (simulates error)
        add_rainfall!(mb, 100.0)
        @test mass_error(mb) ≈ 100.0  # 100 m³ unaccounted

        # Update volume to reflect rainfall
        state.h .+= 0.01  # Add 10mm depth = 100 m³
        update_volume!(mb, state, grid)
        @test mass_error(mb) ≈ 0.0 atol=1e-10
    end

    @testset "compute_mass_balance Function" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)
        state.h .= 0.1

        result = compute_mass_balance(state, grid, 1000.0, 100.0, 100.0)

        # initial(1000) + rainfall(100) - outflow(100) - current(1000) = 0
        @test result.error ≈ 0.0 atol=1e-10
        @test result.relative_error ≈ 0.0 atol=1e-10
    end

    @testset "check_mass_balance Tolerance" begin
        mb = MassBalance(Float64)
        mb.initial_volume = 1000.0
        mb.rainfall_volume = 100.0
        mb.outflow_volume = 0.0
        mb.current_volume = 1090.0  # 10 m³ error

        # 10/1100 ≈ 0.9% error - should pass 1% tolerance
        @test check_mass_balance(mb, tolerance=0.01) == true

        mb.current_volume = 1050.0  # 50 m³ error
        # 50/1100 ≈ 4.5% error - should fail 1% tolerance
        @test check_mass_balance(mb, tolerance=0.01) == false
        # Should pass 5% tolerance
        @test check_mass_balance(mb, tolerance=0.05) == true
    end
end
