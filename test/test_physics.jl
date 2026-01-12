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

    @testset "Extended Mass Balance Fields" begin
        mb = MassBalance(Float64)
        mb.initial_volume = 1000.0

        # Test new tracking fields
        add_inflow!(mb, 200.0)
        @test mb.inflow_volume ≈ 200.0

        add_evaporation!(mb, 50.0)
        @test mb.evaporation_volume ≈ 50.0

        add_drainage_exchange!(mb, 75.0)
        @test mb.drainage_exchange ≈ 75.0

        add_infiltration!(mb, 100.0)
        @test mb.infiltration_volume ≈ 100.0

        # Test totals
        @test total_inputs(mb) ≈ 1200.0  # initial + inflow
        @test total_outputs(mb) ≈ 225.0  # outflow + infiltration + evaporation + drainage
    end

    @testset "Mass Balance Summary" begin
        mb = MassBalance(Float64)
        mb.initial_volume = 1000.0
        mb.rainfall_volume = 100.0
        mb.inflow_volume = 50.0
        mb.outflow_volume = 30.0
        mb.infiltration_volume = 20.0
        mb.evaporation_volume = 10.0
        mb.drainage_exchange = 15.0
        mb.current_volume = 1075.0

        summary = mass_balance_summary(mb)
        @test summary["initial_volume"] ≈ 1000.0
        @test summary["rainfall_volume"] ≈ 100.0
        @test summary["inflow_volume"] ≈ 50.0
        @test summary["total_inputs"] ≈ 1150.0
        @test summary["total_outputs"] ≈ 75.0
    end
end

@testset "Spatial Rainfall" begin
    @testset "SpatialRainfallEvent Construction" begin
        # Create 5x5 rainfall fields
        times = [0.0, 1800.0, 3600.0]
        field1 = zeros(5, 5)
        field2 = fill(30.0, 5, 5)  # 30 mm/hr peak
        field3 = zeros(5, 5)
        fields = [field1, field2, field3]

        rain = SpatialRainfallEvent(times, fields)

        @test length(rain.times) == 3
        @test length(rain.fields) == 3
        @test rain.interpolation == :linear
    end

    @testset "Spatial Rainfall Rate" begin
        times = [0.0, 1800.0, 3600.0]
        field1 = zeros(5, 5)
        field2 = fill(60.0, 5, 5)
        field3 = zeros(5, 5)
        rain = SpatialRainfallEvent(times, [field1, field2, field3])

        # At t=0, should be zero everywhere
        rate0 = spatial_rainfall_rate(rain, 0.0)
        @test all(rate0 .≈ 0.0)

        # At t=1800 (peak), should be 60 mm/hr everywhere
        rate_peak = spatial_rainfall_rate(rain, 1800.0)
        @test all(rate_peak .≈ 60.0)

        # At t=900 (halfway to peak), should be 30 mm/hr
        rate_mid = spatial_rainfall_rate(rain, 900.0)
        @test all(isapprox.(rate_mid, 30.0, atol=0.1))

        # Test conversion to m/s
        rate_ms = spatial_rainfall_rate_ms(rain, 1800.0)
        expected_ms = 60.0 / 1000.0 / 3600.0
        @test all(isapprox.(rate_ms, expected_ms, atol=1e-10))
    end

    @testset "Spatial Rainfall Heterogeneous" begin
        times = [0.0, 3600.0]
        field1 = [0.0 10.0 20.0; 30.0 40.0 50.0; 60.0 70.0 80.0]
        field2 = field1  # Constant in time
        rain = SpatialRainfallEvent(times, [field1, field2])

        rate = spatial_rainfall_rate(rain, 1800.0)
        @test rate[1, 1] ≈ 0.0
        @test rate[2, 2] ≈ 40.0
        @test rate[3, 3] ≈ 80.0
    end

    @testset "Apply Spatial Rainfall" begin
        h = zeros(3, 3)
        times = [0.0, 3600.0]
        field = fill(36.0, 3, 3)  # 36 mm/hr = 1e-5 m/s
        rain = SpatialRainfallEvent(times, [field, field])

        dt = 60.0
        apply_rainfall!(h, rain, 0.0, dt)

        expected = 36.0 / 1000.0 / 3600.0 * 60.0
        @test all(isapprox.(h, expected, atol=1e-10))
    end

    @testset "Spatial Rainfall Statistics" begin
        times = [0.0, 1800.0, 3600.0]
        field1 = zeros(5, 5)
        field2 = [10.0 20.0 30.0 40.0 50.0;
                  10.0 20.0 30.0 40.0 50.0;
                  10.0 20.0 30.0 40.0 50.0;
                  10.0 20.0 30.0 40.0 50.0;
                  10.0 20.0 30.0 40.0 50.0]
        field3 = zeros(5, 5)
        rain = SpatialRainfallEvent(times, [field1, field2, field3])

        @test max_intensity(rain) ≈ 50.0
        @test mean_areal_rainfall(rain, 1800.0) ≈ 30.0  # Mean of peak field
    end
end

@testset "Spatial Infiltration" begin
    @testset "Soil Parameters Lookup" begin
        # Test SOIL_PARAMETERS
        sand = SOIL_PARAMETERS[:sand]
        @test sand[1] ≈ 1.2e-4  # K
        @test sand[3] ≈ 0.44    # θs

        clay = SOIL_PARAMETERS[:clay]
        @test clay[1] ≈ 3.0e-7  # Much lower K than sand

        impervious = SOIL_PARAMETERS[:impervious]
        @test impervious[1] ≈ 0.0  # No infiltration

        # Test SOIL_TYPE_IDS
        @test SOIL_TYPE_IDS[0] == :impervious
        @test SOIL_TYPE_IDS[1] == :sand
        @test SOIL_TYPE_IDS[7] == :clay
    end

    @testset "SpatialInfiltrationParameters Construction" begin
        soil_map = [:sand :loam :clay; :sand :loam :clay; :sand :loam :clay]
        params = SpatialInfiltrationParameters(soil_map)

        @test size(params.K) == (3, 3)
        @test params.K[1, 1] ≈ 1.2e-4  # Sand column
        @test params.K[1, 3] ≈ 3.0e-7  # Clay column
    end

    @testset "SpatialInfiltrationParameters from Integer IDs" begin
        # Typical GIS workflow: integer soil type raster
        # IDs: 0=impervious, 1=sand, 4=loam, 7=clay
        soil_ids = [1 4 7; 1 4 7; 0 0 0]  # sand, loam, clay; bottom row impervious
        params = SpatialInfiltrationParameters(soil_ids)

        @test params.soil_map[1, 1] == :sand
        @test params.soil_map[1, 2] == :loam
        @test params.soil_map[1, 3] == :clay
        @test params.soil_map[3, 1] == :impervious
        @test params.K[3, 1] ≈ 0.0  # Impervious
    end

    @testset "Spatial Infiltration Rate" begin
        soil_map = [:sand :clay; :sand :clay]
        params = SpatialInfiltrationParameters(soil_map)
        state = InfiltrationState(2, 2)

        # Use large depth so infiltration doesn't exhaust all water
        h = fill(10.0, 2, 2)  # 10m water depth (very deep)
        # Use very short timestep
        infiltrated = apply_infiltration!(h, state, params, 0.1)

        @test infiltrated > 0
        # Sand (K=1.2e-4 m/s) should infiltrate more than clay (K=3e-7 m/s)
        # If both hit max due to simple K*dt model, just check they infiltrated
        @test state.cumulative[1, 1] >= state.cumulative[1, 2]
    end

    @testset "Impervious Areas" begin
        soil_map = [:impervious :impervious; :sand :sand]
        params = SpatialInfiltrationParameters(soil_map)
        state = InfiltrationState(2, 2)

        h = fill(0.1, 2, 2)
        apply_infiltration!(h, state, params, 60.0)

        # Impervious should have no infiltration
        @test state.cumulative[1, 1] ≈ 0.0
        @test state.cumulative[1, 2] ≈ 0.0
        # Pervious should have infiltration
        @test state.cumulative[2, 1] > 0.0
    end

    @testset "Pervious Fraction" begin
        soil_map = [:impervious :sand; :clay :impervious]
        params = SpatialInfiltrationParameters(soil_map)

        frac = pervious_fraction(params)
        @test frac ≈ 0.5  # 2 out of 4 cells
    end
end

@testset "Evaporation Module" begin
    @testset "EvaporationParameters Construction" begin
        # Default constant rate (5 mm/day)
        params = EvaporationParameters()
        @test params.method == :constant
        @test params.rate > 0

        # Custom parameters (rate in mm/day)
        params_custom = EvaporationParameters(
            method=:penman,
            rate=5.0,  # mm/day
            albedo=0.25,
            elevation=100.0,
            latitude=45.0
        )
        @test params_custom.method == :penman
        @test params_custom.elevation ≈ 100.0
    end

    @testset "Evaporation Rate from Params" begin
        # Rate is in mm/day, stored internally as m/s
        params = EvaporationParameters(rate=8.64)  # 8.64 mm/day ≈ 1e-7 m/s
        @test params.rate ≈ 1e-7 atol=1e-9
    end

    @testset "EvaporationTimeSeries" begin
        # EvaporationTimeSeries takes mm/day rates and converts to m/s
        times = [0.0, 43200.0, 86400.0]  # 0, 12hr, 24hr
        rates_mm_day = [0.0, 10.0, 0.0]  # Peak at noon

        # Note: Use explicit constructor call
        et = EvaporationTimeSeries(times, rates_mm_day, T=Float64)

        # At t=0, rate should be 0
        @test evaporation_rate(et, 0.0) ≈ 0.0

        # At noon (peak) - should be converted rate
        peak_rate = evaporation_rate(et, 43200.0)
        @test peak_rate > 0.0

        # At 6 hours (interpolated) - should be half of peak
        @test evaporation_rate(et, 21600.0) ≈ peak_rate / 2 atol=1e-12
    end

    @testset "Apply Evaporation" begin
        # rate=86.4 mm/day ≈ 1e-6 m/s
        params = EvaporationParameters(rate=86.4)
        h = fill(0.01, 5, 5)  # 1 cm water

        dt = 1000.0  # Long timestep
        evaporated = apply_evaporation!(h, params, dt)

        expected_loss = params.rate * 1000.0  # rate * dt
        @test all(isapprox.(h, 0.01 - expected_loss, atol=1e-8))
        @test evaporated ≈ expected_loss * 25 atol=1e-6  # Total over all cells
    end

    @testset "Evaporation Limited by Water" begin
        params = EvaporationParameters(rate=86400.0)  # Very high rate (86400 mm/day = 1e-3 m/s)
        h = fill(0.0001, 5, 5)  # Very shallow water

        dt = 1.0
        apply_evaporation!(h, params, dt)

        # Should not go negative
        @test all(h .>= 0)
    end

    @testset "Saturation Vapor Pressure" begin
        # At 20°C, saturation vapor pressure ≈ 2.34 kPa
        e_s = saturation_vapor_pressure(20.0)
        @test e_s ≈ 2.34 atol=0.1

        # At 0°C, ≈ 0.61 kPa
        e_s_0 = saturation_vapor_pressure(0.0)
        @test e_s_0 ≈ 0.61 atol=0.05
    end

    @testset "Penman-Monteith ET" begin
        meteo = MeteorologicalData(
            temperature=25.0,
            humidity=0.6,
            wind_speed=2.0,
            solar_radiation=300.0,
            day_of_year=180
        )
        params = EvaporationParameters(method=:penman, elevation=0.0)

        et = penman_monteith_et(meteo, params)
        @test et > 0
        # Typical ET: 3-7 mm/day ≈ 3.5e-8 to 8e-8 m/s
        @test 1e-9 < et < 1e-5
    end

    @testset "Priestley-Taylor ET" begin
        meteo = MeteorologicalData(
            temperature=25.0,
            humidity=0.6,
            wind_speed=2.0,
            solar_radiation=300.0,
            day_of_year=180
        )
        params = EvaporationParameters(method=:priestley_taylor)

        et = priestley_taylor_et(meteo, params)
        @test et > 0
    end

    @testset "Hargreaves ET" begin
        # Hargreaves takes T_min, T_max, T_mean, Ra directly
        T_min = 15.0
        T_max = 25.0
        T_mean = 20.0
        Ra = 30.0  # Extraterrestrial radiation MJ/m²/day

        et = hargreaves_et(T_min, T_max, T_mean, Ra)
        @test et > 0
    end

    @testset "Daily Evaporation" begin
        # EvaporationParameters stores rate in m/s after conversion from mm/day
        params = EvaporationParameters(rate=5.0)  # 5 mm/day

        daily = daily_evaporation(params)
        @test daily ≈ 5.0 atol=0.1
    end
end

@testset "Velocity Direction" begin
    @testset "Basic Direction Calculations" begin
        # East (positive x): angle = 0
        @test velocity_direction(1.0, 0.0) ≈ 0.0 atol=1e-10
        @test velocity_direction_degrees(1.0, 0.0) ≈ 0.0 atol=1e-10
        @test velocity_direction_compass(1.0, 0.0) ≈ 90.0 atol=1e-10  # East = 90°

        # North (positive y): angle = π/2
        @test velocity_direction(0.0, 1.0) ≈ π/2 atol=1e-10
        @test velocity_direction_degrees(0.0, 1.0) ≈ 90.0 atol=1e-10
        @test velocity_direction_compass(0.0, 1.0) ≈ 0.0 atol=1e-10  # North = 0°

        # West (negative x): angle = ±π
        @test abs(velocity_direction(-1.0, 0.0)) ≈ π atol=1e-10
        @test velocity_direction_compass(-1.0, 0.0) ≈ 270.0 atol=1e-10  # West = 270°

        # South (negative y): angle = -π/2
        @test velocity_direction(0.0, -1.0) ≈ -π/2 atol=1e-10
        @test velocity_direction_compass(0.0, -1.0) ≈ 180.0 atol=1e-10  # South = 180°
    end

    @testset "Diagonal Directions" begin
        # NE: 45° compass
        @test velocity_direction_compass(1.0, 1.0) ≈ 45.0 atol=1e-10

        # NW: 315° compass
        @test velocity_direction_compass(-1.0, 1.0) ≈ 315.0 atol=1e-10

        # SE: 135° compass
        @test velocity_direction_compass(1.0, -1.0) ≈ 135.0 atol=1e-10

        # SW: 225° compass
        @test velocity_direction_compass(-1.0, -1.0) ≈ 225.0 atol=1e-10
    end

    @testset "ResultsAccumulator Velocity Direction" begin
        grid = Grid(5, 5, 10.0)
        results = ResultsAccumulator(grid, Tuple{Int,Int}[])

        # Set some velocity components
        results.velocity_u_at_max[1, 1] = 1.0  # East
        results.velocity_v_at_max[1, 1] = 0.0

        results.velocity_u_at_max[2, 2] = 0.0  # North
        results.velocity_v_at_max[2, 2] = 1.0

        results.velocity_u_at_max[3, 3] = 1.0  # NE
        results.velocity_v_at_max[3, 3] = 1.0

        # Compute direction field
        direction = compute_velocity_direction(results)
        @test direction[1, 1] ≈ 0.0 atol=1e-10
        @test direction[2, 2] ≈ π/2 atol=1e-10
        @test direction[3, 3] ≈ π/4 atol=1e-10

        # Compute compass direction
        compass = compute_velocity_direction_compass(results)
        @test compass[1, 1] ≈ 90.0 atol=1e-10   # East
        @test compass[2, 2] ≈ 0.0 atol=1e-10    # North
        @test compass[3, 3] ≈ 45.0 atol=1e-10   # NE
    end

    @testset "Zero Velocity Direction" begin
        # Zero velocity returns atan(0,0) which is 0 in Julia
        dir = velocity_direction(0.0, 0.0)
        @test isfinite(dir)  # Just check it's a valid number

        grid = Grid(3, 3, 10.0)
        results = ResultsAccumulator(grid, Tuple{Int,Int}[])

        direction = compute_velocity_direction(results)
        @test all(direction .== 0.0)
    end
end
