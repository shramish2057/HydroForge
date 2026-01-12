# Tests for HydroForge numerical modules

using Test
using HydroForge

@testset "Timestep" begin
    @testset "CFL Computation" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)
        params = SimulationParameters(cfl=0.7)

        # Dry domain - should return dt_max
        dt = compute_dt(state, grid, params)
        @test dt == params.dt_max

        # Add some water
        state.h .= 1.0
        dt = compute_dt(state, grid, params)

        # dt should be bounded by CFL condition
        # dt ≤ CFL * dx / sqrt(g*h)
        expected_max = params.cfl * grid.dx / sqrt(params.g * 1.0)
        @test dt <= expected_max
    end

    @testset "Dry Cell Handling" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)
        params = SimulationParameters()

        # Single wet cell
        state.h[5, 5] = 1.0

        dt = compute_dt(state, grid, params)
        @test dt > 0
        @test dt <= params.dt_max
    end

    @testset "TimestepController Construction" begin
        # Default constructor
        controller = TimestepController()
        @test controller.history_size == 10
        @test controller.smoothing_factor == 0.7
        @test controller.min_dt_warning == 0.001
        @test controller.warning_issued == false
        @test isempty(controller.dt_history)

        # Custom parameters
        controller2 = TimestepController(Float32;
            history_size=5,
            smoothing_factor=0.8,
            min_dt_warning=0.01)
        @test controller2.history_size == 5
        @test controller2.smoothing_factor ≈ 0.8f0
        @test controller2.min_dt_warning ≈ 0.01f0
    end

    @testset "TimestepController Smoothing" begin
        controller = TimestepController()

        # First timestep - no smoothing applied
        dt1 = compute_dt_smooth!(controller, 1.0)
        @test dt1 == 1.0
        @test length(controller.dt_history) == 1

        # Second timestep - smoothing applies
        dt2 = compute_dt_smooth!(controller, 0.8)
        @test length(controller.dt_history) == 2
        # Result should be between 0.8 and 1.0 due to smoothing
        @test 0.8 <= dt2 <= 1.0

        # Large reduction - CFL constraint takes priority
        dt3 = compute_dt_smooth!(controller, 0.1)
        @test length(controller.dt_history) == 3
        # When dt_raw (CFL limit) is very small, it takes priority over smoothing
        # This ensures numerical stability
        @test dt3 == 0.1

        # Test smoothing behavior with gradual reduction
        controller2 = TimestepController()
        dt_a = compute_dt_smooth!(controller2, 1.0)
        dt_b = compute_dt_smooth!(controller2, 0.6)  # 40% reduction
        # With 50% reduction limit and smoothing, dt_b should be limited
        @test dt_b >= 0.5  # At least 50% of previous
        @test dt_b <= 1.0  # But not increasing
    end

    @testset "TimestepController History Limiting" begin
        controller = TimestepController(Float64; history_size=3)

        # Add more than history_size entries
        for i in 1:5
            compute_dt_smooth!(controller, Float64(i))
        end

        # History should be limited to size
        @test length(controller.dt_history) == 3
        # Should contain the most recent values
        @test controller.dt_history[end] == 5.0
    end

    @testset "TimestepController Reset" begin
        controller = TimestepController()

        # Add some history
        compute_dt_smooth!(controller, 1.0)
        compute_dt_smooth!(controller, 0.5)
        controller.warning_issued = true

        # Reset
        reset!(controller)

        @test isempty(controller.dt_history)
        @test controller.warning_issued == false
    end

    @testset "compute_dt_array Low-level" begin
        h = ones(10, 10)
        qx = zeros(10, 10)
        qy = zeros(10, 10)
        dx = 10.0
        dy = 10.0
        g = 9.81
        cfl = 0.7
        h_min = 0.001

        dt = compute_dt_array(h, qx, qy, dx, dy, g, cfl, h_min)
        @test dt > 0
        @test dt < Inf

        # Expected: CFL * min(dx,dy) / sqrt(g*h)
        expected = cfl * min(dx, dy) / sqrt(g * 1.0)
        @test dt ≈ expected

        # Dry domain returns Inf
        h_dry = zeros(10, 10)
        dt_dry = compute_dt_array(h_dry, qx, qy, dx, dy, g, cfl, h_min)
        @test dt_dry == Inf
    end

    @testset "check_cfl" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)
        params = SimulationParameters(cfl=0.7)
        state.h .= 1.0

        # Compute stable dt
        stable_dt = compute_dt(state, grid, params)

        # Smaller dt should satisfy CFL
        @test check_cfl(state, grid, params, stable_dt * 0.5) == true

        # Larger dt should violate CFL
        @test check_cfl(state, grid, params, stable_dt * 2.0) == false

        # Exact dt should satisfy CFL
        @test check_cfl(state, grid, params, stable_dt) == true
    end
end

@testset "Flux Computation" begin
    @testset "Flat Surface" begin
        # Flat water surface should have zero flux
        grid = Grid(10, 10, 10.0)
        h = ones(10, 10)
        z = zeros(10, 10)  # Flat bottom
        n = fill(0.03, 10, 10)
        qx = zeros(10, 10)
        qx_new = zeros(10, 10)

        params = SimulationParameters()
        compute_flux_x!(qx_new, qx, h, z, n, grid, params, 0.1)

        # No gradient = no flux
        @test all(abs.(qx_new) .< 1e-10)
    end

    @testset "Sloped Surface" begin
        # Sloped surface should generate flux
        grid = Grid(10, 10, 10.0)
        h = ones(10, 10)

        # Create a slope in bed elevation
        z = zeros(10, 10)
        for i in 1:10, j in 1:10
            z[i, j] = (i - 1) * 0.1  # 10% slope
        end

        n = fill(0.03, 10, 10)
        qx = zeros(10, 10)
        qx_new = zeros(10, 10)

        params = SimulationParameters()
        compute_flux_x!(qx_new, qx, h, z, n, grid, params, 0.1)

        # Should have non-zero flux due to gradient
        # (exact value depends on friction, but should be non-zero)
        @test any(abs.(qx_new[2:9, :]) .> 0)
    end

    @testset "Y-Direction Flux" begin
        # Test flux in y-direction with slope
        grid = Grid(10, 10, 10.0)
        h = ones(10, 10)

        # Create a slope in y-direction
        z = zeros(10, 10)
        for i in 1:10, j in 1:10
            z[i, j] = (j - 1) * 0.1  # 10% slope in y
        end

        n = fill(0.03, 10, 10)
        qy = zeros(10, 10)
        qy_new = zeros(10, 10)

        params = SimulationParameters()
        compute_flux_y!(qy_new, qy, h, z, n, grid, params, 0.1)

        # Should have non-zero flux in y-direction
        @test any(abs.(qy_new[:, 2:8]) .> 0)
    end

    @testset "Face Depth Interpolation" begin
        # Test face depth at step
        h = [1.0 1.0; 0.5 0.5]
        z = [0.0 0.0; 0.5 0.5]  # Step in bed

        # Face between (1,1) and (2,1)
        h_face = face_depth_x(h, z, 1, 1)
        # η_L = 1.0, η_R = 1.0, z_face = max(0.0, 0.5) = 0.5
        # h_face = max(1.0, 1.0) - 0.5 = 0.5
        @test h_face ≈ 0.5

        # Face with different water levels
        h2 = [2.0 2.0; 1.0 1.0]
        z2 = [0.0 0.0; 0.0 0.0]

        h_face_y = face_depth_y(h2, z2, 1, 1)
        # η_B = 2.0, η_T = 1.0, z_face = 0.0
        # h_face = max(2.0, 1.0) - 0.0 = 2.0
        @test h_face_y ≈ 2.0
    end

    @testset "Velocity Computation" begin
        h_min = 0.001

        # Normal velocity
        @test compute_velocity(1.0, 2.0, h_min) ≈ 0.5

        # Dry cell returns zero
        @test compute_velocity(1.0, 0.0005, h_min) == 0.0

        # Zero discharge
        @test compute_velocity(0.0, 1.0, h_min) == 0.0
    end
end

@testset "Wetting and Drying" begin
    @testset "is_wet" begin
        h_min = 0.001
        @test is_wet(0.01, h_min) == true
        @test is_wet(0.001, h_min) == false
        @test is_wet(0.0005, h_min) == false
    end

    @testset "wet_dry_factor" begin
        h_min = 0.001
        h_trans = 0.002

        # Dry cell
        @test wet_dry_factor(0.0005, h_min, h_trans) == 0.0

        # Fully wet
        @test wet_dry_factor(0.01, h_min, h_trans) == 1.0

        # Transition zone (smooth)
        factor = wet_dry_factor(0.002, h_min, h_trans)
        @test 0 < factor < 1
        @test factor ≈ 0.5  # Midpoint of cubic
    end

    @testset "limit_flux_wetdry!" begin
        # h[i,j] layout: row i, column j
        # h[1,:] = first row (i=1), h[2,:] = second row (i=2)
        h = [0.01 0.01; 0.0005 0.0005]  # Row 1 wet, Row 2 dry
        qx = ones(2, 2)
        qy = ones(2, 2)
        h_min = 0.001

        limit_flux_wetdry!(qx, qy, h, h_min)

        # qx[i,j] is between cells (i,j) and (i+1,j)
        # qx[1,1] is at wet/dry interface (h[1,1] wet, h[2,1] dry) - should be reduced
        @test qx[1, 1] < 1.0

        # qy[i,j] is between cells (i,j) and (i,j+1)
        # qy[1,1] is between h[1,1] (wet) and h[1,2] (wet) - should be unchanged
        @test qy[1, 1] ≈ 1.0
    end
end

@testset "Boundary Conditions" begin
    @testset "Closed Boundaries" begin
        state = SimulationState(10, 10)
        state.qx .= 1.0
        state.qy .= 1.0

        apply_closed_boundaries!(state)

        # Boundaries should be zero
        @test all(state.qx[1, :] .== 0)
        @test all(state.qx[10, :] .== 0)
        @test all(state.qy[:, 1] .== 0)
        @test all(state.qy[:, 10] .== 0)

        # Interior should be unchanged
        @test all(state.qx[2:9, 2:9] .== 1.0)
    end

    @testset "Open Boundaries" begin
        state = SimulationState(10, 10)
        state.qx .= 0.0
        state.qy .= 0.0

        # Set outflow at boundaries (positive = flow in positive direction)
        # Right boundary: positive qx = outflow (flowing right, out of domain)
        # Top boundary: positive qy = outflow (flowing up, out of domain)
        state.qx[9, :] .= 0.5  # Outflow at right
        state.qy[:, 9] .= 0.5  # Outflow at top

        apply_open_boundaries!(state)

        # Outflow should be extrapolated
        @test all(state.qx[10, :] .== 0.5)
        @test all(state.qy[:, 10] .== 0.5)

        # Inflow at boundaries should be blocked
        state2 = SimulationState(10, 10)
        # Left boundary: positive qx would mean inflow (flowing right, into domain)
        state2.qx[2, :] .= 0.5

        apply_open_boundaries!(state2)

        # Left boundary: extrapolated from interior but inflow blocked
        # Since qx[2,:] is positive, qx[1,:] would be positive = inflow from left
        @test all(state2.qx[1, :] .== 0)
    end

    @testset "apply_boundaries! dispatch" begin
        state = SimulationState(10, 10)
        state.qx .= 1.0
        state.qy .= 1.0

        apply_boundaries!(state, CLOSED)
        @test all(state.qx[1, :] .== 0)

        state.qx .= 1.0
        state.qy .= 1.0
        apply_boundaries!(state)  # Default is CLOSED
        @test all(state.qx[1, :] .== 0)
    end

    @testset "Positive Depth Enforcement" begin
        state = SimulationState(10, 10)
        state.h[5, 5] = -0.1  # Invalid negative depth

        enforce_positive_depth!(state, 0.001)

        @test state.h[5, 5] == 0.0
        @test all(state.h .>= 0)
    end

    @testset "Dry Cell Discharge Zeroing" begin
        state = SimulationState(10, 10)
        state.h[5, 5] = 0.0005  # Below h_min
        state.qx[5, 5] = 1.0
        state.qy[5, 5] = 1.0

        enforce_positive_depth!(state, 0.001)

        # Discharge should be zeroed for dry cells
        @test state.qx[5, 5] == 0.0
        @test state.qy[5, 5] == 0.0
    end

    @testset "Mass Conservation with Closed Boundaries" begin
        # Start with water, apply closed boundaries, check volume unchanged
        state = SimulationState(10, 10)
        state.h .= 1.0

        initial_volume = sum(state.h)

        apply_closed_boundaries!(state)
        enforce_positive_depth!(state, 0.001)

        final_volume = sum(state.h)

        @test initial_volume ≈ final_volume
    end
end

@testset "Water Surface Elevation" begin
    h = [1.0 2.0; 3.0 4.0]
    z = [0.5 0.5; 0.5 0.5]
    η = similar(h)

    water_surface_elevation!(η, h, z)

    @test η[1, 1] == 1.5
    @test η[1, 2] == 2.5
    @test η[2, 1] == 3.5
    @test η[2, 2] == 4.5
end

@testset "Surface Gradient" begin
    @testset "Uniform Surface" begin
        grid = Grid(5, 5, 10.0)
        η = fill(1.0, 5, 5)  # Flat surface

        dη_dx, dη_dy = surface_gradient(η, grid)

        # Flat surface should have zero gradients
        @test all(abs.(dη_dx) .< 1e-10)
        @test all(abs.(dη_dy) .< 1e-10)
    end

    @testset "Sloped Surface" begin
        grid = Grid(5, 5, 10.0)
        η = zeros(5, 5)
        # Create linear slope in x: η = 0.1 * x
        for i in 1:5, j in 1:5
            η[i, j] = 0.1 * (i - 1) * grid.dx
        end

        dη_dx, dη_dy = surface_gradient(η, grid)

        # Interior should have gradient 0.1
        @test dη_dx[3, 3] ≈ 0.1 atol=1e-10
        # Y-gradient should be zero
        @test dη_dy[3, 3] ≈ 0.0 atol=1e-10
    end

    @testset "In-place Gradient" begin
        grid = Grid(5, 5, 10.0)
        η = ones(5, 5)
        dη_dx = zeros(5, 5)
        dη_dy = zeros(5, 5)

        surface_gradient!(dη_dx, dη_dy, η, grid)

        @test all(abs.(dη_dx) .< 1e-10)
        @test all(abs.(dη_dy) .< 1e-10)
    end
end

@testset "Fixed Depth Boundaries" begin
    @testset "All Sides" begin
        state = SimulationState(10, 10)
        state.h .= 0.5

        apply_fixed_depth_boundaries!(state, 1.0)

        # All boundaries should be 1.0
        @test all(state.h[1, :] .== 1.0)
        @test all(state.h[10, :] .== 1.0)
        @test all(state.h[:, 1] .== 1.0)
        @test all(state.h[:, 10] .== 1.0)

        # Interior unchanged
        @test state.h[5, 5] == 0.5
    end

    @testset "Selective Sides" begin
        state = SimulationState(10, 10)
        state.h .= 0.5

        # Only left and right
        apply_fixed_depth_boundaries!(state, 1.0, (true, true, false, false))

        @test all(state.h[1, :] .== 1.0)
        @test all(state.h[10, :] .== 1.0)
        # Bottom/top interior unchanged (corners are on left/right so they're set)
        @test all(state.h[2:9, 1] .== 0.5)  # Bottom interior unchanged
        @test all(state.h[2:9, 10] .== 0.5)  # Top interior unchanged
    end

    @testset "BoundaryCondition Struct" begin
        bc = BoundaryCondition(FIXED_DEPTH, fixed_depth=0.5)
        @test bc.type == FIXED_DEPTH
        @test bc.fixed_depth == 0.5
        @test bc.sides == (true, true, true, true)

        bc2 = BoundaryCondition(CLOSED, sides=(true, false, true, false))
        @test bc2.type == CLOSED
        @test bc2.sides == (true, false, true, false)
    end

    @testset "apply_boundaries! with FIXED_DEPTH" begin
        state = SimulationState(10, 10)
        state.h .= 0.5

        apply_boundaries!(state, FIXED_DEPTH)

        # Default fixed depth is 0
        @test all(state.h[1, :] .== 0.0)
    end
end

@testset "Time-Varying Boundary Conditions" begin
    @testset "BoundaryTimeSeries" begin
        times = [0.0, 3600.0, 7200.0]
        values = [0.0, 1.0, 0.5]
        bts = BoundaryTimeSeries(times, values)

        # Test interpolation
        @test interpolate_boundary(bts, 0.0) ≈ 0.0
        @test interpolate_boundary(bts, 3600.0) ≈ 1.0
        @test interpolate_boundary(bts, 1800.0) ≈ 0.5  # Midpoint
        @test interpolate_boundary(bts, 5400.0) ≈ 0.75  # Between 1.0 and 0.5

        # Before/after range
        @test interpolate_boundary(bts, -100.0) ≈ 0.0  # Use first value
        @test interpolate_boundary(bts, 10000.0) ≈ 0.5  # Use last value
    end

    @testset "TidalBoundary" begin
        # Simple semidiurnal tide: 12.42 hour period
        period = 12.42 * 3600.0  # seconds
        amplitude = 1.5  # meters
        mean_level = 2.0  # meters

        tide = TidalBoundary(mean_level, amplitude=amplitude, period=period)

        # At t=0 (phase=0), level = mean + amplitude
        @test tidal_level(tide, 0.0) ≈ mean_level + amplitude atol=1e-10

        # At t=period/2, level = mean - amplitude
        @test tidal_level(tide, period/2) ≈ mean_level - amplitude atol=1e-10

        # At t=period/4, level = mean (halfway)
        @test tidal_level(tide, period/4) ≈ mean_level atol=0.1

        # Full period returns to start
        @test tidal_level(tide, period) ≈ tidal_level(tide, 0.0) atol=1e-10
    end

    @testset "TidalBoundary with Phase" begin
        period = 12.0 * 3600.0
        amplitude = 1.0
        mean_level = 0.0
        phase_shift = π/2  # Start at mean, rising

        tide = TidalBoundary(mean_level, amplitude=amplitude, period=period, phase=phase_shift)

        # At t=0 with π/2 phase shift: cos(π/2) = 0
        @test tidal_level(tide, 0.0) ≈ 0.0 atol=1e-10

        # At t=period/4: cos(π) = -1
        @test tidal_level(tide, period/4) ≈ -amplitude atol=1e-10
    end

    @testset "InflowHydrograph" begin
        # Triangular hydrograph
        times = [0.0, 1800.0, 3600.0]
        discharges = [0.0, 10.0, 0.0]  # m³/s
        width = 10.0  # meters

        hydro = InflowHydrograph(times, discharges, width)

        # Test discharge interpolation
        @test inflow_discharge(hydro, 0.0) ≈ 0.0
        @test inflow_discharge(hydro, 1800.0) ≈ 10.0
        @test inflow_discharge(hydro, 900.0) ≈ 5.0  # Rising limb
        @test inflow_discharge(hydro, 2700.0) ≈ 5.0  # Falling limb

        # Test flux (Q/width)
        @test inflow_flux(hydro, 1800.0) ≈ 1.0  # 10 m³/s / 10 m = 1 m²/s
        @test inflow_flux(hydro, 900.0) ≈ 0.5
    end

    @testset "RatingCurve" begin
        # Power law rating curve: Q = a * (h - c)^b
        # Use c=0 so stages go from 0 to max_depth
        a = 5.0
        b = 1.5
        c = 0.0  # reference elevation for power law

        rc = RatingCurve(a, b, c)

        # At stage=0, Q=0
        @test rating_discharge(rc, 0.0) ≈ 0.0

        # At stage=1.0: Q = 5 * 1^1.5 = 5
        @test rating_discharge(rc, 1.0) ≈ a * 1.0^b atol=0.1

        # At stage=2.0: Q = 5 * 2^1.5 ≈ 14.14
        @test rating_discharge(rc, 2.0) ≈ a * 2.0^b atol=0.2

        # Negative stage (below datum) returns first Q value (0)
        @test rating_discharge(rc, -0.5) ≈ 0.0
    end

    @testset "BoundaryType Enum" begin
        @test CLOSED isa BoundaryType
        @test OPEN isa BoundaryType
        @test FIXED_DEPTH isa BoundaryType
        @test INFLOW isa BoundaryType
        @test TIDAL isa BoundaryType
        @test RATING_CURVE isa BoundaryType
    end

    @testset "get_boundary_value" begin
        # Test BoundaryTimeSeries via BoundaryCondition
        times = [0.0, 100.0]
        values = [1.0, 2.0]
        bts = BoundaryTimeSeries(times, values)
        bc_ts = BoundaryCondition(FIXED_DEPTH, time_series=bts)
        @test get_boundary_value(bc_ts, 50.0) ≈ 1.5

        # Test TidalBoundary via BoundaryCondition
        tide = TidalBoundary(5.0, amplitude=1.0, period=100.0)
        bc_tidal = BoundaryCondition(TIDAL, tidal=tide)
        @test get_boundary_value(bc_tidal, 0.0) ≈ 6.0  # mean + amplitude

        # Test InflowHydrograph via BoundaryCondition (returns discharge Q)
        hydro = InflowHydrograph([0.0, 100.0], [0.0, 100.0], width=10.0)
        bc_inflow = BoundaryCondition(INFLOW, hydrograph=hydro)
        @test get_boundary_value(bc_inflow, 50.0) ≈ 50.0  # Q=50 at t=50
    end

    @testset "apply_inflow_boundaries!" begin
        state = SimulationState(10, 10)
        state.h .= 0.5
        state.qx .= 0.0

        # Create an inflow hydrograph
        hydro = InflowHydrograph([0.0, 100.0], [20.0, 20.0], width=10.0)  # Constant 20 m³/s

        # Apply inflow on left boundary at t=0
        apply_inflow_boundaries!(state, hydro, 0.0, (true, false, false, false))

        # Left boundary should have inflow flux = Q/width = 20/10 = 2 m²/s
        @test all(state.qx[1, :] .≈ 2.0)
        # Other boundaries unchanged
        @test all(state.qx[10, :] .== 0.0)
    end

    @testset "apply_rating_curve_boundaries!" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)

        # Set water depth at boundary
        state.h .= 1.0  # 1m depth
        state.qx .= 0.0

        # Rating curve: Q = 5 * h^1.5
        rc = RatingCurve(5.0, 1.5, 0.0)

        # Apply on right boundary (outflow) - note: no topo parameter
        apply_rating_curve_boundaries!(state, rc, (false, true, false, false), grid)

        # Right boundary should have some outflow (non-zero)
        @test any(state.qx[10, 2:9] .> 0)
    end

    @testset "Open Boundaries with Outflow Tracking" begin
        grid = Grid(10, 10, 10.0)
        state = SimulationState(grid)
        state.h .= 1.0
        state.qx .= 0.0
        state.qy .= 0.0

        # Set outward flow at boundaries
        state.qx[9, :] .= 0.5  # Flow to right
        state.qy[:, 9] .= 0.5  # Flow to top

        # Apply open boundaries with grid for outflow tracking
        apply_open_boundaries!(state, (true, true, true, true), grid)

        # Outflow should be extrapolated to boundaries
        @test all(state.qx[10, :] .≈ 0.5)
        @test all(state.qy[:, 10] .≈ 0.5)
    end
end
