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
    end

    @testset "Positive Depth Enforcement" begin
        state = SimulationState(10, 10)
        state.h[5, 5] = -0.1  # Invalid negative depth

        enforce_positive_depth!(state, 0.001)

        @test state.h[5, 5] == 0.0
        @test all(state.h .>= 0)
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
