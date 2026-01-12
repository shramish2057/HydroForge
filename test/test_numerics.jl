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
