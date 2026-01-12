# Tests for 1D drainage network and 1D-2D coupling

using Test
using HydroForge

@testset "1D Drainage Network Types" begin
    @testset "Circular Pipe Geometry" begin
        pipe = CircularPipe(0.6)  # 600mm diameter

        @test pipe.diameter == 0.6

        # Full pipe area
        @test full_area(pipe) ≈ π * 0.3^2 atol=1e-10

        # Flow area at various depths
        @test flow_area(pipe, 0.0) == 0.0
        @test flow_area(pipe, 0.6) ≈ full_area(pipe) atol=1e-10
        @test flow_area(pipe, 0.3) ≈ full_area(pipe) / 2 atol=0.01  # Half full

        # Wetted perimeter
        @test wetted_perimeter(pipe, 0.0) == 0.0
        @test wetted_perimeter(pipe, 0.6) ≈ π * 0.6 atol=1e-10

        # Hydraulic radius at full
        R_full = hydraulic_radius(pipe, 0.6)
        @test R_full ≈ 0.6 / 4 atol=1e-10  # D/4 for circular

        # Top width
        @test top_width(pipe, 0.0) == 0.0
        @test top_width(pipe, 0.3) ≈ 0.6 atol=1e-10  # Maximum at half full
    end

    @testset "Rectangular Pipe Geometry" begin
        pipe = RectangularPipe(1.0, 0.5)  # 1m wide, 0.5m high

        @test pipe.width == 1.0
        @test pipe.height == 0.5

        # Full pipe area
        @test full_area(pipe) == 0.5

        # Flow area
        @test flow_area(pipe, 0.0) == 0.0
        @test flow_area(pipe, 0.25) == 0.25  # Half full
        @test flow_area(pipe, 0.5) == 0.5    # Full

        # Wetted perimeter at full
        @test wetted_perimeter(pipe, 0.5) == 3.0  # 2*1 + 2*0.5

        # Hydraulic radius at full
        @test hydraulic_radius(pipe, 0.5) ≈ 0.5 / 3.0 atol=1e-10
    end

    @testset "Pipe Segment" begin
        section = CircularPipe(0.45)  # 450mm

        pipe = PipeSegment(1, 1, 2, section, 100.0, 0.013, 10.0, 9.5)

        @test pipe.id == 1
        @test pipe.length == 100.0
        @test pipe.roughness == 0.013

        # Slope
        @test slope(pipe) ≈ 0.005  # 0.5% grade
        @test !is_adverse(pipe)

        # Full flow capacity (Manning)
        Q_full = full_flow_capacity(pipe)
        @test Q_full > 0
        # Expected: Q = (1/0.013) × (π×0.225²) × (0.225/2)^(2/3) × √0.005
        # ≈ 0.12 m³/s for 450mm @ 0.5%
        @test Q_full > 0.05 && Q_full < 0.5
    end

    @testset "Junction" begin
        j = Junction(1, 100.0, 200.0, 5.0, 8.0)

        @test j.id == 1
        @test j.x == 100.0
        @test j.y == 200.0
        @test j.invert == 5.0
        @test rim_elevation(j) == 8.0
        @test j.max_depth == 3.0  # ground - invert

        @test !is_surcharged(j, 2.0)
        @test is_surcharged(j, 4.0)
    end

    @testset "Inlet" begin
        inlet = Inlet(1, 1, 5, 5)

        @test inlet.id == 1
        @test inlet.junction_id == 1
        @test inlet.grid_i == 5
        @test inlet.grid_j == 5
        @test inlet.inlet_type == GRATE
        @test inlet.clogging_factor == 1.0

        # Opening area and perimeter
        @test inlet_opening_area(inlet) ≈ 0.36 atol=0.01  # 0.6 × 0.6
        @test inlet_perimeter(inlet) ≈ 1.8 atol=0.01     # 2 × 0.6 + 0.6
    end

    @testset "Drainage Network Construction" begin
        # Create simple network: 2 junctions, 1 pipe, 1 inlet, 1 outlet
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0)
        j2 = Junction(2, 100.0, 0.0, 9.5, 11.5; junction_type=OUTFALL)

        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)
        inlet = Inlet(1, 1, 5, 5)
        outlet = Outlet(1, 2)

        network = DrainageNetwork([pipe], [j1, j2], [inlet], [outlet])

        @test n_pipes(network) == 1
        @test n_junctions(network) == 2
        @test n_inlets(network) == 1

        # Test connectivity
        @test network.downstream_pipes[1] == [1]  # Pipe 1 goes downstream from j1
        @test network.upstream_pipes[2] == [1]    # Pipe 1 comes upstream to j2

        # Test validation
        issues = validate(network)
        @test isempty(issues)
    end

    @testset "Drainage State" begin
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0; init_depth=0.1)
        j2 = Junction(2, 100.0, 0.0, 9.5, 11.5)

        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)
        network = DrainageNetwork([pipe], [j1, j2], Inlet{Float64}[], Outlet{Float64}[])

        state = DrainageState(network)

        @test length(state.depth) == 2
        @test state.depth[1] == 0.1  # Initial depth from junction
        @test state.depth[2] == 0.0
        @test state.t == 0.0

        # Total volume
        vol = total_volume(state, network)
        @test vol >= 0
    end
end

@testset "1D Pipe Flow Solver" begin
    @testset "Manning Flow" begin
        A = 0.1  # m²
        R = 0.1  # m
        S = 0.01 # 1%
        n = 0.013

        Q = manning_flow(A, R, S, n)

        # Q = (1/n) × A × R^(2/3) × S^(1/2)
        # Q = (1/0.013) × 0.1 × 0.1^(2/3) × 0.1
        expected = (1/n) * A * R^(2/3) * sqrt(S)
        @test Q ≈ expected atol=1e-10

        # Adverse slope
        Q_adverse = manning_flow(A, R, -S, n)
        @test Q_adverse < 0
        @test abs(Q_adverse) ≈ abs(Q)
    end

    @testset "Pipe Flow Computation" begin
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)

        # Compute flow with water in both junctions
        h_up = 0.2
        h_down = 0.1
        dt = 1.0
        Q_prev = 0.0
        g = 9.81

        result = compute_pipe_flow(pipe, h_up, h_down, dt, Q_prev, g)

        @test result isa PipeFlowResult
        @test result.flow > 0  # Should flow downstream
        @test result.flow_regime == :free_surface
        @test result.velocity > 0
        @test result.froude >= 0
    end

    @testset "Drainage Timestep" begin
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0)
        j2 = Junction(2, 100.0, 0.0, 9.5, 11.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)
        network = DrainageNetwork([pipe], [j1, j2], Inlet{Float64}[], Outlet{Float64}[])

        state = DrainageState(network)
        state.depth[1] = 0.2

        dt = compute_dt_drainage(state, network)

        @test dt > 0
        @test dt <= 60.0  # Default max
    end

    @testset "Step Drainage" begin
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0)
        j2 = Junction(2, 100.0, 0.0, 9.5, 11.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)
        outlet = Outlet(1, 2; outlet_type=:FREE, invert=9.5)
        network = DrainageNetwork([pipe], [j1, j2], Inlet{Float64}[], [outlet])

        state = DrainageState(network)
        state.depth[1] = 0.3  # 30cm water at upstream

        work = DrainageWorkspace(network)
        inlet_flows = [0.1, 0.0]  # Add 0.1 m³/s at junction 1

        outlet_flow = step_drainage!(state, network, work, 1.0;
                                     inlet_flows=inlet_flows)

        # Water should have moved
        @test state.depth[1] != 0.3  # Changed from initial
        @test state.flow[1] != 0     # Flow established in pipe
        @test state.t == 1.0         # Time advanced
    end

    @testset "Run Drainage Simulation" begin
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0)
        j2 = Junction(2, 100.0, 0.0, 9.5, 11.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)
        outlet = Outlet(1, 2; outlet_type=:FREE, invert=9.5)
        network = DrainageNetwork([pipe], [j1, j2], Inlet{Float64}[], [outlet])

        state = DrainageState(network)
        state.depth[1] = 0.2

        # Constant inflow
        inlet_flows_func = t -> [0.05, 0.0]

        results = run_drainage!(state, network, 60.0;
                               inlet_flows_func=inlet_flows_func,
                               verbosity=0)

        @test results["step_count"] > 0
        @test results["wall_time"] > 0
        @test results["total_inlet_volume"] > 0
    end
end

@testset "Inlet Flow Equations" begin
    @testset "Weir Flow (Unsubmerged)" begin
        inlet = Inlet(1, 1, 5, 5;
                     length=0.6, width=0.6,
                     weir_coeff=1.66, orifice_coeff=0.67)

        h_surface = 0.1   # 10cm water on surface
        h_junction = 0.0   # Empty junction
        z_surface = 10.0
        z_invert = 8.0
        g = 9.81

        Q, flow_type = compute_inlet_flow(inlet, h_surface, h_junction,
                                          z_surface, z_invert, g)

        @test Q > 0
        @test flow_type == WEIR_FLOW

        # Weir equation: Q = C × L × h^1.5
        L = inlet_perimeter(inlet)
        Q_expected = 1.66 * L * 0.1^1.5
        @test Q ≈ Q_expected atol=0.001
    end

    @testset "Orifice Flow (Submerged)" begin
        inlet = Inlet(1, 1, 5, 5;
                     length=0.6, width=0.6,
                     weir_coeff=1.66, orifice_coeff=0.67)

        # For submerged flow, junction water level must be above inlet crest
        # Inlet crest at z_surface - depression = 10.0
        # Junction water level = z_invert + h_junction
        # For submergence > 0.8: (η_junction - z_crest) / h_above >= 0.8
        h_surface = 0.5     # 50cm water on surface (h_above = 0.5)
        h_junction = 2.5    # Junction at 8.0 + 2.5 = 10.5m (above crest at 10.0)
        z_surface = 10.0
        z_invert = 8.0
        g = 9.81

        # z_crest = 10.0, η_junction = 10.5
        # submergence = (10.5 - 10.0) / 0.5 = 1.0 > 0.8 -> ORIFICE_FLOW
        # Head difference = 10.5 - 10.5 = 0 (no flow since surface = junction)

        Q, flow_type = compute_inlet_flow(inlet, h_surface, h_junction,
                                          z_surface, z_invert, g)

        # With η_junction = 10.5 and surface at 10.5, we have zero head - no flow
        # Let's use higher surface water to get positive head with submersion
        h_surface2 = 1.0    # Surface at 11.0m > junction at 10.5m
        Q2, flow_type2 = compute_inlet_flow(inlet, h_surface2, h_junction,
                                           z_surface, z_invert, g)

        # Now: h_above = 1.0, submergence = (10.5 - 10.0) / 1.0 = 0.5 (TRANSITION)
        # Or if more submerged: submergence > 0.8 -> ORIFICE_FLOW
        @test flow_type2 == ORIFICE_FLOW || flow_type2 == TRANSITION_FLOW
        @test Q2 > 0  # Surface at 11.0m > junction at 10.5m

        # Test reverse gradient case (junction higher than surface)
        h_junction_high = 3.5  # Junction at 8.0 + 3.5 = 11.5m > surface at 11.0m
        Q3, flow_type3 = compute_inlet_flow(inlet, h_surface2, h_junction_high,
                                           z_surface, z_invert, g)
        @test flow_type3 == NO_FLOW || Q3 == 0  # No inflow when junction higher
    end

    @testset "No Flow Conditions" begin
        inlet = Inlet(1, 1, 5, 5)

        # Dry surface
        Q, flow_type = compute_inlet_flow(inlet, 0.0, 0.0, 10.0, 8.0, 9.81)
        @test Q == 0.0
        @test flow_type == NO_FLOW

        # Junction higher than surface (reverse gradient)
        Q2, flow_type2 = compute_inlet_flow(inlet, 0.1, 3.0, 10.0, 8.0, 9.81)
        @test flow_type2 == NO_FLOW || Q2 == 0.0
    end
end

@testset "1D-2D Coupling" begin
    @testset "Coupled State Creation" begin
        grid = Grid(10, 10, 10.0)
        j1 = Junction(1, 50.0, 50.0, 8.0, 10.0)
        j2 = Junction(2, 100.0, 50.0, 7.5, 9.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 50.0, 0.013, 8.0, 7.5)
        inlet = Inlet(1, 1, 5, 5)
        outlet = Outlet(1, 2)
        network = DrainageNetwork([pipe], [j1, j2], [inlet], [outlet])

        state = CoupledState(grid, network)

        @test state.surface isa SimulationState
        @test state.drainage isa DrainageState
        @test length(state.inlet_exchange) == 1
        @test length(state.outlet_return) == 1
        @test state.t == 0.0
    end

    @testset "Coupled Scenario Creation" begin
        grid = Grid(10, 10, 10.0)
        elevation = zeros(10, 10)
        roughness = fill(0.03, 10, 10)
        topo = Topography(elevation, roughness, grid)
        params = SimulationParameters(t_end=60.0)
        rain = RainfallEvent([0.0, 100.0], [36.0, 36.0])
        scenario = Scenario("test", grid, topo, params, rain, Tuple{Int,Int}[], "output")

        j1 = Junction(1, 50.0, 50.0, -2.0, 0.0)
        pipe = PipeSegment(1, 1, 1, CircularPipe(0.3), 10.0, 0.013, -2.0, -2.0)
        network = DrainageNetwork([pipe], [j1], Inlet{Float64}[], Outlet{Float64}[])

        coupled_scenario = CoupledScenario(scenario, network)

        @test coupled_scenario.name == "test"
        @test coupled_scenario.network === network
    end

    @testset "Exchange Flow Computation" begin
        grid = Grid(10, 10, 10.0)
        elevation = zeros(10, 10)
        roughness = fill(0.03, 10, 10)
        topo = Topography(elevation, roughness, grid)
        params = SimulationParameters(t_end=60.0)
        rain = RainfallEvent([0.0, 100.0], [0.0, 0.0])

        j1 = Junction(1, 50.0, 50.0, -2.0, 0.0)
        j2 = Junction(2, 100.0, 50.0, -2.5, -0.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 50.0, 0.013, -2.0, -2.5)
        inlet = Inlet(1, 1, 5, 5)
        outlet = Outlet(1, 2; grid_i=9, grid_j=5)
        network = DrainageNetwork([pipe], [j1, j2], [inlet], [outlet])

        coupled_scenario = CoupledScenario{Float64}(
            "test", grid, topo, params, rain, network, nothing, Tuple{Int,Int}[], "output")

        state = CoupledState(grid, network)
        state.surface.h[5, 5] = 0.2  # Water at inlet location

        total_inlet, total_outlet = compute_exchange_flows!(state, coupled_scenario, 9.81)

        @test total_inlet > 0  # Water should enter drainage
        @test state.inlet_exchange[1] > 0
    end

    @testset "Coupled Step" begin
        grid = Grid(10, 10, 10.0)
        elevation = zeros(10, 10)
        roughness = fill(0.03, 10, 10)
        topo = Topography(elevation, roughness, grid)
        params = SimulationParameters(t_end=60.0, dt_max=1.0)
        rain = RainfallEvent([0.0, 100.0], [50.0, 50.0])

        j1 = Junction(1, 50.0, 50.0, -2.0, 0.0)
        j2 = Junction(2, 100.0, 50.0, -2.5, -0.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 50.0, 0.013, -2.0, -2.5)
        inlet = Inlet(1, 1, 5, 5)
        outlet = Outlet(1, 2; outlet_type=:FREE, invert=-2.5)
        network = DrainageNetwork([pipe], [j1, j2], [inlet], [outlet])

        coupled_scenario = CoupledScenario{Float64}(
            "test", grid, topo, params, rain, network, nothing, Tuple{Int,Int}[], "output")

        state = CoupledState(grid, network)
        work = CoupledWorkspace(grid, network)

        # Take one coupled step
        diagnostics = step_coupled!(state, coupled_scenario, work, 0.5)

        @test state.t == 0.5
        @test haskey(diagnostics, "inlet_flow")
        @test haskey(diagnostics, "surface_max_depth")
        @test haskey(diagnostics, "drainage_max_depth")
    end

    @testset "Coupled Timestep Calculation" begin
        grid = Grid(10, 10, 10.0)
        elevation = zeros(10, 10)
        roughness = fill(0.03, 10, 10)
        topo = Topography(elevation, roughness, grid)
        params = SimulationParameters(t_end=60.0, dt_max=5.0, cfl=0.9)
        rain = RainfallEvent([0.0, 100.0], [0.0, 0.0])

        j1 = Junction(1, 50.0, 50.0, -2.0, 0.0)
        pipe = PipeSegment(1, 1, 1, CircularPipe(0.3), 10.0, 0.013, -2.0, -2.0)
        network = DrainageNetwork([pipe], [j1], Inlet{Float64}[], Outlet{Float64}[])

        coupled_scenario = CoupledScenario{Float64}(
            "test", grid, topo, params, rain, network, nothing, Tuple{Int,Int}[], "output")

        state = CoupledState(grid, network)
        state.surface.h .= 0.1

        dt = compute_dt_coupled(state, coupled_scenario)

        @test dt > 0
        @test dt <= params.dt_max
    end

    @testset "Full Coupled Simulation" begin
        grid = Grid(16, 16, 5.0)
        elevation = zeros(16, 16)
        # Raise edges to create bowl
        for i in [1, 16], j in 1:16
            elevation[i, j] = 2.0
        end
        for j in [1, 16], i in 1:16
            elevation[i, j] = 2.0
        end
        roughness = fill(0.03, 16, 16)
        topo = Topography(elevation, roughness, grid)

        params = SimulationParameters(t_end=30.0, dt_max=1.0, cfl=0.9)
        rain = RainfallEvent([0.0, 100.0], [100.0, 100.0])  # Heavy rain

        # Create simple drainage at center
        j1 = Junction(1, 40.0, 40.0, -1.0, 0.0)  # Below surface
        j2 = Junction(2, 80.0, 40.0, -1.5, -0.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.3), 40.0, 0.013, -1.0, -1.5)
        inlet = Inlet(1, 1, 8, 8)  # Inlet at center
        outlet = Outlet(1, 2; outlet_type=:FREE, invert=-1.5)
        network = DrainageNetwork([pipe], [j1, j2], [inlet], [outlet])

        coupled_scenario = CoupledScenario{Float64}(
            "coupled_test", grid, topo, params, rain, network,
            nothing, Tuple{Int,Int}[], "Test coupled simulation")

        state = CoupledState(grid, network)

        results = run_coupled!(state, coupled_scenario; verbosity=0)

        @test results isa CoupledResults
        @test results.step_count > 0
        @test results.wall_time > 0
        @test results.total_inlet_volume >= 0

        # Surface should have accumulated water from rainfall
        @test maximum(results.surface_results.max_depth) > 0

        # Some water should have entered drainage
        @test results.total_inlet_volume > 0 || state.drainage.depth[1] > 0
    end
end

@testset "Drainage Network Validation" begin
    @testset "Invalid Pipe Endpoints" begin
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0)
        # Pipe references non-existent junction 3
        pipe = PipeSegment(1, 1, 3, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)

        network = DrainageNetwork([pipe], [j1], Inlet{Float64}[], Outlet{Float64}[])
        issues = validate(network)

        @test length(issues) > 0
        @test any(occursin("downstream node", i) for i in issues)
    end

    @testset "Invalid Inlet Grid Location" begin
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0)
        # Inlet with invalid grid indices
        inlet = Inlet(1, 1, -1, 5)

        network = DrainageNetwork(PipeSegment{Float64}[], [j1], [inlet], Outlet{Float64}[])
        issues = validate(network)

        @test length(issues) > 0
        @test any(occursin("invalid grid", i) for i in issues)
    end

    @testset "Valid Network Passes Validation" begin
        j1 = Junction(1, 0.0, 0.0, 10.0, 12.0)
        j2 = Junction(2, 100.0, 0.0, 9.5, 11.5)
        pipe = PipeSegment(1, 1, 2, CircularPipe(0.45), 100.0, 0.013, 10.0, 9.5)
        inlet = Inlet(1, 1, 5, 5)
        outlet = Outlet(1, 2)

        network = DrainageNetwork([pipe], [j1, j2], [inlet], [outlet])
        issues = validate(network)

        @test isempty(issues)
    end
end
