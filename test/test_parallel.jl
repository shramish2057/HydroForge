# HydroForge Parallel Computing Tests

using Test
using HydroForge
using Base.Threads: nthreads

@testset "Parallel Computing" begin

    @testset "Backend Types" begin
        # Test backend creation
        serial = SerialBackend()
        @test serial isa ComputeBackend

        threaded = ThreadedBackend()
        @test threaded isa ComputeBackend
        @test threaded.nthreads == nthreads()

        threaded4 = ThreadedBackend(4)
        @test threaded4.nthreads == 4

        gpu = GPUBackend()
        @test gpu isa ComputeBackend
        @test gpu.device_id == 0
    end

    @testset "Backend Selection" begin
        # Save current backend
        original = get_backend()

        # Test setting backends
        set_backend!(:serial)
        @test get_backend() isa SerialBackend

        set_backend!(:threaded)
        @test get_backend() isa ThreadedBackend

        # GPU should fail if CUDA not loaded
        @test_throws Exception set_backend!(:gpu)

        # Unknown backend should fail
        @test_throws Exception set_backend!(:unknown)

        # Restore original backend
        set_backend!(original)
    end

    @testset "Parallel Utilities" begin
        T = Float64

        # Test parallel_for
        A = zeros(T, 100)
        parallel_for(i -> A[i] = T(i), 1:100; backend=SerialBackend())
        @test A == T.(1:100)

        A = zeros(T, 100)
        parallel_for(i -> A[i] = T(i), 1:100; backend=ThreadedBackend())
        @test A == T.(1:100)

        # Test parallel_for_2d
        B = zeros(T, 10, 10)
        parallel_for_2d((i, j) -> B[i, j] = T(i + j), 10, 10; backend=SerialBackend())
        @test B[1, 1] == T(2)
        @test B[10, 10] == T(20)

        B = zeros(T, 10, 10)
        parallel_for_2d((i, j) -> B[i, j] = T(i + j), 10, 10; backend=ThreadedBackend())
        @test B[1, 1] == T(2)
        @test B[10, 10] == T(20)
    end

    @testset "Parallel Array Operations" begin
        T = Float64

        # Test parallel_fill!
        A = zeros(T, 100, 100)
        parallel_fill!(A, T(42); backend=SerialBackend())
        @test all(A .== T(42))

        A = zeros(T, 100, 100)
        parallel_fill!(A, T(42); backend=ThreadedBackend())
        @test all(A .== T(42))

        # Test parallel_copy!
        src = rand(T, 100, 100)
        dst = zeros(T, 100, 100)
        parallel_copy!(dst, src; backend=SerialBackend())
        @test dst == src

        dst = zeros(T, 100, 100)
        parallel_copy!(dst, src; backend=ThreadedBackend())
        @test dst == src

        # Test parallel_maximum
        A = rand(T, 100, 100)
        max_serial = parallel_maximum(A; backend=SerialBackend())
        max_threaded = parallel_maximum(A; backend=ThreadedBackend())
        @test max_serial == maximum(A)
        @test max_threaded == maximum(A)

        # Test parallel_sum
        A = rand(T, 100, 100)
        sum_serial = parallel_sum(A; backend=SerialBackend())
        sum_threaded = parallel_sum(A; backend=ThreadedBackend())
        @test isapprox(sum_serial, sum(A), rtol=1e-10)
        @test isapprox(sum_threaded, sum(A), rtol=1e-10)
    end

    @testset "Threaded Flux Computation" begin
        T = Float64
        nx, ny = 50, 50
        dx, dy = T(1.0), T(1.0)

        grid = Grid(nx, ny, dx, dy, zero(T), zero(T))
        h = fill(T(0.1), nx, ny)
        z = zeros(T, nx, ny)
        n = fill(T(0.03), nx, ny)
        qx = zeros(T, nx, ny)
        qy = zeros(T, nx, ny)
        qx_new = zeros(T, nx, ny)
        qy_new = zeros(T, nx, ny)

        params = SimulationParameters(T=T)
        dt = T(0.1)

        # Serial computation
        compute_flux_x!(qx_new, qx, h, z, n, grid, params, dt)
        qx_serial = copy(qx_new)

        # Threaded computation
        fill!(qx_new, zero(T))
        compute_flux_x_threaded!(qx_new, qx, h, z, n, grid, params, dt)
        qx_threaded = copy(qx_new)

        # Results should match
        @test isapprox(qx_serial, qx_threaded, rtol=1e-10)

        # Same for y-direction
        compute_flux_y!(qy_new, qy, h, z, n, grid, params, dt)
        qy_serial = copy(qy_new)

        fill!(qy_new, zero(T))
        compute_flux_y_threaded!(qy_new, qy, h, z, n, grid, params, dt)
        qy_threaded = copy(qy_new)

        @test isapprox(qy_serial, qy_threaded, rtol=1e-10)
    end

    @testset "Threaded Depth Update" begin
        T = Float64
        nx, ny = 50, 50
        dx, dy = T(1.0), T(1.0)

        grid = Grid(nx, ny, dx, dy, zero(T), zero(T))
        h_serial = fill(T(0.1), nx, ny)
        h_threaded = fill(T(0.1), nx, ny)
        qx = rand(T, nx, ny) * T(0.01)
        qy = rand(T, nx, ny) * T(0.01)
        dt = T(0.1)

        # Serial update
        update_depth!(h_serial, qx, qy, grid, dt)

        # Threaded update
        update_depth_threaded!(h_threaded, qx, qy, grid, dt)

        # Results should match
        @test isapprox(h_serial, h_threaded, rtol=1e-10)
    end

    @testset "ParallelWorkspace" begin
        T = Float64
        grid = Grid(32, 32, T(1.0))

        # Create with default backend
        work = ParallelWorkspace(grid)
        @test size(work.qx_new) == (32, 32)
        @test size(work.qy_new) == (32, 32)
        @test size(work.η) == (32, 32)

        # Create with specific backend
        work_threaded = ParallelWorkspace(grid; backend=ThreadedBackend())
        @test work_threaded.backend isa ThreadedBackend
    end

    @testset "Parallel Simulation" begin
        T = Float64
        nx, ny = 32, 32
        dx, dy = T(1.0), T(1.0)

        # Create scenario
        grid = Grid(nx, ny, dx, dy, zero(T), zero(T))
        elevation = zeros(T, nx, ny)
        roughness = fill(T(0.03), nx, ny)
        topo = Topography(elevation, roughness, grid)

        params = SimulationParameters(
            t_end=10.0,
            dt_max=1.0,
            cfl=0.5,
            T=T
        )

        rainfall = RainfallEvent(T[0, 5, 10], T[50, 50, 0])
        output_points = Tuple{Int,Int}[]
        output_dir = tempdir()

        scenario = Scenario("test", grid, topo, params, rainfall, output_points, output_dir)

        # Run with serial backend
        state_serial = SimulationState(grid)
        results_serial = run_simulation_parallel!(state_serial, scenario;
                                                  backend=SerialBackend(), verbosity=0)

        # Run with threaded backend
        state_threaded = SimulationState(grid)
        results_threaded = run_simulation_parallel!(state_threaded, scenario;
                                                    backend=ThreadedBackend(), verbosity=0)

        # Both should complete and have similar results
        @test results_serial.step_count > 0
        @test results_threaded.step_count > 0

        # Max depths should be similar (not exact due to different execution order)
        max_depth_serial = maximum(results_serial.accumulator.max_depth)
        max_depth_threaded = maximum(results_threaded.accumulator.max_depth)
        @test isapprox(max_depth_serial, max_depth_threaded, rtol=0.01)
    end

    @testset "Auto Backend Selection" begin
        T = Float64

        # Small grid should get serial
        small_grid = Grid(10, 10, T(1.0))
        backend_small = auto_select_backend(small_grid)
        # With single thread, should be serial; with multiple threads could be either
        @test backend_small isa ComputeBackend

        # Large grid should get threaded (if threads available)
        large_grid = Grid(200, 200, T(1.0))
        backend_large = auto_select_backend(large_grid)
        if nthreads() > 1
            @test backend_large isa ThreadedBackend
        else
            @test backend_large isa SerialBackend
        end
    end

    @testset "GPU Availability" begin
        # GPU should not be available by default (unless CUDA is loaded)
        @test gpu_available() == false

        # enable_gpu! should fail without CUDA
        @test_throws Exception enable_gpu!()
    end

    @testset "Backend Info" begin
        # backend_info should run without error
        # Just test that it doesn't throw
        @test begin
            backend_info()
            true
        end
    end

    @testset "Threaded Timestep Computation" begin
        T = Float64
        nx, ny = 50, 50
        dx, dy = T(1.0), T(1.0)

        grid = Grid(nx, ny, dx, dy, zero(T), zero(T))
        state = SimulationState(grid)
        state.h .= T(0.1)

        params = SimulationParameters(T=T)

        # Serial timestep
        dt_serial = compute_dt(state, grid, params)

        # Threaded timestep
        dt_threaded = compute_dt_threaded(state, grid, params)

        # Should be approximately equal
        @test isapprox(dt_serial, dt_threaded, rtol=1e-10)
    end

    @testset "Thread Count" begin
        # Verify thread info
        @test nthreads() >= 1
        threaded = ThreadedBackend()
        @test threaded.nthreads >= 1
    end
end
