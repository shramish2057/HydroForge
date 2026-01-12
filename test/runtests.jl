# HydroForge Test Suite
# Main test entry point

using Test
using HydroForge

@testset "HydroForge.jl" begin

    @testset "Types" begin
        include("test_types.jl")
    end

    @testset "Numerics" begin
        include("test_numerics.jl")
    end

    @testset "Physics" begin
        include("test_physics.jl")
    end

    @testset "Integration" begin
        include("test_integration.jl")
    end

end
