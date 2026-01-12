# HydroForge Parallel Computing Backends
#
# Supports multiple execution backends:
# - Serial: Single-threaded CPU (default)
# - Threaded: Multi-threaded CPU using Julia's threading
# - GPU: CUDA-based GPU acceleration (optional)
#
# Usage:
#   set_backend!(:threaded)  # Use multi-threading
#   set_backend!(:serial)    # Use single thread
#   set_backend!(:gpu)       # Use GPU (requires CUDA.jl)

using Base.Threads: @threads, nthreads, threadid

# =============================================================================
# Backend Types
# =============================================================================

"""
    ComputeBackend

Abstract type for computation backends.
"""
abstract type ComputeBackend end

"""
    SerialBackend <: ComputeBackend

Single-threaded CPU execution (default).
"""
struct SerialBackend <: ComputeBackend end

"""
    ThreadedBackend <: ComputeBackend

Multi-threaded CPU execution using Julia's native threading.
"""
struct ThreadedBackend <: ComputeBackend
    nthreads::Int
end
ThreadedBackend() = ThreadedBackend(nthreads())

"""
    GPUBackend <: ComputeBackend

GPU execution using CUDA.jl (requires CUDA.jl to be loaded).
"""
struct GPUBackend <: ComputeBackend
    device_id::Int
end
GPUBackend() = GPUBackend(0)

# Global backend setting
const CURRENT_BACKEND = Ref{ComputeBackend}(SerialBackend())

# GPU availability flag (set when CUDA is loaded)
const GPU_AVAILABLE = Ref{Bool}(false)

"""
    get_backend()

Get the current computation backend.
"""
get_backend() = CURRENT_BACKEND[]

"""
    set_backend!(backend::ComputeBackend)

Set the computation backend.
"""
function set_backend!(backend::ComputeBackend)
    if backend isa GPUBackend && !GPU_AVAILABLE[]
        error("GPU backend requires CUDA.jl. Load it with `using CUDA` first.")
    end
    CURRENT_BACKEND[] = backend
    backend
end

"""
    set_backend!(name::Symbol)

Set backend by name: :serial, :threaded, or :gpu
"""
function set_backend!(name::Symbol)
    if name == :serial
        set_backend!(SerialBackend())
    elseif name == :threaded
        set_backend!(ThreadedBackend())
    elseif name == :gpu
        set_backend!(GPUBackend())
    else
        error("Unknown backend: $name. Use :serial, :threaded, or :gpu")
    end
end

"""
    backend_info()

Print information about the current backend and available options.
"""
function backend_info()
    println("HydroForge Parallel Computing")
    println("=" ^ 40)
    println("Current backend: ", typeof(CURRENT_BACKEND[]))
    println()
    println("Available backends:")
    println("  • Serial (single-threaded)")
    println("  • Threaded ($(nthreads()) threads available)")
    if GPU_AVAILABLE[]
        println("  • GPU (CUDA available)")
    else
        println("  • GPU (not available - load CUDA.jl)")
    end
    println()
    println("Set backend with: set_backend!(:serial), set_backend!(:threaded), or set_backend!(:gpu)")
end

# =============================================================================
# Threaded Iteration Utilities
# =============================================================================

"""
    parallel_for(f, range; backend=get_backend())

Execute function `f(i)` for each `i` in `range` using the specified backend.
"""
function parallel_for(f::F, range; backend::ComputeBackend=get_backend()) where F
    _parallel_for(f, range, backend)
end

# Serial implementation
function _parallel_for(f::F, range, ::SerialBackend) where F
    @inbounds for i in range
        f(i)
    end
    nothing
end

# Threaded implementation
function _parallel_for(f::F, range, ::ThreadedBackend) where F
    @threads for i in range
        @inbounds f(i)
    end
    nothing
end

"""
    parallel_for_2d(f, nx, ny; backend=get_backend())

Execute function `f(i, j)` for each cell (i,j) in an nx×ny grid.
"""
function parallel_for_2d(f::F, nx::Int, ny::Int; backend::ComputeBackend=get_backend()) where F
    _parallel_for_2d(f, nx, ny, backend)
end

# Serial 2D implementation
function _parallel_for_2d(f::F, nx::Int, ny::Int, ::SerialBackend) where F
    @inbounds for j in 1:ny
        for i in 1:nx
            f(i, j)
        end
    end
    nothing
end

# Threaded 2D implementation - parallelize over rows (better cache locality)
function _parallel_for_2d(f::F, nx::Int, ny::Int, ::ThreadedBackend) where F
    @threads for j in 1:ny
        @inbounds for i in 1:nx
            f(i, j)
        end
    end
    nothing
end

"""
    parallel_reduce(f, op, init, range; backend=get_backend())

Parallel reduction: compute `op(op(init, f(1)), f(2)), ...)` over range.
"""
function parallel_reduce(f::F, op::Op, init::T, range;
                         backend::ComputeBackend=get_backend()) where {F, Op, T}
    _parallel_reduce(f, op, init, range, backend)
end

# Serial reduction
function _parallel_reduce(f::F, op::Op, init::T, range, ::SerialBackend) where {F, Op, T}
    result = init
    @inbounds for i in range
        result = op(result, f(i))
    end
    result
end

# Threaded reduction with per-thread accumulators
function _parallel_reduce(f::F, op::Op, init::T, range, backend::ThreadedBackend) where {F, Op, T}
    nt = backend.nthreads
    accums = [init for _ in 1:nt]

    @threads for i in range
        tid = threadid()
        @inbounds accums[tid] = op(accums[tid], f(i))
    end

    # Combine thread-local results
    result = init
    for a in accums
        result = op(result, a)
    end
    result
end

# =============================================================================
# Parallel Array Operations
# =============================================================================

"""
    parallel_fill!(A, val; backend=get_backend())

Fill array with value in parallel.
"""
function parallel_fill!(A::AbstractArray{T}, val::T;
                        backend::ComputeBackend=get_backend()) where T
    _parallel_fill!(A, val, backend)
end

function _parallel_fill!(A::AbstractArray{T}, val::T, ::SerialBackend) where T
    fill!(A, val)
end

function _parallel_fill!(A::AbstractArray{T}, val::T, ::ThreadedBackend) where T
    @threads for i in eachindex(A)
        @inbounds A[i] = val
    end
    nothing
end

"""
    parallel_copy!(dst, src; backend=get_backend())

Copy array in parallel.
"""
function parallel_copy!(dst::AbstractArray{T}, src::AbstractArray{T};
                        backend::ComputeBackend=get_backend()) where T
    _parallel_copy!(dst, src, backend)
end

function _parallel_copy!(dst::AbstractArray{T}, src::AbstractArray{T}, ::SerialBackend) where T
    copyto!(dst, src)
end

function _parallel_copy!(dst::AbstractArray{T}, src::AbstractArray{T}, ::ThreadedBackend) where T
    @threads for i in eachindex(dst, src)
        @inbounds dst[i] = src[i]
    end
    nothing
end

"""
    parallel_maximum(A; backend=get_backend())

Find maximum value in array in parallel.
"""
function parallel_maximum(A::AbstractArray{T};
                          backend::ComputeBackend=get_backend()) where T
    _parallel_maximum(A, backend)
end

function _parallel_maximum(A::AbstractArray{T}, ::SerialBackend) where T
    maximum(A)
end

function _parallel_maximum(A::AbstractArray{T}, backend::ThreadedBackend) where T
    nt = backend.nthreads
    n = length(A)
    chunk_size = cld(n, nt)

    maxvals = fill(typemin(T), nt)

    @threads for tid in 1:nt
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, n)
        local_max = typemin(T)
        @inbounds for i in start_idx:end_idx
            if A[i] > local_max
                local_max = A[i]
            end
        end
        maxvals[tid] = local_max
    end

    maximum(maxvals)
end

"""
    parallel_sum(A; backend=get_backend())

Sum array values in parallel.
"""
function parallel_sum(A::AbstractArray{T};
                      backend::ComputeBackend=get_backend()) where T
    _parallel_sum(A, backend)
end

function _parallel_sum(A::AbstractArray{T}, ::SerialBackend) where T
    sum(A)
end

function _parallel_sum(A::AbstractArray{T}, backend::ThreadedBackend) where T
    nt = backend.nthreads
    n = length(A)
    chunk_size = cld(n, nt)

    partial_sums = zeros(T, nt)

    @threads for tid in 1:nt
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, n)
        local_sum = zero(T)
        @inbounds for i in start_idx:end_idx
            local_sum += A[i]
        end
        partial_sums[tid] = local_sum
    end

    sum(partial_sums)
end

# =============================================================================
# GPU Support Stubs (activated when CUDA.jl is loaded)
# =============================================================================

# These are placeholder implementations for GPU backend
# They will be overwritten when CUDA extension is loaded

function _parallel_for(f::F, range, ::GPUBackend) where F
    error("GPU backend not available. Load CUDA.jl with `using CUDA` first.")
end

function _parallel_for_2d(f::F, nx::Int, ny::Int, ::GPUBackend) where F
    error("GPU backend not available. Load CUDA.jl with `using CUDA` first.")
end

function _parallel_fill!(A::AbstractArray{T}, val::T, ::GPUBackend) where T
    error("GPU backend not available. Load CUDA.jl with `using CUDA` first.")
end

function _parallel_copy!(dst::AbstractArray{T}, src::AbstractArray{T}, ::GPUBackend) where T
    error("GPU backend not available. Load CUDA.jl with `using CUDA` first.")
end

function _parallel_maximum(A::AbstractArray{T}, ::GPUBackend) where T
    error("GPU backend not available. Load CUDA.jl with `using CUDA` first.")
end

function _parallel_sum(A::AbstractArray{T}, ::GPUBackend) where T
    error("GPU backend not available. Load CUDA.jl with `using CUDA` first.")
end
