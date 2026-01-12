# HydroForge Job Management
# Async job queue (placeholder for Phase 13)

"""
    Job

Represents a simulation job.
"""
mutable struct Job
    id::String
    status::Symbol  # :queued, :running, :completed, :failed
    progress::Float64
    result::Any
    error::Union{Exception, Nothing}
    created_at::DateTime
    started_at::Union{DateTime, Nothing}
    completed_at::Union{DateTime, Nothing}
end

# Job queue will be implemented in Phase 13
