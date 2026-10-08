# src/types.jl

"""
    AbstractOptimizer

Abstract base type for all optimization algorithms in EdiBot.jl.
"""
abstract type AbstractOptimizer end

"""
    AbstractLoss

Abstract base type for loss functions in EdiBot.jl.
"""
abstract type AbstractLoss end

"""
    OptimizationProblem{F, T<:Real}

Defines a bounded numerical optimization problem.

# Fields
- `objective::F`: Objective/cost function `p -> Real` to minimize.
- `lower_bounds::Vector{T}`: Lower bounds for each parameter.
- `upper_bounds::Vector{T}`: Upper bounds for each parameter.
- `dim::Int`: Dimension of the parameter space.
- `param_names::Vector{Symbol}`: Optional symbolic names for the parameters.
"""
struct OptimizationProblem{F, T<:Real}
    objective::F
    lower_bounds::Vector{T}
    upper_bounds::Vector{T}
    dim::Int
    param_names::Vector{Symbol}

    function OptimizationProblem(
        objective::F,
        lower_bounds::AbstractVector{<:Real},
        upper_bounds::AbstractVector{<:Real};
        param_names::Vector{Symbol} = Symbol[]
    ) where {F}
        length(lower_bounds) == length(upper_bounds) || throw(
            DimensionMismatch("lower_bounds and upper_bounds must have identical lengths.")
        )
        all(lower_bounds .<= upper_bounds) || throw(
            ArgumentError("Each lower bound must be less than or equal to its upper bound.")
        )
        T = promote_type(eltype(lower_bounds), eltype(upper_bounds), Float64)
        lb = Vector{T}(lower_bounds)
        ub = Vector{T}(upper_bounds)
        dim = length(lb)
        pnames = isempty(param_names) ? [Symbol("p$i") for i in 1:dim] : param_names
        length(pnames) == dim || throw(
            DimensionMismatch("param_names length ($(length(pnames))) must match dimension ($dim).")
        )
        return new{F, T}(objective, lb, ub, dim, pnames)
    end
end

"""
    OptimizationOptions{T<:Real}

Configuration options for optimization algorithms.

# Fields
- `max_iters::Int`: Maximum number of iterations/generations (default: 500).
- `pop_size::Int`: Population / swarm size (default: 50).
- `tol_f::T`: Cost tolerance for early stopping (default: 1e-8).
- `tol_x::T`: Parameter movement tolerance (default: 1e-8).
- `patience::Int`: Number of stagnant iterations before early termination (default: 50).
- `verbose::Bool`: Whether to print progress logs during optimization (default: false).
- `seed::Union{Nothing, Int}`: Random number generator seed for reproducibility (default: nothing).
- `parallel::Bool`: Whether to use multi-threaded parallel fitness evaluation (default: true).
"""
Base.@kwdef struct OptimizationOptions{T<:Real}
    max_iters::Int = 500
    pop_size::Int = 50
    tol_f::T = 1e-8
    tol_x::T = 1e-8
    patience::Int = 50
    verbose::Bool = false
    seed::Union{Nothing, Int} = nothing
    parallel::Bool = true
end

"""
    OptimizationResult{T<:Real}

Stores the outcome and performance metrics of an optimization execution.

# Fields
- `best_params::Vector{T}`: Optimal parameter vector found.
- `best_cost::T`: Minimum cost/objective value achieved.
- `iterations::Int`: Number of iterations executed.
- `converged::Bool`: Whether the optimizer met convergence or early stopping criteria.
- `termination_reason::String`: Explanation of how the optimization concluded.
- `history::Vector{T}`: Convergence trajectory recording best cost per iteration.
- `runtime_seconds::Float64`: Wall-clock execution time in seconds.
- `param_names::Vector{Symbol}`: Names corresponding to parameters.
"""
struct OptimizationResult{T<:Real}
    best_params::Vector{T}
    best_cost::T
    iterations::Int
    converged::Bool
    termination_reason::String
    history::Vector{T}
    runtime_seconds::Float64
    param_names::Vector{Symbol}
end

"""
    FitResult{T<:Real}

High-level summary of a curve-fitting procedure, including statistical metrics and predicted values.

# Fields
- `opt_result::OptimizationResult{T}`: Underlying numerical optimization outcome.
- `x_data::Vector{T}`: Independent variable data points.
- `y_data::Vector{T}`: Dependent observed variable data points.
- `y_pred::Vector{T}`: Predicted values evaluated at `best_params`.
- `residuals::Vector{T}`: Residual vector `(y_data - y_pred)`.
- `metrics::Dict{Symbol, T}`: Goodness-of-fit statistics (R2, AdjR2, RMSE, MAE, AIC, BIC, ChiSq).
- `model_name::String`: Description or formula representation of the fitted model.
"""
struct FitResult{T<:Real}
    opt_result::OptimizationResult{T}
    x_data::Vector{T}
    y_data::Vector{T}
    y_pred::Vector{T}
    residuals::Vector{T}
    metrics::Dict{Symbol, T}
    model_name::String
end
