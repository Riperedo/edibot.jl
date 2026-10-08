# src/EdiBot.jl

module EdiBot

using LinearAlgebra
using Statistics
using Random
using Printf

# Core Types
include("types.jl")

# Loss Functions
include("loss.jl")

# Optimization Infrastructure & Algorithms
include("algorithms/abstract_optimizer.jl")
include("algorithms/ga.jl")
include("algorithms/pso.jl")

# Models Library
include("models/relaxation.jl")
include("models/structure_factor.jl")
include("models/standard.jl")

# Analytical String Parser
include("parser.jl")

# Statistics & Metrics
include("metrics.jl")

# I/O & Exporters
include("io.jl")

# CLI Support
include("cli.jl")

# -----------------------------------------------------------------------------
# High-Level Public API: optimize and fit
# -----------------------------------------------------------------------------

"""
    optimize(
        objective::Function;
        lower_bounds::AbstractVector{<:Real},
        upper_bounds::AbstractVector{<:Real},
        algorithm::AbstractOptimizer = PSO(),
        options::OptimizationOptions = OptimizationOptions(),
        param_names::Vector{Symbol} = Symbol[]
    ) -> OptimizationResult

Minimizes the scalar objective function `objective(p)` within hypercube bounds `[lower_bounds, upper_bounds]`.

# Example
```julia
using EdiBot
rosenbrock(p) = (1.0 - p[1])^2 + 100.0 * (p[2] - p[1]^2)^2
res = optimize(rosenbrock; lower_bounds=[-5.0, -5.0], upper_bounds=[5.0, 5.0], algorithm=PSO())
```
"""
function optimize(
    objective::Function;
    lower_bounds::AbstractVector{<:Real},
    upper_bounds::AbstractVector{<:Real},
    algorithm::AbstractOptimizer = PSO(),
    options::OptimizationOptions = OptimizationOptions(),
    param_names::Vector{Symbol} = Symbol[]
)
    prob = OptimizationProblem(objective, lower_bounds, upper_bounds; param_names=param_names)
    return solve_optimization(prob, algorithm, options)
end

"""
    fit(
        model::Union{Function, Symbol, String},
        x_data::AbstractVector{<:Real},
        y_data::AbstractVector{<:Real};
        lower_bounds::AbstractVector{<:Real},
        upper_bounds::AbstractVector{<:Real},
        algorithm::AbstractOptimizer = PSO(),
        loss::AbstractLoss = MSE(),
        options::OptimizationOptions = OptimizationOptions(),
        param_names::Vector{Symbol} = Symbol[],
        model_name::String = ""
    ) -> FitResult

Fits an analytical or physical scalar function `y = f(x, p)` to experimental/simulation data `(x_data, y_data)`.

# Model Specifications
- `Function`: Custom Julia callable `(x, p) -> Real`
- `Symbol`: Built-in physical model (`:vft`, `:vft_log10`, `:kww`, `:hard_sphere_sq`, `:yukawa_sq`, `:gaussian`, `:lorentzian`, `:power_law`, `:sigmoid`, `:arrhenius`)
- `String`: Mathematical formula string parsed automatically, e.g. `"a * exp(-(x / b)^c)"`

# Example
```julia
using EdiBot
t = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
phi = [0.95, 0.81, 0.62, 0.41, 0.12, 0.03]
result = fit(:kww, t, phi; lower_bounds=[0.5, 0.1, 0.1], upper_bounds=[1.5, 20.0, 1.0], algorithm=DE())
```
"""
function fit(
    model::Function,
    x_data::AbstractVector{<:Real},
    y_data::AbstractVector{<:Real};
    lower_bounds::AbstractVector{<:Real},
    upper_bounds::AbstractVector{<:Real},
    algorithm::AbstractOptimizer = PSO(),
    loss::AbstractLoss = MSE(),
    options::OptimizationOptions = OptimizationOptions(),
    param_names::Vector{Symbol} = Symbol[],
    model_name::String = "Custom Function"
)
    length(x_data) == length(y_data) || throw(DimensionMismatch("x_data and y_data must have the same length."))
    
    # Preallocate evaluation buffers for speed
    n_points = length(x_data)
    y_buf = zeros(Float64, n_points)
    
    # Define optimization objective
    function objective(p::AbstractVector{<:Real})
        @inbounds for i in 1:n_points
            y_buf[i] = model(x_data[i], p)
        end
        return loss(y_data, y_buf)
    end
    
    opt_res = optimize(
        objective;
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds,
        algorithm = algorithm,
        options = options,
        param_names = param_names
    )
    
    # Compute final predictions and metrics
    best_p = opt_res.best_params
    y_pred = [Float64(model(x, best_p)) for x in x_data]
    residuals = [Float64(y_data[i] - y_pred[i]) for i in 1:n_points]
    metrics = evaluate_metrics(y_data, y_pred, length(best_p))
    
    return FitResult(
        opt_res,
        Vector{Float64}(x_data),
        Vector{Float64}(y_data),
        y_pred,
        residuals,
        metrics,
        model_name
    )
end

function fit(
    model_sym::Symbol,
    x_data::AbstractVector{<:Real},
    y_data::AbstractVector{<:Real};
    kwargs...
)
    fn, default_name, default_params = resolve_builtin_model(model_sym)
    
    # Extract kwargs with defaults
    mname = get(kwargs, :model_name, default_name)
    pnames = get(kwargs, :param_names, default_params)
    kw_filtered = filter(p -> p.first != :model_name && p.first != :param_names, kwargs)
    
    return fit(fn, x_data, y_data; model_name=mname, param_names=pnames, kw_filtered...)
end

function fit(
    expr_str::String,
    x_data::AbstractVector{<:Real},
    y_data::AbstractVector{<:Real};
    kwargs...
)
    pnames_kw = get(kwargs, :param_names, Symbol[])
    fn, parsed_names = parse_expression(expr_str; param_names=pnames_kw)
    
    mname = get(kwargs, :model_name, expr_str)
    pnames = isempty(pnames_kw) ? parsed_names : pnames_kw
    kw_filtered = filter(p -> p.first != :model_name && p.first != :param_names, kwargs)
    
    return fit(fn, x_data, y_data; model_name=mname, param_names=pnames, kw_filtered...)
end

function resolve_builtin_model(sym::Symbol)
    if sym === :vft
        return vft, "Vogel-Fulcher-Tammann (VFT)", [:tau0, :B, :T0]
    elseif sym === :vft_log10
        return vft_log10, "VFT (Log10 scale)", [:log10_tau0, :B, :T0]
    elseif sym === :kww
        return kww, "Kohlrausch-Williams-Watts (KWW)", [:A, :tau, :beta, :y0]
    elseif sym === :kww3
        return (x, p) -> kww(x, p[1], p[2], p[3], 0.0), "KWW (3-parameter)", [:A, :tau, :beta]
    elseif sym === :arrhenius
        return arrhenius, "Arrhenius", [:tau0, :Ea]
    elseif sym === :hard_sphere_sq || sym === :hs
        return hard_sphere_sq, "Hard Sphere S(q) Percus-Yevick", [:phi, :sigma, :scale]
    elseif sym === :yukawa_sq || sym === :yukawa
        return yukawa_rpa_sq, "Yukawa RPA S(q)", [:phi, :sigma, :K, :z]
    elseif sym === :gaussian
        return gaussian, "Gaussian", [:A, :mu, :sigma, :y0]
    elseif sym === :lorentzian
        return lorentzian, "Lorentzian", [:A, :x0, :gamma, :y0]
    elseif sym === :power_law
        return power_law, "Power Law", [:A, :k, :y0]
    elseif sym === :exponential_decay
        return exponential_decay, "Exponential Decay", [:A, :tau, :y0]
    elseif sym === :sigmoid
        return sigmoid, "Sigmoid", [:L, :k, :x0, :y0]
    else
        throw(ArgumentError("Unknown built-in model :$sym. Available models: :vft, :vft_log10, :kww, :arrhenius, :hard_sphere_sq, :yukawa_sq, :gaussian, :lorentzian, :power_law, :exponential_decay, :sigmoid"))
    end
end

# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

# Main Entry Points
export fit, optimize

# Core Types
export AbstractOptimizer, AbstractLoss
export OptimizationProblem, OptimizationOptions, OptimizationResult, FitResult

# Optimizers
export GA, DE, PSO

# Losses
export MSE, RMSE, MAE, WeightedMSE, LogMSE, CustomLoss

# Built-in Physical & Standard Models
export vft, vft_log10, kww, arrhenius
export hard_sphere_sq, yukawa_rpa_sq
export gaussian, lorentzian, power_law, exponential_decay, sigmoid, polynomial

# Analytical Parser & Statistics & I/O
export parse_expression, evaluate_metrics
export load_dataset, export_json, export_latex_report

# CLI Entrypoint
export run_cli

end # module
