# src/algorithms/abstract_optimizer.jl

"""
    initialize_population(lb::Vector{T}, ub::Vector{T}, pop_size::Int; rng = Random.default_rng()) where {T<:Real}

Generates an initial uniform random population of parameter vectors inside the bounding box `[lb, ub]`.
"""
function initialize_population(
    lb::Vector{T},
    ub::Vector{T},
    pop_size::Int;
    rng::Random.AbstractRNG = Random.default_rng()
) where {T<:Real}
    dim = length(lb)
    pop = [Vector{T}(undef, dim) for _ in 1:pop_size]
    for i in 1:pop_size
        for d in 1:dim
            pop[i][d] = lb[d] + rand(rng, T) * (ub[d] - lb[d])
        end
    end
    return pop
end

"""
    enforce_bounds!(x::Vector{T}, lb::Vector{T}, ub::Vector{T}; strategy::Symbol = :clamp, rng = Random.default_rng()) where {T<:Real}

Applies boundary enforcement on vector `x` relative to `[lb, ub]`.

Supported strategies:
- `:clamp`: Clamps out-of-bound coordinates to the nearest bound.
- `:reflect`: Bounces out-of-bound values off the boundaries.
- `:periodic`: Wraps out-of-bound values modulo the interval width.
- `:random`: Reinitializes coordinate uniformly at random within bounds.
"""
function enforce_bounds!(
    x::Vector{T},
    lb::Vector{T},
    ub::Vector{T};
    strategy::Symbol = :clamp,
    rng::Random.AbstractRNG = Random.default_rng()
) where {T<:Real}
    dim = length(x)
    @inbounds for d in 1:dim
        low, high = lb[d], ub[d]
        width = high - low
        val = x[d]

        if val < low
            if strategy === :clamp
                x[d] = low
            elseif strategy === :reflect
                x[d] = min(high, low + (low - val))
            elseif strategy === :periodic
                x[d] = width > 0 ? high - mod(low - val, width) : low
            elseif strategy === :random
                x[d] = low + rand(rng, T) * width
            else
                x[d] = low
            end
        elseif val > high
            if strategy === :clamp
                x[d] = high
            elseif strategy === :reflect
                x[d] = max(low, high - (val - high))
            elseif strategy === :periodic
                x[d] = width > 0 ? low + mod(val - high, width) : high
            elseif strategy === :random
                x[d] = low + rand(rng, T) * width
            else
                x[d] = high
            end
        end
    end
    return x
end

"""
    evaluate_population!(fitnesses::Vector{T}, objective::F, population::Vector{Vector{T}}, parallel::Bool) where {F, T<:Real}

Evaluates the objective function on a population of parameter vectors, optionally using multi-threading.
Handles non-finite values (NaN / Inf) by assigning a large penalty.
"""
function evaluate_population!(
    fitnesses::Vector{T},
    objective::F,
    population::Vector{Vector{T}},
    parallel::Bool
) where {F, T<:Real}
    pop_size = length(population)
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in 1:pop_size
            fitnesses[i] = safe_eval(objective, population[i], T)
        end
    else
        for i in 1:pop_size
            fitnesses[i] = safe_eval(objective, population[i], T)
        end
    end
    return fitnesses
end

"""
    safe_eval(objective::F, x::Vector{T}, ::Type{T}) where {F, T<:Real}

Safely evaluates `objective(x)`, trapping NaNs, Infs, or domain exceptions.
"""
function safe_eval(objective::F, x::Vector{T}, ::Type{T}) where {F, T<:Real}
    try
        val = objective(x)
        if isnan(val) || isinf(val)
            return T(1e12)
        end
        return T(val)
    catch
        return T(1e12)
    end
end
