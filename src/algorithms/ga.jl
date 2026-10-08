# src/algorithms/ga.jl

"""
    DE <: AbstractOptimizer

Differential Evolution optimizer for continuous global parameter estimation.

# Fields
- `strategy::Symbol`: Mutation scheme (`:rand1bin`, `:best1bin`, `:current_to_best1bin`). Default: `:rand1bin`.
- `F::Float64`: Differential mutation scale factor ``F \\in [0.1, 1.5]`` (default: 0.8).
- `CR::Float64`: Crossover probability ``CR \\in [0.0, 1.0]`` (default: 0.9).
- `bounds_strategy::Symbol`: Method to handle boundary violations (`:clamp`, `:reflect`, `:periodic`, `:random`). Default: `:clamp`.
"""
Base.@kwdef struct DE <: AbstractOptimizer
    strategy::Symbol = :rand1bin
    F::Float64 = 0.8
    CR::Float64 = 0.9
    bounds_strategy::Symbol = :clamp
end

"""
    GA <: AbstractOptimizer

Real-coded Genetic Algorithm with Simulated Binary Crossover (SBX), Polynomial Mutation, and Tournament Selection.

# Fields
- `crossover_rate::Float64`: Probability of mating pairs ``p_c \\in [0.0, 1.0]`` (default: 0.9).
- `mutation_rate::Float64`: Per-gene mutation probability ``p_m`` (default: -1.0, auto-set to ``1/\\text{dim}``).
- `eta_c::Float64`: SBX distribution index (default: 15.0).
- `eta_m::Float64`: Polynomial mutation distribution index (default: 20.0).
- `tournament_size::Int`: Number of individuals in tournament selection (default: 3).
- `elitism::Int`: Number of best individuals directly preserved each generation (default: 2).
- `bounds_strategy::Symbol`: Boundary handling method (default: `:clamp`).
"""
Base.@kwdef struct GA <: AbstractOptimizer
    crossover_rate::Float64 = 0.9
    mutation_rate::Float64 = -1.0
    eta_c::Float64 = 15.0
    eta_m::Float64 = 20.0
    tournament_size::Int = 3
    elitism::Int = 2
    bounds_strategy::Symbol = :clamp
end

# -----------------------------------------------------------------------------
# Differential Evolution Solver Implementation
# -----------------------------------------------------------------------------

function solve_optimization(
    prob::OptimizationProblem{F, T},
    opt::DE,
    options::OptimizationOptions{T}
) where {F, T<:Real}
    start_time = time()
    rng = options.seed === nothing ? Random.default_rng() : Random.MersenneTwister(options.seed)
    
    dim = prob.dim
    lb = prob.lower_bounds
    ub = prob.upper_bounds
    pop_size = max(options.pop_size, 4)
    
    pop = initialize_population(lb, ub, pop_size; rng=rng)
    fitnesses = Vector{T}(undef, pop_size)
    evaluate_population!(fitnesses, prob.objective, pop, options.parallel)
    
    best_idx = argmin(fitnesses)
    best_cost = fitnesses[best_idx]
    best_params = copy(pop[best_idx])
    
    history = T[best_cost]
    stagnant_count = 0
    converged = false
    term_reason = "Max iterations reached ($(options.max_iters))"
    
    trials = [Vector{T}(undef, dim) for _ in 1:pop_size]
    trial_fitnesses = Vector{T}(undef, pop_size)
    
    for iter in 1:options.max_iters
        # Generate mutant and trial vectors
        for i in 1:pop_size
            r1, r2, r3 = select_distinct_indices(pop_size, i; rng=rng)
            j_rand = rand(rng, 1:dim)
            
            for d in 1:dim
                if rand(rng) < opt.CR || d == j_rand
                    if opt.strategy === :rand1bin
                        trials[i][d] = pop[r1][d] + opt.F * (pop[r2][d] - pop[r3][d])
                    elseif opt.strategy === :best1bin
                        trials[i][d] = best_params[d] + opt.F * (pop[r1][d] - pop[r2][d])
                    elseif opt.strategy === :current_to_best1bin
                        trials[i][d] = pop[i][d] + opt.F * (best_params[d] - pop[i][d]) + opt.F * (pop[r1][d] - pop[r2][d])
                    else
                        trials[i][d] = pop[r1][d] + opt.F * (pop[r2][d] - pop[r3][d])
                    end
                else
                    trials[i][d] = pop[i][d]
                end
            end
            enforce_bounds!(trials[i], lb, ub; strategy=opt.bounds_strategy, rng=rng)
        end
        
        # Parallel evaluation of trial vectors
        evaluate_population!(trial_fitnesses, prob.objective, trials, options.parallel)
        
        # Selection step
        prev_best = best_cost
        for i in 1:pop_size
            if trial_fitnesses[i] <= fitnesses[i]
                pop[i] = copy(trials[i])
                fitnesses[i] = trial_fitnesses[i]
                if fitnesses[i] < best_cost
                    best_cost = fitnesses[i]
                    best_params = copy(pop[i])
                end
            end
        end
        
        push!(history, best_cost)
        
        if options.verbose && (iter % 25 == 0 || iter == 1)
            Printf.@printf("DE Iter %4d | Best Cost: %.6e\n", iter, best_cost)
        end
        
        # Check tolerance & stagnation
        if abs(prev_best - best_cost) < options.tol_f
            stagnant_count += 1
        else
            stagnant_count = 0
        end
        
        if best_cost <= options.tol_f
            converged = true
            term_reason = "Cost tolerance satisfied (f <= $(options.tol_f))"
            break
        elseif stagnant_count >= options.patience
            converged = true
            term_reason = "Early stopping due to stagnation over $(options.patience) iterations"
            break
        end
    end
    
    elapsed = time() - start_time
    return OptimizationResult{T}(
        best_params,
        best_cost,
        length(history) - 1,
        converged,
        term_reason,
        history,
        elapsed,
        prob.param_names
    )
end

function select_distinct_indices(n::Int, exclude::Int; rng = Random.default_rng())
    r1 = rand(rng, 1:n)
    while r1 == exclude
        r1 = rand(rng, 1:n)
    end
    r2 = rand(rng, 1:n)
    while r2 == exclude || r2 == r1
        r2 = rand(rng, 1:n)
    end
    r3 = rand(rng, 1:n)
    while r3 == exclude || r3 == r1 || r3 == r2
        r3 = rand(rng, 1:n)
    end
    return r1, r2, r3
end

# -----------------------------------------------------------------------------
# Real-Coded Genetic Algorithm Solver Implementation
# -----------------------------------------------------------------------------

function solve_optimization(
    prob::OptimizationProblem{F, T},
    opt::GA,
    options::OptimizationOptions{T}
) where {F, T<:Real}
    start_time = time()
    rng = options.seed === nothing ? Random.default_rng() : Random.MersenneTwister(options.seed)
    
    dim = prob.dim
    lb = prob.lower_bounds
    ub = prob.upper_bounds
    pop_size = max(options.pop_size, 4)
    pm = opt.mutation_rate <= 0 ? 1.0 / dim : opt.mutation_rate
    
    pop = initialize_population(lb, ub, pop_size; rng=rng)
    fitnesses = Vector{T}(undef, pop_size)
    evaluate_population!(fitnesses, prob.objective, pop, options.parallel)
    
    best_idx = argmin(fitnesses)
    best_cost = fitnesses[best_idx]
    best_params = copy(pop[best_idx])
    
    history = T[best_cost]
    stagnant_count = 0
    converged = false
    term_reason = "Max iterations reached ($(options.max_iters))"
    
    new_pop = [Vector{T}(undef, dim) for _ in 1:pop_size]
    
    for iter in 1:options.max_iters
        # Elitism: copy best individuals
        sorted_indices = sortperm(fitnesses)
        n_elite = min(opt.elitism, pop_size)
        for e in 1:n_elite
            new_pop[e] = copy(pop[sorted_indices[e]])
        end
        
        # Breed remaining population
        idx = n_elite + 1
        while idx <= pop_size
            p1 = tournament_select(fitnesses, opt.tournament_size; rng=rng)
            p2 = tournament_select(fitnesses, opt.tournament_size; rng=rng)
            
            c1, c2 = copy(pop[p1]), copy(pop[p2])
            if rand(rng) < opt.crossover_rate
                c1, c2 = sbx_crossover(pop[p1], pop[p2], lb, ub, opt.eta_c; rng=rng)
            end
            
            polynomial_mutation!(c1, lb, ub, pm, opt.eta_m; rng=rng)
            enforce_bounds!(c1, lb, ub; strategy=opt.bounds_strategy, rng=rng)
            new_pop[idx] = c1
            idx += 1
            
            if idx <= pop_size
                polynomial_mutation!(c2, lb, ub, pm, opt.eta_m; rng=rng)
                enforce_bounds!(c2, lb, ub; strategy=opt.bounds_strategy, rng=rng)
                new_pop[idx] = c2
                idx += 1
            end
        end
        
        pop = [copy(p) for p in new_pop]
        evaluate_population!(fitnesses, prob.objective, pop, options.parallel)
        
        cur_best_idx = argmin(fitnesses)
        prev_best = best_cost
        if fitnesses[cur_best_idx] < best_cost
            best_cost = fitnesses[cur_best_idx]
            best_params = copy(pop[cur_best_idx])
        end
        
        push!(history, best_cost)
        
        if options.verbose && (iter % 25 == 0 || iter == 1)
            Printf.@printf("GA Iter %4d | Best Cost: %.6e\n", iter, best_cost)
        end
        
        if abs(prev_best - best_cost) < options.tol_f
            stagnant_count += 1
        else
            stagnant_count = 0
        end
        
        if best_cost <= options.tol_f
            converged = true
            term_reason = "Cost tolerance satisfied (f <= $(options.tol_f))"
            break
        elseif stagnant_count >= options.patience
            converged = true
            term_reason = "Early stopping due to stagnation over $(options.patience) iterations"
            break
        end
    end
    
    elapsed = time() - start_time
    return OptimizationResult{T}(
        best_params,
        best_cost,
        length(history) - 1,
        converged,
        term_reason,
        history,
        elapsed,
        prob.param_names
    )
end

function tournament_select(fitnesses::Vector{T}, k::Int; rng = Random.default_rng()) where {T}
    n = length(fitnesses)
    best_idx = rand(rng, 1:n)
    best_fit = fitnesses[best_idx]
    for _ in 2:k
        cand = rand(rng, 1:n)
        if fitnesses[cand] < best_fit
            best_fit = fitnesses[cand]
            best_idx = cand
        end
    end
    return best_idx
end

function sbx_crossover(
    parent1::Vector{T},
    parent2::Vector{T},
    lb::Vector{T},
    ub::Vector{T},
    eta_c::Float64;
    rng = Random.default_rng()
) where {T<:Real}
    dim = length(parent1)
    child1 = Vector{T}(undef, dim)
    child2 = Vector{T}(undef, dim)
    
    for d in 1:dim
        p1 = parent1[d]
        p2 = parent2[d]
        if rand(rng) <= 0.5 && abs(p1 - p2) > 1e-14
            y1 = min(p1, p2)
            y2 = max(p1, p2)
            yl = lb[d]
            yu = ub[d]
            
            u = rand(rng)
            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - beta^(-(eta_c + 1.0))
            beta_q = u <= (1.0 / alpha) ? (u * alpha)^(1.0 / (eta_c + 1.0)) : (1.0 / (2.0 - u * alpha))^(1.0 / (eta_c + 1.0))
            c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
            
            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
            alpha = 2.0 - beta^(-(eta_c + 1.0))
            beta_q = u <= (1.0 / alpha) ? (u * alpha)^(1.0 / (eta_c + 1.0)) : (1.0 / (2.0 - u * alpha))^(1.0 / (eta_c + 1.0))
            c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
            
            child1[d] = clamp(c1, yl, yu)
            child2[d] = clamp(c2, yl, yu)
        else
            child1[d] = p1
            child2[d] = p2
        end
    end
    return child1, child2
end

function polynomial_mutation!(
    x::Vector{T},
    lb::Vector{T},
    ub::Vector{T},
    pm::Float64,
    eta_m::Float64;
    rng = Random.default_rng()
) where {T<:Real}
    dim = length(x)
    for d in 1:dim
        if rand(rng) < pm
            y = x[d]
            yl = lb[d]
            yu = ub[d]
            delta1 = (y - yl) / (yu - yl + 1e-15)
            delta2 = (yu - y) / (yu - yl + 1e-15)
            u = rand(rng)
            mut_pow = 1.0 / (eta_m + 1.0)
            if u <= 0.5
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy^(eta_m + 1.0))
                delta_q = val^mut_pow - 1.0
            else
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy^(eta_m + 1.0))
                delta_q = 1.0 - val^mut_pow
            end
            x[d] = clamp(y + delta_q * (yu - yl), yl, yu)
        end
    end
    return x
end
