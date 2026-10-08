# src/algorithms/pso.jl

"""
    PSO <: AbstractOptimizer

Particle Swarm Optimization algorithm for continuous global optimization.

# Fields
- `w_start::Float64`: Starting inertia weight (default: 0.9).
- `w_end::Float64`: Ending inertia weight for linear decay (default: 0.4).
- `c1::Float64`: Cognitive / personal acceleration coefficient (default: 2.0).
- `c2::Float64`: Social / swarm acceleration coefficient (default: 2.0).
- `v_max_factor::Float64`: Maximum velocity as a fraction of domain span ``(ub - lb)`` (default: 0.2).
- `bounds_strategy::Symbol`: Boundary handling method (`:clamp`, `:reflect`, `:periodic`, `:random`). Default: `:clamp`.
"""
Base.@kwdef struct PSO <: AbstractOptimizer
    w_start::Float64 = 0.9
    w_end::Float64 = 0.4
    c1::Float64 = 2.0
    c2::Float64 = 2.0
    v_max_factor::Float64 = 0.2
    bounds_strategy::Symbol = :clamp
end

function solve_optimization(
    prob::OptimizationProblem{F, T},
    opt::PSO,
    options::OptimizationOptions{T}
) where {F, T<:Real}
    start_time = time()
    rng = options.seed === nothing ? Random.default_rng() : Random.MersenneTwister(options.seed)
    
    dim = prob.dim
    lb = prob.lower_bounds
    ub = prob.upper_bounds
    pop_size = max(options.pop_size, 4)
    
    span = ub .- lb
    v_max = opt.v_max_factor .* span
    
    # Initialize positions and velocities
    positions = initialize_population(lb, ub, pop_size; rng=rng)
    velocities = [zeros(T, dim) for _ in 1:pop_size]
    for i in 1:pop_size
        for d in 1:dim
            velocities[i][d] = (rand(rng, T) * 2 - 1) * v_max[d]
        end
    end
    
    # Personal and global bests
    pbest_positions = [copy(positions[i]) for i in 1:pop_size]
    fitnesses = Vector{T}(undef, pop_size)
    evaluate_population!(fitnesses, prob.objective, positions, options.parallel)
    pbest_fitnesses = copy(fitnesses)
    
    gbest_idx = argmin(fitnesses)
    gbest_cost = fitnesses[gbest_idx]
    gbest_pos = copy(positions[gbest_idx])
    
    history = T[gbest_cost]
    stagnant_count = 0
    converged = false
    term_reason = "Max iterations reached ($(options.max_iters))"
    
    for iter in 1:options.max_iters
        # Linearly decreasing inertia weight
        w = opt.w_start - ((opt.w_start - opt.w_end) * (iter / options.max_iters))
        
        for i in 1:pop_size
            for d in 1:dim
                r1 = rand(rng, T)
                r2 = rand(rng, T)
                
                cognitive = opt.c1 * r1 * (pbest_positions[i][d] - positions[i][d])
                social = opt.c2 * r2 * (gbest_pos[d] - positions[i][d])
                
                new_v = w * velocities[i][d] + cognitive + social
                velocities[i][d] = clamp(new_v, -v_max[d], v_max[d])
                
                positions[i][d] += velocities[i][d]
            end
            enforce_bounds!(positions[i], lb, ub; strategy=opt.bounds_strategy, rng=rng)
        end
        
        # Parallel fitness evaluation
        evaluate_population!(fitnesses, prob.objective, positions, options.parallel)
        
        prev_best = gbest_cost
        
        for i in 1:pop_size
            if fitnesses[i] < pbest_fitnesses[i]
                pbest_fitnesses[i] = fitnesses[i]
                pbest_positions[i] = copy(positions[i])
                
                if fitnesses[i] < gbest_cost
                    gbest_cost = fitnesses[i]
                    gbest_pos = copy(positions[i])
                end
            end
        end
        
        push!(history, gbest_cost)
        
        if options.verbose && (iter % 25 == 0 || iter == 1)
            Printf.@printf("PSO Iter %4d | Best Cost: %.6e\n", iter, gbest_cost)
        end
        
        # Check convergence & stagnation
        if abs(prev_best - gbest_cost) < options.tol_f
            stagnant_count += 1
        else
            stagnant_count = 0
        end
        
        if gbest_cost <= options.tol_f
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
        gbest_pos,
        gbest_cost,
        length(history) - 1,
        converged,
        term_reason,
        history,
        elapsed,
        prob.param_names
    )
end
