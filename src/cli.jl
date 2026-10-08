# src/cli.jl

"""
    run_cli(args::Vector{String} = ARGS)

Command-line interface entry point for EdiBot.jl.
"""
function run_cli(args::Vector{String} = ARGS)
    if isempty(args) || "--help" in args || "-h" in args || "help" in args
        print_cli_help()
        return 0
    end
    
    # Parse CLI flags
    parsed_args = parse_cli_args(args)
    
    if !haskey(parsed_args, :data)
        println(stderr, "Error: Missing required argument --data <filepath>")
        print_cli_help()
        return 1
    end
    
    data_file = parsed_args[:data]
    if !isfile(data_file)
        println(stderr, "Error: Data file not found: $data_file")
        return 1
    end
    
    # Load dataset
    x_data, y_data, weights = load_dataset(data_file)
    println("Loaded dataset from '$data_file' with $(length(x_data)) points.")
    
    # Parse bounds
    if !haskey(parsed_args, :bounds)
        println(stderr, "Error: Missing required argument --bounds <bounds_spec>")
        println(stderr, "Example: --bounds \"0..1, 0.1..10, 0.1..2\" or --bounds \"A:0..1, tau:0.1..10, beta:0.1..2\"")
        return 1
    end
    
    lb, ub, named_bounds_map = parse_bounds_spec(parsed_args[:bounds])
    
    # Select algorithm
    algo_str = lowercase(get(parsed_args, :algo, "pso"))
    algo = if algo_str == "pso"
        PSO()
    elseif algo_str == "de"
        DE()
    elseif algo_str == "ga"
        GA()
    else
        println(stderr, "Warning: Unknown algorithm '$algo_str', defaulting to PSO.")
        PSO()
    end
    
    # Select loss
    loss_str = lowercase(get(parsed_args, :loss, "mse"))
    loss = if loss_str == "mse"
        MSE()
    elseif loss_str == "rmse"
        RMSE()
    elseif loss_str == "mae"
        MAE()
    elseif loss_str == "log_mse" || loss_str == "logmse"
        LogMSE()
    elseif loss_str == "weighted"
        weights !== nothing ? WeightedMSE(weights) : WeightedMSE(; x_data=x_data)
    else
        MSE()
    end
    
    # Configure options
    max_iters = parse(Int, get(parsed_args, :max_iters, "500"))
    pop_size = parse(Int, get(parsed_args, :pop_size, "50"))
    verbose = haskey(parsed_args, :verbose)
    seed_val = haskey(parsed_args, :seed) ? parse(Int, parsed_args[:seed]) : nothing
    
    options = OptimizationOptions(
        max_iters = max_iters,
        pop_size = pop_size,
        verbose = verbose,
        seed = seed_val
    )
    
    # Explicit custom param names if passed
    custom_params = if haskey(parsed_args, :params)
        [Symbol(strip(s)) for s in split(parsed_args[:params], ",")]
    else
        Symbol[]
    end
    
    # Run fit
    result = if haskey(parsed_args, :expr)
        expr_str = parsed_args[:expr]
        println("Fitting analytical expression: $expr_str")
        
        # If named bounds were provided without explicit --params, resolve them
        fn, detected_pnames = parse_expression(expr_str; param_names=custom_params)
        target_pnames = isempty(custom_params) ? detected_pnames : custom_params
        
        final_lb, final_ub = if !isempty(named_bounds_map)
            align_named_bounds(target_pnames, named_bounds_map, lb, ub)
        else
            lb, ub
        end
        
        fit(fn, x_data, y_data; lower_bounds=final_lb, upper_bounds=final_ub, algorithm=algo, loss=loss, options=options, param_names=target_pnames, model_name=expr_str)
    elseif haskey(parsed_args, :model)
        m_sym = Symbol(lowercase(parsed_args[:model]))
        println("Fitting built-in model: :$m_sym")
        fit(m_sym, x_data, y_data; lower_bounds=lb, upper_bounds=ub, algorithm=algo, loss=loss, options=options, param_names=custom_params)
    else
        println(stderr, "Error: Must specify either --model <name> or --expr <formula>")
        return 1
    end
    
    # Print results summary
    println("\n" * "="^60)
    println("   EdiBot Optimization Summary")
    println("="^60)
    println("Model:            ", result.model_name)
    println("Algorithm:        ", algo_str)
    println("Status:           ", result.opt_result.termination_reason)
    println("Iterations:       ", result.opt_result.iterations)
    Printf.@printf("Runtime:          %.4f s\n", result.opt_result.runtime_seconds)
    Printf.@printf("Best Cost (Loss): %.6e\n", result.opt_result.best_cost)
    println("-"^60)
    println("Optimized Parameters:")
    for (name, val) in zip(result.opt_result.param_names, result.opt_result.best_params)
        Printf.@printf("  %-15s : %.6e\n", string(name), val)
    end
    println("-"^60)
    println("Goodness of Fit Metrics:")
    Printf.@printf("  R² (Coeff of Det): %.6f\n", result.metrics[:r2])
    Printf.@printf("  Adjusted R²:      %.6f\n", result.metrics[:adj_r2])
    Printf.@printf("  RMSE:             %.6e\n", result.metrics[:rmse])
    Printf.@printf("  MAE:              %.6e\n", result.metrics[:mae])
    Printf.@printf("  AIC / BIC:        %.2f / %.2f\n", result.metrics[:aic], result.metrics[:bic])
    println("="^60 * "\n")
    
    # Optional exports
    if haskey(parsed_args, :out_json)
        json_path = parsed_args[:out_json]
        export_json(json_path, result)
        println("Saved JSON results to: $json_path")
    end
    
    if haskey(parsed_args, :out_report)
        rep_path = parsed_args[:out_report]
        export_latex_report(rep_path, result; dataset_name=basename(data_file), algorithm_name=uppercase(algo_str))
        println("Saved LaTeX report to: $rep_path")
    end
    
    return 0
end

function parse_cli_args(args::Vector{String})
    dict = Dict{Symbol, String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key_str = replace(arg[3:end], "-" => "_")
            key = Symbol(key_str)
            if i + 1 <= length(args) && !startswith(args[i+1], "--")
                dict[key] = args[i+1]
                i += 2
            else
                dict[key] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return dict
end

function parse_bounds_spec(str::String)
    # Accepts "0..1, 0.1..10" or "A:0..1, tau:0.1..10" or "0:1, 0.1:10"
    cleaned = replace(str, "[" => "", "]" => "", " " => "")
    intervals = split(cleaned, ",")
    lb = Float64[]
    ub = Float64[]
    named_map = Dict{Symbol, Tuple{Float64, Float64}}()
    
    for interv in intervals
        isempty(interv) && continue
        
        # Check if named bound "name:lower..upper"
        if contains(interv, ":") && !contains(interv, "..")
            parts = split(interv, ":")
            if length(parts) == 2 && tryparse(Float64, parts[1]) !== nothing
                # anonymous interval "0:1"
                push!(lb, parse(Float64, parts[1]))
                push!(ub, parse(Float64, parts[2]))
            else
                error("Malformed bound '$interv'. Use '0..1' or 'name:0..1'.")
            end
        elseif contains(interv, ":") && contains(interv, "..")
            parts = split(interv, ":")
            name_sym = Symbol(parts[1])
            range_parts = split(parts[2], "..")
            l_val = parse(Float64, range_parts[1])
            u_val = parse(Float64, range_parts[2])
            push!(lb, l_val)
            push!(ub, u_val)
            named_map[name_sym] = (l_val, u_val)
        elseif contains(interv, "..")
            range_parts = split(interv, "..")
            push!(lb, parse(Float64, range_parts[1]))
            push!(ub, parse(Float64, range_parts[2]))
        else
            error("Cannot parse interval '$interv'. Use format 'lower..upper' or 'name:lower..upper'.")
        end
    end
    return lb, ub, named_map
end

function align_named_bounds(pnames::Vector{Symbol}, named_map::Dict{Symbol, Tuple{Float64, Float64}}, default_lb::Vector{Float64}, default_ub::Vector{Float64})
    dim = length(pnames)
    lb = zeros(Float64, dim)
    ub = zeros(Float64, dim)
    for (i, p) in enumerate(pnames)
        if haskey(named_map, p)
            lb[i], ub[i] = named_map[p]
        elseif i <= length(default_lb)
            lb[i], ub[i] = default_lb[i], default_ub[i]
        else
            error("Missing bound for parameter '$p'")
        end
    end
    return lb, ub
end

function print_cli_help()
    println("""
EdiBot.jl - High-Performance Analytical & Physical Curve Fitting CLI

Usage:
  julia --project bin/edibot [options]

Required Options:
  --data <file>          Path to dataset file (.dat, .csv, .txt)
  --bounds <specs>       Parameter bounds (e.g. "0..1, 0.1..10, 0.1..2" or "A:0..1, tau:0.1..10, beta:0.1..2")
  --model <name>         Built-in model (:vft, :vft_log10, :kww, :hs, :yukawa, :gaussian, :sigmoid, etc.)
     OR
  --expr <formula>       Analytical formula string (e.g. "A * exp(-(x / tau)^beta)")

Algorithm & Loss Options:
  --algo <name>          Optimization algorithm: pso (default), de, ga
  --loss <name>          Loss function: mse (default), rmse, mae, log_mse, weighted
  --max-iters <int>      Maximum iterations (default: 500)
  --pop-size <int>       Population / Swarm size (default: 50)
  --seed <int>           Random seed for reproducibility
  --verbose              Print iteration progress

Output Options:
  --out-json <file>      Export optimization metrics and convergence to JSON
  --out-report <file>    Export LaTeX summary table and report

Examples:
  # Fit KWW relaxation data using PSO:
  julia --project bin/edibot --data data.dat --model kww --algo pso --bounds "0.5..1.5, 0.1..20, 0.1..1.0"

  # Fit analytical formula using Differential Evolution (DE) with named bounds:
  julia --project bin/edibot --data data.dat --expr "A * exp(-(x / tau)^beta)" --algo de --bounds "A:0.5..1.5, tau:0.5..10.0, beta:0.2..1.0"
""")
end
