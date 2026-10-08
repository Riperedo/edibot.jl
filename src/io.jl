# src/io.jl

"""
    load_dataset(filepath::String)

Loads independent variable `x` and dependent variable `y` (and optional weights `w`) from a text/data file (`.dat`, `.csv`, `.txt`).
Automatically ignores header comments starting with `#`, `%`, or `//` and handles comma, tab, or whitespace delimiters.
"""
function load_dataset(filepath::String)
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))
    
    xs = Float64[]
    ys = Float64[]
    ws = Float64[]
    has_weights = false
    
    for line in eachline(filepath)
        trimmed = strip(line)
        isempty(trimmed) && continue
        startswith(trimmed, "#") && continue
        startswith(trimmed, "%") && continue
        startswith(trimmed, "//") && continue
        
        # Replace commas with spaces
        clean_line = replace(trimmed, "," => " ")
        parts = split(clean_line)
        
        if length(parts) >= 2
            try
                x_val = parse(Float64, parts[1])
                y_val = parse(Float64, parts[2])
                push!(xs, x_val)
                push!(ys, y_val)
                
                if length(parts) >= 3
                    has_weights = true
                    push!(ws, parse(Float64, parts[3]))
                end
            catch
                # Skip unparseable lines (e.g. text headers)
                continue
            end
        end
    end
    
    isempty(xs) && throw(ArgumentError("No valid numeric data found in $filepath"))
    
    if has_weights && length(ws) == length(xs)
        return xs, ys, ws
    else
        return xs, ys, nothing
    end
end

"""
    export_json(filepath::String, result::FitResult)

Exports fit results, parameters, statistical metrics, and convergence trajectory to a JSON file.
"""
function export_json(filepath::String, result::FitResult)
    opt = result.opt_result
    
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"model\": \"", result.model_name, "\",")
        println(io, "  \"converged\": ", opt.converged, ",")
        println(io, "  \"termination_reason\": \"", opt.termination_reason, "\",")
        println(io, "  \"iterations\": ", opt.iterations, ",")
        println(io, "  \"runtime_seconds\": ", opt.runtime_seconds, ",")
        println(io, "  \"best_cost\": ", opt.best_cost, ",")
        
        # Parameters
        println(io, "  \"parameters\": {")
        n_p = length(opt.best_params)
        for i in 1:n_p
            comma = i < n_p ? "," : ""
            println(io, "    \"", opt.param_names[i], "\": ", opt.best_params[i], comma)
        end
        println(io, "  },")
        
        # Metrics
        println(io, "  \"metrics\": {")
        metric_keys = collect(keys(result.metrics))
        for (idx, k) in enumerate(metric_keys)
            comma = idx < length(metric_keys) ? "," : ""
            println(io, "    \"", k, "\": ", result.metrics[k], comma)
        end
        println(io, "  },")
        
        # History
        println(io, "  \"convergence_history\": [")
        n_hist = length(opt.history)
        for i in 1:n_hist
            comma = i < n_hist ? "," : ""
            println(io, "    ", opt.history[i], comma)
        end
        println(io, "  ]")
        println(io, "}")
    end
    return filepath
end

"""
    export_latex_report(
        filepath::String,
        result::FitResult;
        dataset_name::String = "Dataset",
        algorithm_name::String = "EdiBot Optimizer"
    )

Generates an academic-grade LaTeX summary table and standalone document containing optimization results.
"""
function export_latex_report(
    filepath::String,
    result::FitResult;
    dataset_name::String = "Dataset",
    algorithm_name::String = "EdiBot Optimizer"
)
    opt = result.opt_result
    m = result.metrics
    
    open(filepath, "w") do io
        println(io, "% EdiBot.jl Optimization & Curve Fitting Report")
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Fitting Summary for \\textbf{", dataset_name, "} using \\textbf{", result.model_name, "} (", algorithm_name, ")}")
        println(io, "\\begin{tabular}{ll}")
        println(io, "\\hline\\hline")
        println(io, "\\textbf{Optimization Parameter / Metric} & \\textbf{Value} \\\\")
        println(io, "\\hline")
        
        # Parameters
        for i in 1:length(opt.best_params)
            Printf.@printf(io, "Parameter \$%s\$ & %.6e \\\\\n", string(opt.param_names[i]), opt.best_params[i])
        end
        println(io, "\\hline")
        
        # Metrics
        Printf.@printf(io, "Best Loss (Cost) & %.6e \\\\\n", opt.best_cost)
        Printf.@printf(io, "Coefficient of Determination (\$R^2\$) & %.6f \\\\\n", m[:r2])
        Printf.@printf(io, "Adjusted \$R^2\$ & %.6f \\\\\n", m[:adj_r2])
        Printf.@printf(io, "RMSE & %.6e \\\\\n", m[:rmse])
        Printf.@printf(io, "MAE & %.6e \\\\\n", m[:mae])
        Printf.@printf(io, "AIC & %.2f \\\\\n", m[:aic])
        Printf.@printf(io, "BIC & %.2f \\\\\n", m[:bic])
        println(io, "\\hline")
        
        # Performance
        Printf.@printf(io, "Iterations & %d \\\\\n", opt.iterations)
        Printf.@printf(io, "Runtime & %.4f s \\\\\n", opt.runtime_seconds)
        println(io, "Status & ", opt.termination_reason, " \\\\")
        println(io, "\\hline\\hline")
        println(io, "\\end{tabular}")
        println(io, "\\label{tab:edibot_fit_", replace(dataset_name, r"[^a-zA-Z0-9]" => "_"), "}")
        println(io, "\\end{table}")
    end
    return filepath
end
