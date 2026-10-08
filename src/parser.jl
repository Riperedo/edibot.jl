# src/parser.jl

"""
    ParsedModel{F} <: Function

Callable wrapper for dynamically compiled mathematical expressions, ensuring world-age compatibility via `Base.invokelatest`.
"""
struct ParsedModel{F} <: Function
    fn::F
end

(m::ParsedModel)(x::Real, p::AbstractVector{<:Real}) = Base.invokelatest(m.fn, x, p)

"""
    parse_expression(expr_str::String; param_names::Vector{Symbol} = Symbol[])

Parses a mathematical formula string into a fast Julia callable `(x::Real, p::AbstractVector{<:Real}) -> Real`.

# Supported syntaxes:
1. Indexed parameters with variable `x`: e.g., `"p[1] * exp(-(x / p[2])^p[3]) + p[4]"`
2. Named parameters with variable `x`: e.g., `"a * exp(-b * x) + c"` with `param_names = [:a, :b, :c]`.
   If `param_names` is empty, unique symbols (excluding standard mathematical functions and `x`) are identified in order of appearance.

# Examples
```julia
fn, pnames = parse_expression("a * exp(-b * x) + c")
# pnames is [:a, :b, :c]
# fn(1.0, [2.0, 0.5, 1.0]) -> 2.0 * exp(-0.5 * 1.0) + 1.0
```
"""
function parse_expression(expr_str::String; param_names::Vector{Symbol} = Symbol[])
    parsed_ast = Meta.parse(expr_str)
    
    # Check if expression uses p[i] indexing directly
    uses_indexed_p = contains(expr_str, "p[")
    
    if uses_indexed_p
        max_idx = 1
        for m in eachmatch(r"p\[(\d+)\]", expr_str)
            idx = parse(Int, m.captures[1])
            max_idx = max(max_idx, idx)
        end
        pnames = isempty(param_names) ? [Symbol("p$i") for i in 1:max_idx] : param_names
        
        func_expr = :( (x, p) -> $parsed_ast )
        raw_fn = Core.eval(@__MODULE__, func_expr)
        return ParsedModel(raw_fn), pnames
    else
        # Named parameters
        if isempty(param_names)
            extracted = extract_symbols_in_order(parsed_ast)
            math_symbols = Set([
                :x, :pi, :e, :sin, :cos, :tan, :exp, :log, :log10, :log2, :sqrt, :cbrt,
                :abs, :sinh, :cosh, :tanh, :asin, :acos, :atan, :erf, :erfc, :sincos,
                :min, :max, :clamp, :+ , :-, :*, :/, :^, :%
            ])
            pnames = [s for s in extracted if !(s in math_symbols)]
        else
            pnames = param_names
        end
        
        # Replace each parameter symbol in the AST with p[i]
        param_map = Dict{Symbol, Expr}(pnames[i] => :(p[$i]) for i in 1:length(pnames))
        transformed_ast = substitute_symbols(parsed_ast, param_map)
        
        func_expr = :( (x, p) -> $transformed_ast )
        raw_fn = Core.eval(@__MODULE__, func_expr)
        return ParsedModel(raw_fn), pnames
    end
end

function extract_symbols_in_order(ast)
    symbols = Symbol[]
    seen = Set{Symbol}()
    _extract_symbols_in_order!(symbols, seen, ast)
    return symbols
end

function _extract_symbols_in_order!(symbols::Vector{Symbol}, seen::Set{Symbol}, ast)
    if ast isa Symbol
        if !(ast in seen)
            push!(symbols, ast)
            push!(seen, ast)
        end
    elseif ast isa Expr
        for arg in ast.args
            _extract_symbols_in_order!(symbols, seen, arg)
        end
    end
end

function substitute_symbols(ast, param_map::Dict{Symbol, Expr})
    if ast isa Symbol
        return get(param_map, ast, ast)
    elseif ast isa Expr
        new_args = [substitute_symbols(arg, param_map) for arg in ast.args]
        return Expr(ast.head, new_args...)
    else
        return ast
    end
end
