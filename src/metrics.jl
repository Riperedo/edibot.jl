# src/metrics.jl

"""
    evaluate_metrics(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real}, n_params::Int)

Calculates a comprehensive dictionary of statistical goodness-of-fit metrics:
- `:r2`: Coefficient of determination ``R^2``
- `:adj_r2`: Adjusted ``R^2``
- `:rmse`: Root Mean Squared Error
- `:mae`: Mean Absolute Error
- `:aic`: Akaike Information Criterion
- `:bic`: Bayesian Information Criterion
- `:chisq`: Total sum of squared residuals ``\\sum (y_i - \\hat{y}_i)^2``
- `:reduced_chisq`: ``\\chi^2 / (N - k)`` degrees of freedom
"""
function evaluate_metrics(
    y_true::AbstractVector{<:Real},
    y_pred::AbstractVector{<:Real},
    n_params::Int
)
    N = length(y_true)
    N == length(y_pred) || throw(DimensionMismatch("y_true and y_pred must have identical lengths."))
    
    T = promote_type(eltype(y_true), eltype(y_pred), Float64)
    
    if N == 0
        return Dict{Symbol, T}(
            :r2 => T(NaN), :adj_r2 => T(NaN), :rmse => T(NaN), :mae => T(NaN),
            :aic => T(NaN), :bic => T(NaN), :chisq => T(NaN), :reduced_chisq => T(NaN)
        )
    end
    
    y_mean = Statistics.mean(y_true)
    ss_tot = sum((y - y_mean)^2 for y in y_true)
    ss_res = sum((y_true[i] - y_pred[i])^2 for i in 1:N)
    sa_res = sum(abs(y_true[i] - y_pred[i]) for i in 1:N)
    
    r2 = ss_tot > 0.0 ? 1.0 - (ss_res / ss_tot) : 1.0
    
    dof = N - n_params
    adj_r2 = (N > n_params + 1 && ss_tot > 0.0) ? 1.0 - ((1.0 - r2) * (N - 1) / (N - n_params - 1)) : r2
    
    mse_val = ss_res / N
    rmse = sqrt(mse_val)
    mae = sa_res / N
    
    # Information criteria (Gaussian likelihood formulation)
    safe_mse = max(mse_val, 1e-30)
    aic = N * log(safe_mse) + 2 * n_params
    bic = N * log(safe_mse) + n_params * log(N)
    
    reduced_chisq = dof > 0 ? (ss_res / dof) : ss_res
    
    return Dict{Symbol, T}(
        :r2 => T(r2),
        :adj_r2 => T(adj_r2),
        :rmse => T(rmse),
        :mae => T(mae),
        :aic => T(aic),
        :bic => T(bic),
        :chisq => T(ss_res),
        :reduced_chisq => T(reduced_chisq)
    )
end
