# src/loss.jl

"""
    MSE <: AbstractLoss

Mean Squared Error loss:
``L(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2``
"""
struct MSE <: AbstractLoss end

function (::MSE)(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    length(y_true) == length(y_pred) || throw(DimensionMismatch("Vector lengths must match."))
    N = length(y_true)
    N == 0 && return 0.0
    s = 0.0
    @inbounds for i in 1:N
        diff = y_true[i] - y_pred[i]
        s += diff * diff
    end
    return s / N
end

"""
    RMSE <: AbstractLoss

Root Mean Squared Error loss:
``L(y, \\hat{y}) = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2}``
"""
struct RMSE <: AbstractLoss end

function (::RMSE)(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    return sqrt(MSE()(y_true, y_pred))
end

"""
    MAE <: AbstractLoss

Mean Absolute Error loss:
``L(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|``
"""
struct MAE <: AbstractLoss end

function (::MAE)(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    length(y_true) == length(y_pred) || throw(DimensionMismatch("Vector lengths must match."))
    N = length(y_true)
    N == 0 && return 0.0
    s = 0.0
    @inbounds for i in 1:N
        s += abs(y_true[i] - y_pred[i])
    end
    return s / N
end

"""
    WeightedMSE{W} <: AbstractLoss

Weighted Mean Squared Error loss:
``L(y, \\hat{y}) = \\frac{\\sum_{i=1}^N w_i (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^N w_i}``

# Constructors
- `WeightedMSE(weights::AbstractVector{<:Real})`: Explicit vector of positive weights.
- `WeightedMSE(weight_fn::Function, x_data::AbstractVector{<:Real})`: Computes weights from `weight_fn(x)`.
- `WeightedMSE(; low_q_power::Real = 1.0, x_data::AbstractVector{<:Real})`: Automatically sets weights ``w_i = 1 / x_i^{\\text{low\\_q\\_power}}``.
"""
struct WeightedMSE{W<:AbstractVector{<:Real}} <: AbstractLoss
    weights::W

    function WeightedMSE(weights::AbstractVector{<:Real})
        all(weights .>= 0) || throw(ArgumentError("Weights must be non-negative."))
        sum(weights) > 0 || throw(ArgumentError("Sum of weights must be positive."))
        return new{typeof(weights)}(weights)
    end
end

function WeightedMSE(weight_fn::Function, x_data::AbstractVector{<:Real})
    w = [Float64(weight_fn(x)) for x in x_data]
    return WeightedMSE(w)
end

function WeightedMSE(; low_q_power::Real = 1.0, x_data::AbstractVector{<:Real})
    w = [1.0 / (abs(x) > 1e-12 ? abs(x)^low_q_power : 1e-12^low_q_power) for x in x_data]
    return WeightedMSE(w)
end

function (loss::WeightedMSE)(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    length(y_true) == length(y_pred) || throw(DimensionMismatch("Vector lengths must match."))
    length(y_true) == length(loss.weights) || throw(
        DimensionMismatch("Weights length ($(length(loss.weights))) must match data length ($(length(y_true))).")
    )
    N = length(y_true)
    N == 0 && return 0.0
    s_weighted = 0.0
    w_sum = 0.0
    @inbounds for i in 1:N
        w = loss.weights[i]
        diff = y_true[i] - y_pred[i]
        s_weighted += w * diff * diff
        w_sum += w
    end
    return s_weighted / w_sum
end

"""
    LogMSE <: AbstractLoss

Logarithmic Mean Squared Error loss:
``L(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^N (\\log_{10}(y_i) - \\log_{10}(\\hat{y}_i))^2``

Critical for physical processes spanning several orders of magnitude (e.g. relaxation times, viscosity in VFT).
Non-positive predicted values receive an adaptive penalty.
"""
struct LogMSE <: AbstractLoss
    eps::Float64
    penalty::Float64

    LogMSE(; eps::Float64 = 1e-12, penalty::Float64 = 1e6) = new(eps, penalty)
end

function (loss::LogMSE)(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    length(y_true) == length(y_pred) || throw(DimensionMismatch("Vector lengths must match."))
    N = length(y_true)
    N == 0 && return 0.0
    s = 0.0
    @inbounds for i in 1:N
        yt = y_true[i]
        yp = y_pred[i]
        if yp <= 0.0 || isnan(yp) || isinf(yp)
            s += loss.penalty + abs(yt)
        else
            log_yt = log10(max(yt, loss.eps))
            log_yp = log10(max(yp, loss.eps))
            diff = log_yt - log_yp
            s += diff * diff
        end
    end
    return s / N
end

"""
    CustomLoss{F} <: AbstractLoss

Wraps a user-defined function `(y_true, y_pred) -> Real`.
"""
struct CustomLoss{F} <: AbstractLoss
    fn::F
end

(loss::CustomLoss)(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real}) = loss.fn(y_true, y_pred)
