# src/models/standard.jl

"""
    gaussian(x, p)
    gaussian(x, A, mu, sigma, y0=0.0)

Gaussian / Normal peak function:
``f(x) = A \\exp\\left( -\\frac{(x - \\mu)^2}{2 \\sigma^2} \\right) + y_0``
"""
function gaussian(x::Real, p::AbstractVector{<:Real})
    A = p[1]
    mu = p[2]
    sigma = p[3]
    y0 = length(p) >= 4 ? p[4] : 0.0
    sigma <= 0.0 && return y0
    diff = x - mu
    return A * exp(-0.5 * (diff / sigma)^2) + y0
end

"""
    lorentzian(x, p)
    lorentzian(x, A, x0, gamma, y0=0.0)

Lorentzian / Cauchy peak function:
``f(x) = \\frac{A}{\\pi} \\frac{\\gamma / 2}{(x - x_0)^2 + (\\gamma / 2)^2} + y_0``
"""
function lorentzian(x::Real, p::AbstractVector{<:Real})
    A = p[1]
    x0 = p[2]
    gamma = p[3]
    y0 = length(p) >= 4 ? p[4] : 0.0
    gamma <= 0.0 && return y0
    half_g = 0.5 * gamma
    diff = x - x0
    return (A / pi) * (half_g / (diff * diff + half_g * half_g)) + y0
end

"""
    power_law(x, p)

Power law function:
``f(x) = A x^k + y_0``
"""
function power_law(x::Real, p::AbstractVector{<:Real})
    A = p[1]
    k = p[2]
    y0 = length(p) >= 3 ? p[3] : 0.0
    x <= 0.0 && return y0
    return A * (x^k) + y0
end

"""
    exponential_decay(x, p)

Exponential decay function:
``f(x) = A \\exp(-x / \\tau) + y_0``
"""
function exponential_decay(x::Real, p::AbstractVector{<:Real})
    A = p[1]
    tau = p[2]
    y0 = length(p) >= 3 ? p[3] : 0.0
    tau <= 0.0 && return y0
    return A * exp(-x / tau) + y0
end

"""
    sigmoid(x, p)

Logistic sigmoid function:
``f(x) = \\frac{L}{1 + \\exp(-k (x - x_0))} + y_0``
"""
function sigmoid(x::Real, p::AbstractVector{<:Real})
    L = p[1]
    k = p[2]
    x0 = p[3]
    y0 = length(p) >= 4 ? p[4] : 0.0
    return (L / (1.0 + exp(-k * (x - x0)))) + y0
end

"""
    polynomial(x, p)

General polynomial function ``f(x) = \\sum_{i=1}^n p_i x^{i-1}``.
"""
function polynomial(x::Real, p::AbstractVector{<:Real})
    res = 0.0
    x_pow = 1.0
    @inbounds for c in p
        res += c * x_pow
        x_pow *= x
    end
    return res
end
