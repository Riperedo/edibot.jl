# src/models/relaxation.jl

"""
    vft(T, p)
    vft(T, tau0, B, T0)

Vogel-Fulcher-Tammann (VFT) relaxation equation:
``\\tau(T) = \\tau_0 \\exp\\left( \\frac{B}{T - T_0} \\right)``

# Parameters `p = [tau0, B, T0]`
- `p[1] = tau0`: High-temperature limit relaxation time / pre-exponential factor.
- `p[2] = B`: Pseudo-activation energy parameter (in temperature units, K).
- `p[3] = T0`: Vogel temperature (ideal glass transition temperature, K).
"""
function vft(T::Real, p::AbstractVector{<:Real})
    tau0, B, T0 = p[1], p[2], p[3]
    if T <= T0
        return 1e30
    end
    return tau0 * exp(B / (T - T0))
end

vft(T::Real, tau0::Real, B::Real, T0::Real) = vft(T, [tau0, B, T0])

"""
    vft_log10(T, p)

Base-10 logarithmic Vogel-Fulcher-Tammann equation:
``\\log_{10} \\tau(T) = \\log_{10}(\\tau_0) + \\frac{B}{\\ln(10) (T - T_0)}``

Directly fits data provided in logarithmic units (e.g., ``\\log_{10}(\\tau)`` or ``\\log_{10}(\\eta)``).
"""
function vft_log10(T::Real, p::AbstractVector{<:Real})
    log_tau0, B, T0 = p[1], p[2], p[3]
    if T <= T0
        return 100.0
    end
    return log_tau0 + (B / (log(10.0) * (T - T0)))
end

"""
    kww(t, p)
    kww(t, A, tau, beta, y0=0.0)

Kohlrausch-Williams-Watts (KWW) stretched exponential relaxation function:
``\\phi(t) = A \\exp\\left( -\\left(\\frac{t}{\\tau}\\right)^\\beta \\right) + y_0``

# Parameters `p = [A, tau, beta, y0]` (or `[A, tau, beta]`)
- `p[1] = A`: Amplitude.
- `p[2] = tau`: Characteristic relaxation time.
- `p[3] = beta`: Stretching exponent ``0 < \\beta \\le 1``.
- `p[4] = y0`: Baseline offset (optional, default: 0.0).
"""
function kww(t::Real, p::AbstractVector{<:Real})
    A = p[1]
    tau = p[2]
    beta = p[3]
    y0 = length(p) >= 4 ? p[4] : 0.0
    
    if t < 0.0 || tau <= 0.0 || beta <= 0.0
        return y0
    end
    arg = (t / tau)^beta
    return A * exp(-arg) + y0
end

kww(t::Real, A::Real, tau::Real, beta::Real, y0::Real = 0.0) = kww(t, [A, tau, beta, y0])

"""
    arrhenius(T, p)

Arrhenius thermal activation function:
``\\tau(T) = \\tau_0 \\exp\\left( \\frac{E_a}{T} \\right)``

# Parameters `p = [tau0, Ea]`
- `p[1] = tau0`: Pre-exponential factor.
- `p[2] = Ea`: Activation energy divided by Boltzmann constant ``E_a / k_B`` (in Kelvin).
"""
function arrhenius(T::Real, p::AbstractVector{<:Real})
    tau0, Ea = p[1], p[2]
    T <= 0.0 && return 1e30
    return tau0 * exp(Ea / T)
end
