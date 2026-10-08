# src/models/structure_factor.jl

"""
    hard_sphere_sq(q, p)
    hard_sphere_sq(q, phi, sigma)

Analytical static structure factor ``S(q)`` for monodisperse hard spheres under the Percus-Yevick closure (Wertheim-Thiele analytical solution).

# Parameters `p = [phi, sigma]` (or `[phi, sigma, scale]`)
- `p[1] = phi`: Volume fraction ``\\phi \\in [0, 0.65]``.
- `p[2] = sigma`: Hard sphere particle diameter ``\\sigma`` (same unit as ``1/q``).
- `p[3] = scale`: Optional scaling prefactor (default: 1.0).
"""
function hard_sphere_sq(q::Real, p::AbstractVector{<:Real})
    phi = p[1]
    sigma = p[2]
    scale = length(p) >= 3 ? p[3] : 1.0

    if phi <= 0.0 || sigma <= 0.0
        return 1.0 * scale
    elseif phi >= 0.74 # Physical close packing limit
        return 0.0
    end

    k = q * sigma
    
    # Low-q asymptotic expansion
    if abs(k) < 1e-4
        s0 = ((1.0 - phi)^4) / ((1.0 + 2.0 * phi)^2)
        return s0 * scale
    end

    eta = phi
    one_minus_eta = 1.0 - eta
    alpha = (1.0 + 2.0 * eta)^2 / (one_minus_eta^4)
    beta = -6.0 * eta * (1.0 + 0.5 * eta)^2 / (one_minus_eta^4)
    gamma = 0.5 * eta * alpha

    sin_k, cos_k = sincos(k)
    k2 = k * k
    k3 = k2 * k
    k4 = k2 * k2
    k6 = k3 * k3

    term1 = alpha * k3 * (sin_k - k * cos_k)
    term2 = beta * k2 * (2.0 * k * sin_k - (k2 - 2.0) * cos_k - 2.0)
    term3 = gamma * ((4.0 * k3 - 24.0 * k) * sin_k - (k4 - 12.0 * k2 + 24.0) * cos_k + 24.0)

    # Dimensionless direct correlation function: -24 * eta * (...) / k^6
    c_k = -24.0 * eta * (term1 + term2 + term3) / k6

    sq = 1.0 / (1.0 - c_k)
    return sq * scale
end

hard_sphere_sq(q::Real, phi::Real, sigma::Real, scale::Real = 1.0) = hard_sphere_sq(q, [phi, sigma, scale])

"""
    yukawa_rpa_sq(q, p)

Screened Coulomb / repulsive Yukawa static structure factor in the Random Phase Approximation (RPA).

# Parameters `p = [phi, sigma, K, z]`
- `p[1] = phi`: Hard core volume fraction.
- `p[2] = sigma`: Hard core diameter.
- `p[3] = K`: Dimensionless contact energy ``\\beta \\epsilon``.
- `p[4] = z`: Dimensionless screening parameter ``\\kappa \\sigma``.
"""
function yukawa_rpa_sq(q::Real, p::AbstractVector{<:Real})
    phi = p[1]
    sigma = p[2]
    K = p[3]
    z = p[4]

    s_hs = hard_sphere_sq(q, [phi, sigma])
    c_hs = 1.0 - (1.0 / s_hs)

    k = q * sigma
    k2 = k * k
    z2 = z * z

    sin_k, cos_k = sincos(k)
    # Fourier transform of tail: beta * u_tail(q)
    c_yukawa = -24.0 * phi * K * (z * sin_k + k * cos_k) / (k * (k2 + z2))

    sq = 1.0 / (1.0 - (c_hs + c_yukawa))
    return max(0.0, sq)
end
